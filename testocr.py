import io
import os
import re
import json
import hashlib
import asyncio
import tempfile
import uuid
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import aiohttp
import aiofiles
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from paddleocr import PaddleOCR
from langdetect import detect
import redis.asyncio as redis
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configuration
MAX_CONCURRENT_REQUESTS = 100
MAX_IMAGE_SIZE = 1280
CACHE_EXPIRY = 60 * 60 * 24  # 24 hours
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
TIMEOUT_SECONDS = 30

# Redis configuration from environment
REDISHOST = os.environ["REDISHOST"]
REDISPORT = int(os.environ["REDISPORT"])
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
ocr_models = {}
redis_client = None
executor = None

# Initialize resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_models, redis_client, executor
    
    logger.info("Initializing OCR models...")

    def init_models():
        return {
            'en': PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False),
            'fr': PaddleOCR(use_angle_cls=True, lang='fr', use_gpu=False, show_log=False),
            'ar': PaddleOCR(use_angle_cls=True, lang='arabic', use_gpu=False, show_log=False)
        }
    
    # Use thread pool for CPU-intensive OCR operations
    executor = ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_event_loop()
    ocr_models = await loop.run_in_executor(executor, init_models)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            redis_client = redis.Redis(
                host=REDISHOST,
                port=REDISPORT,
                password=REDIS_PASSWORD,
                db=0,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            await redis_client.ping()
            logger.info(f"Connected to Redis at {REDISHOST}:{REDISPORT}")
            break
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("Failed to connect to Redis after all attempts")
                raise
            await asyncio.sleep(2 ** attempt)

    logger.info("All resources initialized successfully")
    yield
    
    # Cleanup
    if redis_client:
        await redis_client.close()
    if executor:
        executor.shutdown(wait=True)
    logger.info("Resources cleaned up")

app = FastAPI(
    title="High-Performance OCR API",
    description="Scalable OCR API with caching and concurrent processing",
    version="1.0.0",
    lifespan=lifespan
)

# Semaphore to limit concurrent requests
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class OCRProcessor:
    @staticmethod
    def get_cache_key(image_url: str) -> str:
        return f"ocr:v2:{hashlib.md5(image_url.encode()).hexdigest()}"
    
    @staticmethod
    def detect_language(texts: list[str]) -> str:
        if not texts:
            return "unknown"
        full_text = ' '.join(texts)[:1000]
        try:
            return detect(full_text)
        except:
            return "unknown"

    @staticmethod
    def clean_words(words: list[str]) -> list[str]:
        return [w.strip() for w in words if len(w.strip()) > 1 and re.search(r'\w', w)]

    @staticmethod
    async def download_image(image_url: str, session: aiohttp.ClientSession) -> Image.Image:
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
            async with session.get(image_url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {resp.status}")
                content_length = resp.headers.get('content-length')
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail="Image file too large")
                content = await resp.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail="Image file too large")
                try:
                    image = Image.open(io.BytesIO(content))
                    return image.convert("RGB")
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid image format")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Image download timeout")
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")

    @staticmethod
    def resize_image(image: Image.Image) -> Image.Image:
        if max(image.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple([int(x * ratio) for x in image.size])
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def process_ocr_sync(image_path: str, models: Dict) -> Dict[str, Any]:
        best_result = None
        for lang, model in models.items():
            try:
                result = model.ocr(image_path, cls=True)
                if not result or not result[0]:
                    continue
                words = [line[1][0] for line in result[0] if line[1][1] > 0.5]
                words = OCRProcessor.clean_words(words)
                if not best_result or len(words) > len(best_result.get('words', [])):
                    text_area = sum([
                        (box[2][0] - box[0][0]) * (box[2][1] - box[0][1]) 
                        for box, _ in result[0]
                    ])
                    with Image.open(image_path) as img:
                        image_area = img.size[0] * img.size[1]
                    best_result = {
                        "words": words,
                        "text": ' '.join(words),
                        "lang": lang,
                        "word_count": len(words),
                        "text_area_percent": (text_area / image_area) * 100 if image_area > 0 else 0
                    }
            except Exception as e:
                logger.error(f"OCR error for lang {lang}: {str(e)}")
        return best_result or {"words": [], "text": "", "lang": "unknown", "word_count": 0, "text_area_percent": 0}

async def cleanup_temp_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Temp file cleanup error: {str(e)}")

@app.get("/ocr")
async def ocr_from_url(
    image_url: str = Query(..., description="URL of the image"),
    background_tasks: BackgroundTasks = None
):
    async with request_semaphore:
        start_time = time.time()
        cache_key = OCRProcessor.get_cache_key(image_url)
        try:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result["cached"] = True
                result["processing_time"] = time.time() - start_time
                return result
        except Exception as e:
            logger.warning(f"Cache error: {e}")

        async with aiohttp.ClientSession() as session:
            image = await OCRProcessor.download_image(image_url, session)
        image = OCRProcessor.resize_image(image)

        os.makedirs("/tmp/ocr", exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.jpg', delete=False, prefix=f'ocr_{uuid.uuid4().hex[:8]}_', dir='/tmp/ocr'
        )

        image.save(temp_file.name, 'JPEG', quality=85, optimize=True)

        ocr_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            OCRProcessor.process_ocr_sync,
            temp_file.name,
            ocr_models
        )

        if ocr_result["words"]:
            ocr_result["lang_detected"] = OCRProcessor.detect_language(ocr_result["words"])
        else:
            ocr_result["lang_detected"] = "unknown"

        ocr_result.update({
            "image_url": image_url,
            "cached": False,
            "processing_time": time.time() - start_time,
            "image_dimensions": image.size
        })

        try:
            await redis_client.setex(cache_key, CACHE_EXPIRY, json.dumps(ocr_result))
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_file.name)

        return ocr_result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await redis_client.ping()
        return {
            "status": "healthy",
            "redis": "connected",
            "REDISHOST": f"{REDISHOST}:{REDISPORT}",
            "models_loaded": len(ocr_models),
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "REDISHOST": f"{REDISHOST}:{REDISPORT}",
            "timestamp": time.time()
        }

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    try:
        # Get Redis info
        redis_info = await redis_client.info()
        
        return {
            "concurrent_limit": MAX_CONCURRENT_REQUESTS,
            "active_connections": redis_info.get('connected_clients', 0),
            "cache_keys": await redis_client.dbsize(),
            "memory_usage": redis_info.get('used_memory_human', 'unknown'),
            "models_available": list(ocr_models.keys()),
            "uptime": redis_info.get('uptime_in_seconds', 0),
            "REDISHOST": f"{REDISHOST}:{REDISPORT}"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "testocr:app",
        host="0.0.0.0",
        port=8000
    )