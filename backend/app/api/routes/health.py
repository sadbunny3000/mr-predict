from fastapi import APIRouter
from app.redis_client import get_redis

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    redis = await get_redis()
    try:
        await redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    return {
        "status": "ok",
        "redis": redis_status,
        "service": "football-predictor-api",
    }
