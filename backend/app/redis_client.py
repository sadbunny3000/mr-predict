import logging
from typing import Any

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return redis_client


async def close_redis() -> None:
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")


async def cache_set(key: str, value: str, ttl_seconds: int = 300) -> None:
    client = await get_redis()
    await client.setex(key, ttl_seconds, value)


async def cache_get(key: str) -> str | None:
    client = await get_redis()
    return await client.get(key)


async def cache_delete(key: str) -> None:
    client = await get_redis()
    await client.delete(key)
