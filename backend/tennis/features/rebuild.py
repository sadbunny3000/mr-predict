"""
Runs both feature builders in sequence after new match data lands:
  1. Elo/serve-stat builder (async, walks tennis_matches)
  2. Rolling-window builder (sync psycopg2, run off the event loop)
Both are safe to re-run — everything upserts.
"""
import asyncio
import logging
import os
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from tennis.features.feature_builder import TennisEloBuilder
from tennis.features.rolling_features import build_rolling_features

logger = logging.getLogger(__name__)


def _get_sync_database_url() -> str:
    """psycopg2 needs a plain postgresql:// URL, not the asyncpg driver variant
    the rest of the app uses. Read the same real DATABASE_URL the async engine
    uses (not settings.database_url_sync, which is just an unused localhost
    fallback) and strip the driver suffix."""
    url = os.environ.get("DATABASE_URL", settings.database_url)
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


async def run_full_feature_rebuild(session: AsyncSession) -> dict:
    logger.info("Starting Elo/serve-stat rebuild")
    elo_matches = await TennisEloBuilder(session).build()

    logger.info("Starting rolling-window feature rebuild")
    rolling_result = await asyncio.to_thread(
        build_rolling_features, _get_sync_database_url()
    )

    return {
        "elo_matches_processed": elo_matches,
        "rolling_features": rolling_result,
    }
