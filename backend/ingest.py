#!/usr/bin/env python3
"""
Day 2 — Data Ingestion Script
Run this to pull match data and odds into the database.

Usage:
    python scripts/ingest.py              # ingest fixtures + odds for all leagues
    python scripts/ingest.py --odds-only  # only fetch odds
    python scripts/ingest.py --league 39  # only Premier League
"""
import argparse
import asyncio
import logging
import os
import sys

# Make sure app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.database import AsyncSessionLocal
from data_pipeline.ingestion.match_ingestion import MatchIngestion, TRACKED_LEAGUES
from data_pipeline.ingestion.odds_ingestion import OddsIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


async def run_ingestion(league_id: int | None = None, odds_only: bool = False):
    api_football_key = os.getenv("API_FOOTBALL_KEY", "")
    odds_api_key = os.getenv("ODDS_API_KEY", "")

    if not api_football_key and not odds_only:
        logger.error("API_FOOTBALL_KEY not set in .env")
        sys.exit(1)
    if not odds_api_key:
        logger.warning("ODDS_API_KEY not set — skipping odds ingestion")

    async with AsyncSessionLocal() as session:
        # ─── Fixtures ────────────────────────────────────────
        if not odds_only and api_football_key:
            ingestion = MatchIngestion(session, api_football_key)
            try:
                if league_id:
                    logger.info(f"Ingesting league {league_id} only")
                    await ingestion.ingest_teams(league_id)
                    count = await ingestion.ingest_fixtures(league_id)
                    logger.info(f"✅ Fixtures ingested: {count}")
                else:
                    logger.info("Ingesting all tracked leagues...")
                    results = await ingestion.ingest_all_leagues(next_fixtures=5)
                    for league, data in results.items():
                        if "error" in data:
                            logger.error(f"  ❌ {league}: {data['error']}")
                        else:
                            logger.info(
                                f"  ✅ {league}: {data['teams']} teams, {data['fixtures']} fixtures"
                            )
            finally:
                await ingestion.close()

        # ─── Odds ────────────────────────────────────────────
        if odds_api_key:
            odds = OddsIngestion(session, odds_api_key)
            try:
                if league_id:
                    count = await odds.ingest_odds_for_league(league_id, is_opening=True)
                    logger.info(f"✅ Odds ingested: {count} records")
                else:
                    results = await odds.ingest_all_leagues(is_opening=True)
                    total = sum(results.values())
                    logger.info(f"✅ Total odds records ingested: {total}")
            finally:
                await odds.close()

    logger.info("🏁 Ingestion complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football data ingestion")
    parser.add_argument("--league", type=int, help="Only ingest a specific league ID")
    parser.add_argument("--odds-only", action="store_true", help="Only fetch odds")
    args = parser.parse_args()

    print("\nTracked leagues:")
    for lid, name in TRACKED_LEAGUES.items():
        print(f"  {lid}: {name}")
    print()

    asyncio.run(run_ingestion(league_id=args.league, odds_only=args.odds_only))
