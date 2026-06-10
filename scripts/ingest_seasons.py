#!/usr/bin/env python3
"""
Ingest historical seasons 2022 and 2023 for Premier League.
Run this once to bulk up training data.
Uses ~4 API calls total.
"""
import asyncio, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SEASONS = [
    {"season": 2022, "from_date": "2022-08-01", "to_date": "2023-05-31"},
    {"season": 2023, "from_date": "2023-08-01", "to_date": "2024-05-31"},
]
LEAGUES = [39, 140, 135, 78]  # EPL, La Liga, Serie A, Bundesliga

async def run():
    from app.database import AsyncSessionLocal
    from data_pipeline.ingestion.match_ingestion import MatchIngestion
    api_key = os.getenv("API_FOOTBALL_KEY")

    async with AsyncSessionLocal() as session:
        ing = MatchIngestion(session, api_key)
        total = 0
        for s in SEASONS:
            for league_id in LEAGUES:
                try:
                    # Temporarily override season and dates
                    from data_pipeline.connectors.api_football import APIFootballClient
                    fixtures = await ing.client.get_fixtures(
                        league_id=league_id,
                        season=s["season"],
                        from_date=s["from_date"],
                        to_date=s["to_date"],
                    )
                    logger.info(f"Season {s['season']} league {league_id}: {len(fixtures)} fixtures available")

                    # Ingest teams first
                    teams_data = await ing.client.get_teams(league_id, s["season"])
                    for item in teams_data:
                        t = item.get("team", {})
                        await ing.writer.upsert_team(
                            api_id=t["id"], name=t["name"],
                            logo_url=t.get("logo"), country=t.get("country"),
                            league_id=league_id,
                        )

                    # Ingest fixtures
                    from data_pipeline.storage.db_writer import DBWriter
                    from data_pipeline.ingestion.match_ingestion import _parse_datetime, _safe_int, TRACKED_LEAGUES
                    count = 0
                    for fixture in fixtures:
                        f = fixture.get("fixture", {})
                        teams = fixture.get("teams", {})
                        goals = fixture.get("goals", {})
                        score = fixture.get("score", {})
                        league = fixture.get("league", {})
                        home_api_id = teams.get("home", {}).get("id")
                        away_api_id = teams.get("away", {}).get("id")
                        if not home_api_id or not away_api_id: continue
                        home = await ing.writer.upsert_team(api_id=home_api_id, name=teams["home"]["name"], logo_url=teams["home"].get("logo"), league_id=league_id)
                        away = await ing.writer.upsert_team(api_id=away_api_id, name=teams["away"]["name"], logo_url=teams["away"].get("logo"), league_id=league_id)
                        match_date = _parse_datetime(f["date"])
                        status = f.get("status", {}).get("short", "NS")
                        ht = score.get("halftime", {})
                        await ing.writer.upsert_match(
                            api_id=f["id"], home_team_id=home.id, away_team_id=away.id,
                            league_id=league_id,
                            league_name=TRACKED_LEAGUES.get(league_id, league.get("name")),
                            season=str(s["season"]), match_date=match_date, status=status,
                            venue=f.get("venue", {}).get("name"), referee=f.get("referee"),
                            home_score=_safe_int(goals.get("home")),
                            away_score=_safe_int(goals.get("away")),
                            home_score_ht=_safe_int(ht.get("home")),
                            away_score_ht=_safe_int(ht.get("away")),
                        )
                        count += 1
                    await session.commit()
                    total += count
                    logger.info(f"  ✅ Ingested {count} fixtures")
                except Exception as e:
                    logger.error(f"Failed season {s['season']} league {league_id}: {e}")
        await ing.close()
        logger.info(f"🏁 Total fixtures ingested: {total}")

asyncio.run(run())
