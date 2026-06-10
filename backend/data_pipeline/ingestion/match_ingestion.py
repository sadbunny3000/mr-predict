import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from data_pipeline.connectors.api_football import APIFootballClient
from data_pipeline.storage.db_writer import DBWriter
logger = logging.getLogger(__name__)
TRACKED_LEAGUES = {39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1", 2: "UEFA Champions League", 1: "World Cup"}
CURRENT_SEASON = 2025
def _parse_datetime(dt_str):
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)
def _safe_int(value):
    try: return int(value) if value is not None else None
    except: return None
def _safe_float(value):
    try:
        if isinstance(value, str): value = value.replace("%", "")
        return float(value) if value is not None else None
    except: return None
class MatchIngestion:
    def __init__(self, session, api_key):
        self.session = session
        self.writer = DBWriter(session)
        self.client = APIFootballClient(api_key)
    async def close(self): await self.client.close()
    async def ingest_teams(self, league_id):
        logger.info(f"Ingesting teams for league {league_id}")
        teams_data = await self.client.get_teams(league_id, CURRENT_SEASON)
        count = 0
        for item in teams_data:
            t = item.get("team", {})
            await self.writer.upsert_team(api_id=t["id"], name=t["name"], logo_url=t.get("logo"), country=t.get("country"), league_id=league_id, founded=t.get("founded"))
            count += 1
        await self.session.commit()
        logger.info(f"Ingested {count} teams for league {league_id}")
        return count
    async def ingest_fixtures(self, league_id, from_date='2024-08-01', to_date='2025-05-31'):
        logger.info(f"Ingesting fixtures for league {league_id}")
        fixtures = await self.client.get_fixtures(league_id=league_id, season=2024, from_date=from_date, to_date=to_date)
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
            home = await self.writer.upsert_team(api_id=home_api_id, name=teams["home"]["name"], logo_url=teams["home"].get("logo"), league_id=league_id)
            away = await self.writer.upsert_team(api_id=away_api_id, name=teams["away"]["name"], logo_url=teams["away"].get("logo"), league_id=league_id)
            match_date = _parse_datetime(f["date"])
            status = f.get("status", {}).get("short", "NS")
            ht = score.get("halftime", {})
            await self.writer.upsert_match(api_id=f["id"], home_team_id=home.id, away_team_id=away.id, league_id=league_id, league_name=TRACKED_LEAGUES.get(league_id, league.get("name")), season=str(CURRENT_SEASON), match_date=match_date, status=status, venue=f.get("venue", {}).get("name"), referee=f.get("referee"), home_score=_safe_int(goals.get("home")), away_score=_safe_int(goals.get("away")), home_score_ht=_safe_int(ht.get("home")), away_score_ht=_safe_int(ht.get("away")))
            count += 1
        await self.session.commit()
        logger.info(f"Ingested {count} fixtures for league {league_id}")
        return count
    async def ingest_all_leagues(self, last_fixtures=10):
        results = {}
        for league_id, name in TRACKED_LEAGUES.items():
            try:
                teams = await self.ingest_teams(league_id)
                fixtures = await self.ingest_fixtures(league_id, from_date='2024-08-01', to_date='2025-05-31')
                results[name] = {"teams": teams, "fixtures": fixtures}
            except Exception as e:
                logger.error(f"Failed to ingest {name}: {e}")
                results[name] = {"error": str(e)}
        return results
    async def ingest_fixture_stats(self, fixture_api_id):
        from sqlalchemy import select
        from app.models.match import Match
        from app.models.team import Team
        result = await self.session.execute(select(Match).where(Match.api_id == fixture_api_id))
        match = result.scalar_one_or_none()
        if not match: return False
        stats_data = await self.client.get_fixture_statistics(fixture_api_id)
        if not stats_data: return False
        for team_stats in stats_data:
            team_api_id = team_stats.get("team", {}).get("id")
            r2 = await self.session.execute(select(Team).where(Team.api_id == team_api_id))
            team = r2.scalar_one_or_none()
            if not team: continue
            is_home = team.id == match.home_team_id
            raw = {s["type"]: s["value"] for s in team_stats.get("statistics", [])}
            def si(v):
                try: return int(v) if v is not None else None
                except: return None
            def sf(v):
                try:
                    if isinstance(v, str): v = v.replace("%","")
                    return float(v) if v is not None else None
                except: return None
            await self.writer.upsert_match_stats(match_id=match.id, team_id=team.id, is_home=is_home, possession=sf(raw.get("Ball Possession")), passes_total=si(raw.get("Total passes")), passes_accurate=si(raw.get("Passes accurate")), pass_accuracy=sf(raw.get("Passes %")), shots_total=si(raw.get("Total Shots")), shots_on_target=si(raw.get("Shots on Goal")), corners_total=si(raw.get("Corner Kicks")), fouls=si(raw.get("Fouls")), yellow_cards=si(raw.get("Yellow Cards")), red_cards=si(raw.get("Red Cards")))
        await self.session.commit()
        return True
