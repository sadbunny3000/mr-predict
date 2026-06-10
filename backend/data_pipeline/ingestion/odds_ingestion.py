import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.match import Match
from app.models.team import Team
from data_pipeline.connectors.odds_api import OddsAPIClient, SPORT_KEYS
from data_pipeline.storage.db_writer import DBWriter

logger = logging.getLogger(__name__)

# Map our league IDs to Odds API sport keys
LEAGUE_TO_SPORT = {
    39: SPORT_KEYS["EPL"],
    140: SPORT_KEYS["LA_LIGA"],
    135: SPORT_KEYS["SERIE_A"],
    78: SPORT_KEYS["BUNDESLIGA"],
    61: SPORT_KEYS["LIGUE_1"],
    2: SPORT_KEYS["CHAMPIONS_LEAGUE"],
}


def _implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return round(1 / odds, 4) if odds and odds > 0 else 0.0


class OddsIngestion:
    def __init__(self, session: AsyncSession, api_key: str):
        self.session = session
        self.writer = DBWriter(session)
        self.client = OddsAPIClient(api_key)

    async def close(self):
        await self.client.close()

    async def _find_match(self, home_name: str, away_name: str) -> Match | None:
        """Try to find a match in DB by fuzzy team name matching."""
        result = await self.session.execute(
            select(Match, Team)
            .join(Team, Match.home_team_id == Team.id)
        )
        for match, home_team in result:
            if (
                home_name.lower() in home_team.name.lower()
                or home_team.name.lower() in home_name.lower()
            ):
                return match
        return None

    async def ingest_odds_for_league(
        self, league_id: int, is_opening: bool = False
    ) -> int:
        sport_key = LEAGUE_TO_SPORT.get(league_id)
        if not sport_key:
            logger.warning(f"No sport key for league {league_id}")
            return 0

        logger.info(f"Fetching odds for {sport_key}")
        odds_list = await self.client.get_odds(
            sport_key=sport_key,
            regions="eu",
            markets="h2h",
        )

        count = 0
        for event in odds_list:
            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")

            # Find the match in our DB
            match = await self._find_match(home_name, away_name)
            if not match:
                logger.debug(f"No match found for {home_name} vs {away_name}")
                continue

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("key", "unknown")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue

                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    home_odds = outcomes.get(home_name)
                    away_odds = outcomes.get(away_name)
                    draw_odds = outcomes.get("Draw")

                    if not home_odds or not away_odds:
                        continue

                    await self.writer.save_odds(
                        match_id=match.id,
                        bookmaker=bk_name,
                        market="h2h",
                        is_opening=is_opening,
                        home_odds=home_odds,
                        draw_odds=draw_odds,
                        away_odds=away_odds,
                    )
                    count += 1

        await self.session.commit()
        logger.info(f"Saved {count} odds records for league {league_id}")
        return count

    async def ingest_all_leagues(self, is_opening: bool = False) -> dict:
        """Fetch odds for all tracked leagues. Uses ~6 API calls."""
        results = {}
        for league_id in LEAGUE_TO_SPORT:
            try:
                count = await self.ingest_odds_for_league(league_id, is_opening)
                results[league_id] = count
            except Exception as e:
                logger.error(f"Odds ingestion failed for league {league_id}: {e}")
                results[league_id] = 0
        return results
