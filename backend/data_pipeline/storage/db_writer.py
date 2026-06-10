import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.match import Match
from app.models.prediction import MatchStats, Odds
from app.models.team import Team

logger = logging.getLogger(__name__)


class DBWriter:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ─── Teams ───────────────────────────────────────────────

    async def upsert_team(self, api_id: int, name: str, **kwargs) -> Team:
        result = await self.session.execute(
            select(Team).where(Team.api_id == api_id)
        )
        team = result.scalar_one_or_none()
        if team:
            team.name = name
            for k, v in kwargs.items():
                if hasattr(team, k):
                    setattr(team, k, v)
        else:
            team = Team(api_id=api_id, name=name, **kwargs)
            self.session.add(team)
        await self.session.flush()
        return team

    # ─── Matches ─────────────────────────────────────────────

    async def upsert_match(
        self,
        api_id: int,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        match_date: datetime,
        **kwargs,
    ) -> Match:
        result = await self.session.execute(
            select(Match).where(Match.api_id == api_id)
        )
        match = result.scalar_one_or_none()
        if match:
            match.home_team_id = home_team_id
            match.away_team_id = away_team_id
            match.match_date = match_date
            for k, v in kwargs.items():
                if hasattr(match, k):
                    setattr(match, k, v)
        else:
            match = Match(
                api_id=api_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league_id=league_id,
                match_date=match_date,
                **kwargs,
            )
            self.session.add(match)
        await self.session.flush()
        return match

    # ─── Match Stats ─────────────────────────────────────────

    async def upsert_match_stats(
        self, match_id: int, team_id: int, is_home: bool, **kwargs
    ) -> MatchStats:
        result = await self.session.execute(
            select(MatchStats).where(
                MatchStats.match_id == match_id,
                MatchStats.team_id == team_id,
            )
        )
        stats = result.scalar_one_or_none()
        if stats:
            for k, v in kwargs.items():
                if hasattr(stats, k):
                    setattr(stats, k, v)
        else:
            stats = MatchStats(
                match_id=match_id, team_id=team_id, is_home=is_home, **kwargs
            )
            self.session.add(stats)
        await self.session.flush()
        return stats

    # ─── Odds ────────────────────────────────────────────────

    async def save_odds(
        self,
        match_id: int,
        bookmaker: str,
        market: str,
        is_opening: bool = False,
        **kwargs,
    ) -> Odds:
        odds = Odds(
            match_id=match_id,
            bookmaker=bookmaker,
            market=market,
            is_opening=is_opening,
            recorded_at=datetime.now(tz=timezone.utc),
            **kwargs,
        )
        self.session.add(odds)
        await self.session.flush()
        return odds
