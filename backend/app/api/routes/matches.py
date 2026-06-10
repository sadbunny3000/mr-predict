import json
import logging
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.match import Match
from app.redis_client import cache_get, cache_set
from app.schemas.match import MatchSchema

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("", response_model=list[MatchSchema])
async def get_matches(
    match_date: date | None = Query(default=None, description="Filter by date (YYYY-MM-DD)"),
    league_id: int | None = Query(default=None, description="Filter by league ID"),
    status: str | None = Query(default=None, description="NS, 1H, HT, 2H, FT"),
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    cache_key = f"matches:{match_date}:{league_id}:{status}:{limit}"
    cached = await cache_get(cache_key)
    if cached:
        return json.loads(cached)

    query = select(Match).options(
        selectinload(Match.home_team),
        selectinload(Match.away_team),
    )

    if match_date:
        start = datetime.combine(match_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        query = query.where(Match.match_date >= start, Match.match_date < end)
    else:
        # Default: today and next 3 days
        now = datetime.now(tz=timezone.utc)
        query = query.where(
            Match.match_date >= now,
            Match.match_date < now + timedelta(days=3),
        )

    if league_id:
        query = query.where(Match.league_id == league_id)
    if status:
        query = query.where(Match.status == status)

    query = query.order_by(Match.match_date).limit(limit)
    result = await db.execute(query)
    matches = result.scalars().all()

    data = [MatchSchema.model_validate(m).model_dump(mode="json") for m in matches]
    await cache_set(cache_key, json.dumps(data), ttl_seconds=120)
    return data


@router.get("/{match_id}", response_model=MatchSchema)
async def get_match(match_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Match)
        .options(selectinload(Match.home_team), selectinload(Match.away_team))
        .where(Match.id == match_id)
    )
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match
