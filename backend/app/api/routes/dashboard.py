"""
Cross-sport read endpoints for the picks UI: rolling accuracy and combined
alert history. Nothing here triggers any computation — pure reads.
"""
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import text

from app.database import get_db
from app.models.match import Match
from app.models.prediction import Prediction, Alert

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])


def _actual_football_outcome(match: Match):
    if match.home_score is None or match.away_score is None:
        return None
    if match.home_score > match.away_score:
        return "H"
    if match.away_score > match.home_score:
        return "A"
    return "D"


@router.get("/accuracy")
async def get_accuracy(db: AsyncSession = Depends(get_db)):
    """Rolling 30-day hit rate for each sport, plus total alerts sent.
    Tennis: winner_correct is set by the result-check scheduler once a match
    finishes. Football: accuracy is derived by comparing each sent alert's
    predicted outcome against the match's final score."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    tennis_row = (await db.execute(text('''
        SELECT COUNT(*), COUNT(*) FILTER (WHERE winner_correct = TRUE)
        FROM tennis_upcoming_matches
        WHERE result_checked = TRUE AND commence_time >= :cutoff
    '''), {'cutoff': cutoff})).fetchone()
    t_total, t_correct = tennis_row
    tennis_hit_rate = round((t_correct / t_total) * 100, 1) if t_total else None

    tennis_alerts_sent = (await db.execute(text('''
        SELECT COUNT(*) FROM tennis_upcoming_matches
        WHERE alert_sent = TRUE AND commence_time >= :cutoff
    '''), {'cutoff': cutoff})).scalar() or 0

    fb_result = await db.execute(
        select(Alert, Prediction, Match)
        .join(Prediction, Alert.prediction_id == Prediction.id)
        .join(Match, Prediction.match_id == Match.id)
        .where(Alert.sent_telegram == True, Match.status == "FT", Alert.created_at >= cutoff)
    )
    fb_total = 0
    fb_correct = 0
    for alert, pred, match in fb_result.all():
        actual = _actual_football_outcome(match)
        if actual is None:
            continue
        fb_total += 1
        if pred.predicted_outcome == actual:
            fb_correct += 1
    football_hit_rate = round((fb_correct / fb_total) * 100, 1) if fb_total else None

    football_alerts_sent = (await db.execute(
        select(func.count()).select_from(Alert)
        .where(Alert.sent_telegram == True, Alert.created_at >= cutoff)
    )).scalar() or 0

    return {
        "tennis_hit_rate_30d": tennis_hit_rate,
        "football_hit_rate_30d": football_hit_rate,
        "alerts_sent_30d": tennis_alerts_sent + football_alerts_sent,
    }


@router.get("/alerts/history")
async def get_alerts_history(limit: int = 20, db: AsyncSession = Depends(get_db)):
    """Combined tennis + football alert history, most recent first.
    Note: tennis entries use commence_time as a stand-in for 'sent_at' since
    the alert engine doesn't store the exact send timestamp separately —
    it's accurate to within a few hours, not to the second."""
    tennis_rows = (await db.execute(text('''
        SELECT p1_name, p2_name, confidence, commence_time, result_checked, winner_correct
        FROM tennis_upcoming_matches
        WHERE alert_sent = TRUE
        ORDER BY commence_time DESC
        LIMIT :limit
    '''), {'limit': limit})).fetchall()

    combined = []
    for p1, p2, confidence, commence_time, result_checked, winner_correct in tennis_rows:
        outcome = "pending" if not result_checked else ("win" if winner_correct else "loss")
        combined.append({
            "sport": "tennis",
            "match": f"{p1} vs {p2}",
            "confidence": confidence,
            "sent_at": commence_time,
            "outcome": outcome,
        })

    fb_result = await db.execute(
        select(Alert, Prediction, Match)
        .join(Prediction, Alert.prediction_id == Prediction.id)
        .join(Match, Prediction.match_id == Match.id)
        .options(selectinload(Match.home_team), selectinload(Match.away_team))
        .where(Alert.sent_telegram == True)
        .order_by(Alert.created_at.desc())
        .limit(limit)
    )
    for alert, pred, match in fb_result.all():
        if match.status != "FT":
            outcome = "pending"
        else:
            actual = _actual_football_outcome(match)
            outcome = "pending" if actual is None else ("win" if pred.predicted_outcome == actual else "loss")
        combined.append({
            "sport": "football",
            "match": f"{match.home_team.name} vs {match.away_team.name}",
            "confidence": alert.model_prob * 100 if alert.model_prob is not None else None,
            "sent_at": alert.created_at,
            "outcome": outcome,
        })

    combined.sort(key=lambda a: a["sent_at"], reverse=True)
    return combined[:limit]
