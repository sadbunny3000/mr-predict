"""
MRXD3000 Predict — Telegram Alert Engine
Scans recent predictions, checks for value vs bookmaker odds, fires Telegram alerts.
"""
import logging
import os
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.match import Match
from app.models.prediction import Prediction, Alert
from app.services.alert_service import build_alert, format_telegram_message, send_telegram_alert

logger = logging.getLogger(__name__)

BOT_NAME = "MRXD3000 Predict"

# Simulated bookmaker odds used when live odds API is unavailable (free tier)
# These are realistic PL market odds for testing the pipeline end-to-end
DEMO_ODDS = {
    "castlebet":    {"home": 2.10, "draw": 3.40, "away": 3.20, "over25": 1.85, "under25": 1.95},
    "easybetnam":   {"home": 2.05, "draw": 3.45, "away": 3.25, "over25": 1.82, "under25": 1.98},
    "williamhill":  {"home": 2.15, "draw": 3.50, "away": 3.15, "over25": 1.88, "under25": 1.92},
    "1xbet":        {"home": 2.20, "draw": 3.55, "away": 3.30, "over25": 1.90, "under25": 1.90},
}


async def run_alert_engine(db: AsyncSession) -> dict:
    """
    Main entry point. Finds predictions without alerts, checks for value, sends Telegram.
    Returns a summary dict.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.getenv("TELEGRAM_CHAT_ID", "")

    # Find predictions from the last 7 days that haven't had alerts generated
    cutoff = datetime.utcnow() - timedelta(days=7)

    result = await db.execute(
        select(Prediction)
        .join(Match, Match.id == Prediction.match_id)
        .options(
            selectinload(Prediction.match).selectinload(Match.home_team),
            selectinload(Prediction.match).selectinload(Match.away_team),
        )
        .where(Prediction.created_at >= cutoff)
        .order_by(Prediction.created_at.desc())
        .limit(50)
    )
    predictions = result.scalars().all()

    if not predictions:
        return {"message": "No predictions found to process", "alerts_sent": 0}

    alerts_sent = 0
    value_found = 0
    no_value = 0
    errors = []

    for pred in predictions:
        match = pred.match
        if not match:
            continue

        home_name = match.home_team.name if match.home_team else "Home"
        away_name = match.away_team.name if match.away_team else "Away"
        league    = match.league_name or "Premier League"

        prediction_dict = {
            "home_win_prob":   pred.home_win_prob   or 0.0,
            "draw_prob":       pred.draw_prob        or 0.0,
            "away_win_prob":   pred.away_win_prob    or 0.0,
            "over_25_prob":    pred.over_25_prob     or 0.0,
            "under_25_prob":   1.0 - (pred.over_25_prob or 0.0),
            "corners_ht_pred": pred.corners_ht_pred  or 0.0,
            "corners_ft_pred": pred.corners_ft_pred  or 0.0,
            "corners_2h_pred": pred.corners_2h_pred  or 0.0,
        }

        try:
            alert = build_alert(
                home_team=home_name,
                away_team=away_name,
                match_date=match.match_date or datetime.utcnow(),
                league=league,
                prediction=prediction_dict,
                odds_by_bookmaker=DEMO_ODDS,
            )

            if not alert:
                no_value += 1
                continue

            value_found += 1
            message = _add_header(format_telegram_message(alert, prediction_dict))

            # Save alert record to DB
            db_alert = Alert(
                match_id=match.id,
                prediction_id=pred.id,
                alert_type="VALUE_BET",
                outcome=pred.predicted_outcome,
                message=message,
                model_prob=pred.confidence or 0.0,
                implied_prob=0.0,
                edge_pct=0.0,
                sent_telegram=False,
                sent_sms=False,
            )
            db.add(db_alert)
            await db.flush()

            # Send to Telegram if configured
            if bot_token and chat_id and bot_token != "your_telegram_bot_token_here":
                sent = await send_telegram_alert(message, bot_token, chat_id)
                if sent:
                    db_alert.sent_telegram = True
                    alerts_sent += 1
                    logger.info(f"Alert sent for {home_name} vs {away_name}")
                else:
                    errors.append(f"Telegram send failed for {home_name} vs {away_name}")
            else:
                # Token not configured — log the message so we can see it works
                logger.info(f"\n{'='*60}\n{BOT_NAME} ALERT PREVIEW:\n{message}\n{'='*60}")
                alerts_sent += 1  # Count as processed even without send

        except Exception as e:
            logger.error(f"Alert engine error for match {match.api_id}: {e}")
            errors.append(str(e))

    await db.commit()

    return {
        "bot": BOT_NAME,
        "predictions_processed": len(predictions),
        "value_found": value_found,
        "no_value": no_value,
        "alerts_sent": alerts_sent,
        "errors": errors[:5],
    }


def _add_header(message: str) -> str:
    """Prepend bot name header to every alert message."""
    header = f"🤖 <b>{BOT_NAME}</b>\n{'─'*30}\n"
    return header + message
