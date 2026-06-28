"""
MRXD3000 Predict — Alert Engine v3
- Only fires for UPCOMING matches (kickoff in next 3 hours)
- Never sends duplicate alerts for the same match
- No fallback to past matches
"""
import logging
import os
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.models.match import Match
from app.models.prediction import Prediction, Alert
from app.services.alert_service import build_alert, format_telegram_message, send_telegram_alert

logger = logging.getLogger(__name__)
BOT_NAME = "MRXD3000 Predict"

SPORT_KEYS = ["soccer_epl", "soccer_fifa_world_cup"]

DEMO_ODDS = {
    "williamhill": {"home": 2.15, "draw": 3.50, "away": 3.15, "over25": 1.88,
                    "under25": 1.92, "corners_over_9_5": 1.85, "corners_under_9_5": 1.95},
    "1xbet": {"home": 2.20, "draw": 3.55, "away": 3.30, "over25": 1.90,
              "under25": 1.90, "corners_over_9_5": 1.87, "corners_under_9_5": 1.93},
}


async def fetch_live_odds() -> dict:
    import httpx
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key or api_key == "your_odds_api_key_here":
        logger.warning("ODDS_API_KEY not set — using demo odds")
        return {}

    all_odds = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        for sport_key in SPORT_KEYS:
            try:
                response = await client.get(
                    f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                    params={
                        "apiKey": api_key,
                        "regions": "eu,uk",
                        "markets": "h2h,totals",
                        "oddsFormat": "decimal",
                    }
                )
                response.raise_for_status()
                remaining = response.headers.get("x-requests-remaining", "?")
                logger.info(f"Odds API [{sport_key}] remaining: {remaining}")
                data = response.json()

                for event in data:
                    home = event.get("home_team", "")
                    away = event.get("away_team", "")
                    match_key = f"{home} vs {away}"
                    odds_by_bk = {}

                    for bookmaker in event.get("bookmakers", []):
                        bk_key = bookmaker.get("key", "")
                        display_key = "1xbet" if bk_key == "onexbet" else bk_key
                        bk_odds = {}

                        for market in bookmaker.get("markets", []):
                            if market["key"] == "h2h":
                                for outcome in market.get("outcomes", []):
                                    name = outcome["name"]
                                    price = outcome["price"]
                                    if name == home: bk_odds["home"] = price
                                    elif name == away: bk_odds["away"] = price
                                    else: bk_odds["draw"] = price
                            elif market["key"] == "totals":
                                for outcome in market.get("outcomes", []):
                                    name = outcome["name"].lower()
                                    point = outcome.get("point", 0)
                                    price = outcome["price"]
                                    if abs(point - 2.5) < 0.1:
                                        if "over" in name: bk_odds["over25"] = price
                                        elif "under" in name: bk_odds["under25"] = price
                                    if abs(point - 9.5) < 0.1:
                                        if "over" in name: bk_odds["corners_over_9_5"] = price
                                        elif "under" in name: bk_odds["corners_under_9_5"] = price

                        if bk_odds:
                            odds_by_bk[display_key] = bk_odds

                    if odds_by_bk:
                        all_odds[match_key] = odds_by_bk

            except Exception as e:
                logger.error(f"Odds API failed for {sport_key}: {e}")

    logger.info(f"Live odds fetched for {len(all_odds)} matches")
    return all_odds


def _find_odds(live_odds: dict, home_name: str, away_name: str) -> dict:
    if not live_odds:
        return DEMO_ODDS
    for match_key, odds in live_odds.items():
        key_lower = match_key.lower()
        if home_name.lower() in key_lower and away_name.lower() in key_lower:
            return odds
    return DEMO_ODDS


async def _already_alerted(db: AsyncSession, match_id: int) -> bool:
    """Check if we already sent a Telegram alert for this match today."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.match_id == match_id,
                Alert.sent_telegram == True,
                Alert.created_at >= today_start,
            )
        )
    )
    return result.scalar_one_or_none() is not None


async def run_alert_engine(db: AsyncSession) -> dict:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    live_odds = await fetch_live_odds()

    now = datetime.utcnow()
    window_start = now  # must be in the future
    window_end = now + timedelta(hours=3)

    # ONLY upcoming matches — no past matches ever
    result = await db.execute(
        select(Prediction)
        .join(Match, Match.id == Prediction.match_id)
        .options(
            selectinload(Prediction.match).selectinload(Match.home_team),
            selectinload(Prediction.match).selectinload(Match.away_team),
        )
        .where(Match.match_date > window_start)   # strictly future
        .where(Match.match_date <= window_end)    # within 3 hours
        .where(Match.status == "NS")              # not started only
        .order_by(Match.match_date.asc())
    )
    predictions = result.scalars().all()

    if not predictions:
        return {
            "message": "No upcoming matches in the next 3 hours",
            "alerts_sent": 0,
            "checked_window": f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} UTC",
        }

    alerts_sent = value_found = no_value = skipped_duplicate = 0
    errors = []

    for pred in predictions:
        match = pred.match
        if not match:
            continue

        # Skip if we already sent an alert for this match today
        already_sent = await _already_alerted(db, match.id)
        if already_sent:
            skipped_duplicate += 1
            logger.info(f"Skipping duplicate alert for {match.id}")
            continue

        home_name = match.home_team.name if match.home_team else "Home"
        away_name = match.away_team.name if match.away_team else "Away"
        league = match.league_name or "Premier League"
        odds_by_bookmaker = _find_odds(live_odds, home_name, away_name)

        is_wc_proxy = pred.model_version == "wc_style_proxy_v1"

        prediction_dict = {
            "corners_over_9_5_prob": pred.confidence if is_wc_proxy else 0.52,
            "corners_over_10_5_prob": 0.42,
            "corners_ft_pred": pred.corners_ft_pred or 10.0,
        }

        if not is_wc_proxy:
            # Only include outcome/goals markets when we have a real model behind them
            prediction_dict["home_win_prob"] = pred.home_win_prob or 0.0
            prediction_dict["draw_prob"] = pred.draw_prob or 0.0
            prediction_dict["away_win_prob"] = pred.away_win_prob or 0.0
            prediction_dict["over_25_prob"] = pred.over_25_prob or 0.0
            prediction_dict["under_25_prob"] = 1.0 - (pred.over_25_prob or 0.0)

        try:
            alert = build_alert(
                home_team=home_name,
                away_team=away_name,
                match_date=match.match_date or now,
                league=league,
                prediction=prediction_dict,
                odds_by_bookmaker=odds_by_bookmaker,
            )

            if not alert:
                no_value += 1
                continue

            value_found += 1
            message = _add_header(format_telegram_message(alert, prediction_dict))

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

            if bot_token and chat_id and bot_token != "your_telegram_bot_token_here":
                sent = await send_telegram_alert(message, bot_token, chat_id)
                if sent:
                    db_alert.sent_telegram = True
                    alerts_sent += 1
                    logger.info(f"Alert sent: {home_name} vs {away_name}")
                else:
                    errors.append(f"Telegram failed: {home_name} vs {away_name}")
            else:
                alerts_sent += 1

        except Exception as e:
            logger.error(f"Alert error {home_name} vs {away_name}: {e}")
            errors.append(str(e))

    await db.commit()

    return {
        "bot": BOT_NAME,
        "window": f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} UTC",
        "live_odds_loaded": len(live_odds) > 0,
        "upcoming_matches_found": len(predictions),
        "skipped_already_alerted": skipped_duplicate,
        "value_found": value_found,
        "no_value": no_value,
        "alerts_sent": alerts_sent,
        "errors": errors[:5],
    }


def _add_header(message: str) -> str:
    return f"🤖 <b>{BOT_NAME}</b>\n{'─'*30}\n" + message
