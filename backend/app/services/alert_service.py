"""
Alert Engine
Compares model probabilities against bookmaker odds and generates
value bet alerts in the specified format.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Edge thresholds — lowered to catch 1-5% edges
BET_THIS_THRESHOLD = 0.03   # 3%+
CONSIDER_THRESHOLD = 0.01   # 1-2%

# Bookmakers we track
BOOKMAKERS = ["williamhill", "1xbet"]

BOOKMAKER_DISPLAY = {
    "williamhill": "William Hill",
    "1xbet": "1xBet",
    "onexbet": "1xBet",
}


@dataclass
class OddsLine:
    bookmaker: str
    outcome: str
    odds: float
    implied_prob: float
    edge: float
    rating: str  # BET_THIS, CONSIDER, NO_VALUE


@dataclass
class ValueAlert:
    match: str
    match_date: datetime
    league: str
    bet_this: list[OddsLine]
    consider: list[OddsLine]
    no_value: list[str]
    best_odds: dict


def implied_probability(odds: float) -> float:
    if not odds or odds <= 1.0:
        return 0.0
    return round(1 / odds, 4)


def fair_odds(prob: float) -> float:
    if not prob or prob <= 0:
        return 0.0
    return round(1 / prob, 2)


def edge(model_prob: float, implied_prob: float) -> float:
    return round(model_prob - implied_prob, 4)


def rate_edge(edge_val: float) -> str:
    if edge_val >= BET_THIS_THRESHOLD:
        return "BET_THIS"
    elif edge_val >= CONSIDER_THRESHOLD:
        return "CONSIDER"
    return "NO_VALUE"


def build_alert(
    home_team: str,
    away_team: str,
    match_date: datetime,
    league: str,
    prediction: dict,
    odds_by_bookmaker: dict,
) -> Optional[ValueAlert]:
    match_name = f"{home_team} vs {away_team}"

    markets = []
    if "home_win_prob" in prediction:
        markets.append(("Home Win", "home", prediction["home_win_prob"]))
        markets.append(("Draw", "draw", prediction["draw_prob"]))
        markets.append(("Away Win", "away", prediction["away_win_prob"]))
    if "over_25_prob" in prediction:
        markets.append(("Over 2.5 Goals", "over25", prediction["over_25_prob"]))
        markets.append(("Under 2.5 Goals", "under25", prediction.get("under_25_prob", 0)))
    if "corners_over_9_5_prob" in prediction:
        markets.append(("Over 9.5 Corners (90 min)", "corners_over_9_5", prediction["corners_over_9_5_prob"]))
    if "corners_over_10_5_prob" in prediction:
        markets.append(("Over 10.5 Corners (90 min)", "corners_over_10_5", prediction["corners_over_10_5_prob"]))

    bet_this = []
    consider = []
    no_value = []
    best_odds = {}

    for market_name, market_key, model_prob in markets:
        best_edge_val = -999
        best_bookmaker = None
        best_odds_val = None
        all_lines = []

        for bk_key, bk_odds in odds_by_bookmaker.items():
            bk_odds_val = bk_odds.get(market_key)
            if not bk_odds_val:
                continue
            imp_prob = implied_probability(bk_odds_val)
            edge_val = edge(model_prob, imp_prob)
            rating = rate_edge(edge_val)
            line = OddsLine(
                bookmaker=BOOKMAKER_DISPLAY.get(bk_key, bk_key),
                outcome=market_name,
                odds=bk_odds_val,
                implied_prob=imp_prob,
                edge=edge_val,
                rating=rating,
            )
            all_lines.append(line)
            if bk_odds_val > (best_odds_val or 0):
                best_odds_val = bk_odds_val
                best_bookmaker = BOOKMAKER_DISPLAY.get(bk_key, bk_key)
            if edge_val > best_edge_val:
                best_edge_val = edge_val

        if not all_lines:
            continue

        best_odds[market_key] = {"bookmaker": best_bookmaker, "odds": best_odds_val}

        if best_edge_val >= BET_THIS_THRESHOLD:
            bet_this.append(OddsLine(
                bookmaker=best_bookmaker,
                outcome=market_name,
                odds=best_odds_val,
                implied_prob=implied_probability(best_odds_val),
                edge=best_edge_val,
                rating="BET_THIS",
            ))
        elif best_edge_val >= CONSIDER_THRESHOLD:
            consider.append(OddsLine(
                bookmaker=best_bookmaker,
                outcome=market_name,
                odds=best_odds_val,
                implied_prob=implied_probability(best_odds_val),
                edge=best_edge_val,
                rating="CONSIDER",
            ))
        else:
            no_value.append(f"{market_name} — {round(best_edge_val*100, 1)}% edge, skip")

    if not bet_this and not consider:
        return None

    return ValueAlert(
        match=match_name,
        match_date=match_date,
        league=league,
        bet_this=bet_this,
        consider=consider,
        no_value=no_value,
        best_odds=best_odds,
    )


def format_telegram_message(alert: ValueAlert, prediction: dict) -> str:
    date_str = alert.match_date.strftime("%A %d %B, %I:%M %p")
    lines = []
    lines.append("🔥 VALUE BET ALERT")
    lines.append(f"Match: {alert.match}")
    lines.append(f"Date: {date_str}")
    lines.append(f"League: {alert.league}")
    lines.append("")

    corner_bets = [b for b in alert.bet_this + alert.consider if "Corner" in b.outcome]
    outcome_bets = [b for b in alert.bet_this + alert.consider if b.outcome in ("Home Win", "Draw", "Away Win")]
    goals_bets = [b for b in alert.bet_this + alert.consider if "Goal" in b.outcome]

    if corner_bets:
        lines.append("📊 CORNERS")
        for b in corner_bets:
            emoji = "✅ BET THIS" if b.rating == "BET_THIS" else "🟡 CONSIDER"
            lines.append(f"{b.outcome}")
            lines.append(f"Your model: {round((b.implied_prob + b.edge)*100)}% → Fair odds: {fair_odds(b.implied_prob + b.edge)}")
            lines.append(f"Bookmaker: {round(b.implied_prob*100)}% → Their odds: {b.odds}")
            lines.append(f"Edge: {round(b.edge*100, 1)}% {emoji}")
            lines.append(f"👉 Best odds: {b.bookmaker} at {b.odds}")
            lines.append("")

    if outcome_bets:
        lines.append("⚽ OUTCOME")
        for b in outcome_bets:
            emoji = "✅ BET THIS" if b.rating == "BET_THIS" else "🟡 CONSIDER"
            model_prob_pct = round((b.implied_prob + b.edge) * 100)
            lines.append(f"{b.outcome}")
            lines.append(f"Your model: {model_prob_pct}% → Fair odds: {fair_odds(b.implied_prob + b.edge)}")
            lines.append(f"Bookmaker: {round(b.implied_prob*100)}% → Their odds: {b.odds}")
            lines.append(f"Edge: {round(b.edge*100, 1)}% {emoji}")
            lines.append(f"👉 Best odds: {b.bookmaker} at {b.odds}")
            lines.append("")

    if goals_bets:
        lines.append("🎯 GOALS")
        for b in goals_bets:
            emoji = "✅ BET THIS" if b.rating == "BET_THIS" else "🟡 CONSIDER"
            model_prob_pct = round((b.implied_prob + b.edge) * 100)
            lines.append(f"{b.outcome}")
            lines.append(f"Your model: {model_prob_pct}% → Fair odds: {fair_odds(b.implied_prob + b.edge)}")
            lines.append(f"Bookmaker: {round(b.implied_prob*100)}% → Their odds: {b.odds}")
            lines.append(f"Edge: {round(b.edge*100, 1)}% {emoji}")
            lines.append(f"👉 Best odds: {b.bookmaker} at {b.odds}")
            lines.append("")

    if alert.no_value:
        lines.append("❌ NO VALUE")
        for nv in alert.no_value:
            lines.append(nv)

    return "\n".join(lines)


async def send_telegram_alert(message: str, bot_token: str, chat_id: str) -> bool:
    import httpx
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
            })
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False
