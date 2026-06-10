"""
Extra features: Weather (Open-Meteo, free) + Referee tendencies (from DB).
These are fetched at prediction time and added to the feature vector.
"""
import logging
from datetime import datetime

import httpx
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


async def fetch_weather(latitude: float, longitude: float, match_date: datetime) -> dict:
    """
    Fetch weather forecast for a match using Open-Meteo (free, no API key needed).
    Returns temperature, precipitation, wind_speed.
    """
    date_str = match_date.strftime("%Y-%m-%d")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            daily = data.get("daily", {})
            return {
                "temperature": daily.get("temperature_2m_max", [None])[0] or 15.0,
                "precipitation": daily.get("precipitation_sum", [0])[0] or 0.0,
                "wind_speed": daily.get("windspeed_10m_max", [0])[0] or 0.0,
            }
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e} — using defaults")
        return {"temperature": 15.0, "precipitation": 0.0, "wind_speed": 0.0}


def get_referee_stats(db_url: str, referee_name: str) -> dict:
    """
    Compute referee tendencies from historical matches.
    Returns avg corners per game and avg cards per game for this referee.
    """
    if not referee_name:
        return {"referee_avg_corners": 10.0, "referee_avg_cards": 3.0}

    engine = create_engine(db_url)
    try:
        query = text("""
            SELECT
                COUNT(m.id) as match_count,
                AVG(ms_home.corners_total + ms_away.corners_total) as avg_corners,
                AVG(ms_home.yellow_cards + ms_away.yellow_cards +
                    ms_home.red_cards + ms_away.red_cards) as avg_cards
            FROM matches m
            LEFT JOIN match_stats ms_home ON m.id = ms_home.match_id AND ms_home.is_home = true
            LEFT JOIN match_stats ms_away ON m.id = ms_away.match_id AND ms_away.is_home = false
            WHERE m.referee ILIKE :referee
              AND m.status = 'FT'
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"referee": f"%{referee_name}%"}).fetchone()

        if result and result[0] and result[0] >= 3:
            return {
                "referee_avg_corners": float(result[1] or 10.0),
                "referee_avg_cards": float(result[2] or 3.0),
            }
    except Exception as e:
        logger.warning(f"Referee stats fetch failed: {e}")

    return {"referee_avg_corners": 10.0, "referee_avg_cards": 3.0}


# Stadium coordinates for Premier League venues
# Used for weather lookups
STADIUM_COORDS = {
    "Emirates Stadium": (51.5549, -0.1084),
    "Stamford Bridge": (51.4816, -0.1910),
    "Old Trafford": (53.4631, -2.2913),
    "Anfield": (53.4308, -2.9608),
    "Etihad Stadium": (53.4831, -2.2004),
    "Tottenham Hotspur Stadium": (51.6043, -0.0665),
    "St. James' Park": (54.9756, -1.6218),
    "Villa Park": (52.5090, -1.8847),
    "Goodison Park": (53.4388, -2.9662),
    "Molineux Stadium": (52.5901, -2.1302),
    "King Power Stadium": (52.6204, -1.1422),
    "Amex Stadium": (50.8616, -0.0837),
    "Selhurst Park": (51.3983, -0.0855),
    "Brentford Community Stadium": (51.4882, -0.2886),
    "Vitality Stadium": (50.7352, -1.8382),
    "Gtech Community Stadium": (51.4882, -0.2886),
    # Default London coords for unknown venues
    "default": (51.5074, -0.1278),
}


def get_stadium_coords(venue: str) -> tuple[float, float]:
    """Look up stadium coordinates for weather fetch."""
    if not venue:
        return STADIUM_COORDS["default"]
    for stadium, coords in STADIUM_COORDS.items():
        if stadium.lower() in venue.lower() or venue.lower() in stadium.lower():
            return coords
    return STADIUM_COORDS["default"]
