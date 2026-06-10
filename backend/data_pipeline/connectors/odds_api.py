import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# Sport keys for football leagues on The Odds API
SPORT_KEYS = {
    "EPL": "soccer_epl",
    "LA_LIGA": "soccer_spain_la_liga",
    "SERIE_A": "soccer_italy_serie_a",
    "BUNDESLIGA": "soccer_germany_bundesliga",
    "LIGUE_1": "soccer_france_ligue_one",
    "CHAMPIONS_LEAGUE": "soccer_uefa_champs_league",
}


class OddsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def _get(self, endpoint: str, params: dict = {}) -> Any:
        params["apiKey"] = self.api_key
        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            remaining = response.headers.get("x-requests-remaining", "?")
            used = response.headers.get("x-requests-used", "?")
            logger.info(
                f"Odds API {endpoint} — used: {used}, remaining: {remaining}"
            )
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Odds API request failed: {e}")
            raise

    async def get_odds(
        self,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h",
        bookmakers: str | None = None,
    ) -> list[dict]:
        params: dict = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        data = await self._get(f"/sports/{sport_key}/odds", params=params)
        return data if isinstance(data, list) else []

    async def get_sports(self) -> list[dict]:
        data = await self._get("/sports")
        return data if isinstance(data, list) else []
