import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"


class APIFootballClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-apisports-key": api_key,
        }
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=self.headers,
            timeout=30.0,
        )

    async def close(self):
        await self.client.aclose()

    async def _get(self, endpoint: str, params: dict = {}) -> dict:
        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            remaining = response.headers.get("x-ratelimit-requests-remaining", "?")
            logger.info(f"API-Football {endpoint} — requests remaining: {remaining}")
            return data
        except httpx.HTTPError as e:
            logger.error(f"API-Football request failed: {e}")
            raise

    async def get_fixtures(
    self,
    league_id: int,
    season: int,
    date: str | None = None,
    status: str | None = None,
    last: int | None = None,
    next: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict]:
        params: dict[str, Any] = {"league": league_id, "season": season}
        if date:
            params["from"] = from_date
            params["to"] = to_date
        if status:
            params["status"] = status
        if last:
            params["last"] = last
        if next:
            params["next"] = next
        data = await self._get("/fixtures", params=params)
        return data.get("response", [])

    async def get_fixture_statistics(self, fixture_id: int) -> list[dict]:
        data = await self._get("/fixtures/statistics", params={"fixture": fixture_id})
        return data.get("response", [])

    async def get_team_statistics(
        self, team_id: int, league_id: int, season: int
    ) -> dict:
        data = await self._get(
            "/teams/statistics",
            params={"team": team_id, "league": league_id, "season": season},
        )
        return data.get("response", {})

    async def get_standings(self, league_id: int, season: int) -> list[dict]:
        data = await self._get(
            "/standings", params={"league": league_id, "season": season}
        )
        return data.get("response", [])

    async def get_teams(self, league_id: int, season: int) -> list[dict]:
        data = await self._get(
            "/teams", params={"league": league_id, "season": season}
        )
        return data.get("response", [])
