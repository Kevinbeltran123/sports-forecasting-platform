"""Tests for the API-Football client extensions (TSP Fase 1.1).

Only contract tests — no real HTTP calls. The retry decorator and HTTP
plumbing are already covered by the existing client test suite.
"""
from __future__ import annotations

import httpx
import pytest

from bip.sports.football.client import ApiFootballClient


@pytest.fixture
def client() -> ApiFootballClient:
    return ApiFootballClient(api_key="test-key")


class TestFixturesByTeamSeason:
    @pytest.mark.asyncio
    async def test_signature(self, client: ApiFootballClient) -> None:
        assert hasattr(client, "get_fixtures_by_team_season")
        called_with: dict = {}

        class FakeResp:
            def raise_for_status(self) -> None: ...
            def json(self) -> dict:
                return {"response": []}

        async def fake_get(path: str, params: dict) -> FakeResp:  # type: ignore[override]
            called_with["path"] = path
            called_with["params"] = params
            return FakeResp()

        client._client.get = fake_get  # type: ignore[assignment]
        result = await client.get_fixtures_by_team_season(team_id=26, season=2024)
        assert result == {"response": []}
        assert called_with["path"] == "/fixtures"
        assert called_with["params"] == {"team": 26, "season": 2024}
        await client.aclose()


class TestFixtureEvents:
    @pytest.mark.asyncio
    async def test_signature(self, client: ApiFootballClient) -> None:
        assert hasattr(client, "get_fixture_events")
        called_with: dict = {}

        class FakeResp:
            def raise_for_status(self) -> None: ...
            def json(self) -> dict:
                return {"response": [{"time": {"elapsed": 23}, "type": "Goal"}]}

        async def fake_get(path: str, params: dict) -> FakeResp:  # type: ignore[override]
            called_with["path"] = path
            called_with["params"] = params
            return FakeResp()

        client._client.get = fake_get  # type: ignore[assignment]
        result = await client.get_fixture_events(fixture_id=12345)
        assert called_with["path"] == "/fixtures/events"
        assert called_with["params"] == {"fixture": 12345}
        assert result["response"][0]["time"]["elapsed"] == 23
        await client.aclose()


class TestTeamCoaches:
    @pytest.mark.asyncio
    async def test_signature(self, client: ApiFootballClient) -> None:
        assert hasattr(client, "get_team_coaches")
        called_with: dict = {}

        class FakeResp:
            def raise_for_status(self) -> None: ...
            def json(self) -> dict:
                return {"response": []}

        async def fake_get(path: str, params: dict) -> FakeResp:  # type: ignore[override]
            called_with["path"] = path
            called_with["params"] = params
            return FakeResp()

        client._client.get = fake_get  # type: ignore[assignment]
        await client.get_team_coaches(team_id=26)
        assert called_with["path"] == "/coachs"
        assert called_with["params"] == {"team": 26}
        await client.aclose()
