"""ApiFootballClient.get_odds dual-mode dispatch tests.

D-07 (adjusted per research finding 1): API-Football /odds?fixture={id} has
a 7-day historical lookback; bulk per-season mode is required for seeding.

Per-fixture mode: GET /odds?fixture={id} (live / 7-day lookback only).
Bulk mode:        GET /odds?league={lid}&season={year} (historical seeding).
"""
from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock


class TestGetOdds:
    """Dual-mode dispatch tests for get_odds (D-07 + research finding 1)."""

    async def test_get_odds_fixture_mode(self, httpx_mock: HTTPXMock, settings):
        """Per-fixture mode sends fixture= param and returns response dict."""
        httpx_mock.add_response(
            status_code=200,
            json={"response": [{"bookmakers": []}]},
        )
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_odds(fixture_id=12345)

        assert "response" in result
        req = httpx_mock.get_requests()[0]
        assert "fixture=12345" in str(req.url)

    async def test_get_odds_bulk_mode(self, httpx_mock: HTTPXMock, settings):
        """Bulk mode sends league= + season= params for historical seeding."""
        httpx_mock.add_response(
            status_code=200,
            json={"response": []},
        )
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_odds(league_id=39, season=2024)

        assert "response" in result
        req = httpx_mock.get_requests()[0]
        assert "league=39" in str(req.url)
        assert "season=2024" in str(req.url)

    async def test_get_odds_rejects_neither(self, settings):
        """Calling get_odds() without fixture_id or (league_id + season) raises ValueError."""
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            with pytest.raises(ValueError, match="fixture_id"):
                await client.get_odds()
