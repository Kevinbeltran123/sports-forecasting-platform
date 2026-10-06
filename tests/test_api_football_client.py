"""API-Football client tests — DATA-01 (retry/rate-limit)."""

import pytest
from pytest_httpx import HTTPXMock


class TestApiFootballClient:
    """DATA-01: ApiFootballClient retries on 429 with exponential backoff."""

    async def test_get_fixtures_returns_response_dict(self, httpx_mock: HTTPXMock, settings):
        """Successful GET /fixtures returns parsed JSON dict."""
        httpx_mock.add_response(
            status_code=200,
            json={"response": [{"fixture": {"id": 12345}}]},
        )
        from bip.sports.football.client import ApiFootballClient
        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_fixtures(league_id=39, date="2026-04-22")
        assert "response" in result

    async def test_get_fixtures_retries_on_429(self, httpx_mock: HTTPXMock, settings):
        """On 429, client must retry and succeed on subsequent attempt."""
        # First call: rate limited; second call: success
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(
            status_code=200,
            json={"response": []},
        )
        from bip.sports.football.client import ApiFootballClient
        # tenacity will retry — test must not raise
        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_fixtures(league_id=39, date="2026-04-22")
        assert result == {"response": []}

    async def test_api_key_in_request_header(self, httpx_mock: HTTPXMock, settings):
        """x-apisports-key header must be present on every request."""
        httpx_mock.add_response(status_code=200, json={"response": []})
        from bip.sports.football.client import ApiFootballClient
        async with ApiFootballClient(api_key="test-key-xyz") as client:
            await client.get_fixtures(league_id=39, date="2026-04-22")
        request = httpx_mock.get_requests()[0]
        assert request.headers.get("x-apisports-key") == "test-key-xyz"

    async def test_get_lineups_retries_on_500(self, httpx_mock: HTTPXMock, settings):
        """Server error 500 is retryable — client retries and succeeds."""
        httpx_mock.add_response(status_code=500)
        httpx_mock.add_response(status_code=200, json={"response": []})
        from bip.sports.football.client import ApiFootballClient
        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_lineups(fixture_id=12345)
        assert "response" in result


class TestApiFootballPlayerEndpoints:
    """Phase 1.1 — player-level endpoints for the tournament evaluator spike.

    Tests the 5 new methods added by spike/national-team-tournament-evaluator:
    get_player_statistics, get_fixture_players, get_squad, get_team_statistics,
    get_league. Each test verifies request shape (URL + params) and response
    parsing. Retry behaviour is covered structurally by the @retry decorator
    being applied identically to all methods (smoke-tested in TestApiFootballClient).
    """

    async def test_get_player_statistics_passes_required_params(
        self, httpx_mock: HTTPXMock, settings
    ):
        httpx_mock.add_response(status_code=200, json={"response": [{"player": {"id": 100}}]})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_player_statistics(player_id=100, season=2025)

        request = httpx_mock.get_requests()[0]
        assert request.url.path == "/players"
        assert request.url.params["id"] == "100"
        assert request.url.params["season"] == "2025"
        assert "team" not in request.url.params
        assert result["response"][0]["player"]["id"] == 100

    async def test_get_player_statistics_with_team_filter(
        self, httpx_mock: HTTPXMock, settings
    ):
        httpx_mock.add_response(status_code=200, json={"response": []})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            await client.get_player_statistics(player_id=100, season=2025, team_id=42)

        request = httpx_mock.get_requests()[0]
        assert request.url.params["team"] == "42"

    async def test_get_fixture_players_passes_fixture_id(
        self, httpx_mock: HTTPXMock, settings
    ):
        httpx_mock.add_response(status_code=200, json={"response": []})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            await client.get_fixture_players(fixture_id=999)

        request = httpx_mock.get_requests()[0]
        assert request.url.path == "/fixtures/players"
        assert request.url.params["fixture"] == "999"

    async def test_get_squad_passes_team_id(self, httpx_mock: HTTPXMock, settings):
        httpx_mock.add_response(
            status_code=200,
            json={"response": [{"team": {"id": 6}, "players": [{"id": 1}, {"id": 2}]}]},
        )
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_squad(team_id=6)

        request = httpx_mock.get_requests()[0]
        assert request.url.path == "/players/squads"
        assert request.url.params["team"] == "6"
        assert len(result["response"][0]["players"]) == 2

    async def test_get_team_statistics_requires_three_params(
        self, httpx_mock: HTTPXMock, settings
    ):
        httpx_mock.add_response(status_code=200, json={"response": {"fixtures": {"played": {"total": 8}}}})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            await client.get_team_statistics(league_id=13, season=2025, team_id=6)

        request = httpx_mock.get_requests()[0]
        assert request.url.path == "/teams/statistics"
        assert request.url.params["league"] == "13"
        assert request.url.params["season"] == "2025"
        assert request.url.params["team"] == "6"

    async def test_get_league_with_id(self, httpx_mock: HTTPXMock, settings):
        httpx_mock.add_response(status_code=200, json={"response": [{"league": {"id": 13}}]})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            await client.get_league(league_id=13)

        request = httpx_mock.get_requests()[0]
        assert request.url.path == "/leagues"
        assert request.url.params["id"] == "13"

    async def test_get_league_with_country_and_season(
        self, httpx_mock: HTTPXMock, settings
    ):
        httpx_mock.add_response(status_code=200, json={"response": []})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            await client.get_league(country="Brazil", season=2025)

        request = httpx_mock.get_requests()[0]
        assert request.url.params["country"] == "Brazil"
        assert request.url.params["season"] == "2025"

    async def test_get_league_no_params_raises(self, settings):
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            with pytest.raises(ValueError, match="at least one"):
                await client.get_league()

    async def test_player_endpoint_retries_on_429(
        self, httpx_mock: HTTPXMock, settings
    ):
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=200, json={"response": []})
        from bip.sports.football.client import ApiFootballClient

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            result = await client.get_squad(team_id=6)
        assert result == {"response": []}
