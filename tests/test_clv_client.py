"""Odds API client tests — CLV-01."""

from datetime import UTC, datetime, timedelta

import pytest
from pytest_httpx import HTTPXMock


PINNACLE_RESPONSE = {
    "id": "event-uuid-123",
    "bookmakers": [
        {
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "Draw", "price": 3.40},
                        {"name": "Chelsea", "price": 3.80},
                    ],
                }
            ],
        }
    ],
}


class TestOddsApiClient:
    """CLV-01: OddsApiClient fetches Pinnacle closing odds."""

    async def test_fetch_pinnacle_closing_odds_returns_bookmaker_dict(
        self, httpx_mock: HTTPXMock, settings
    ):
        """fetch_pinnacle_closing_odds() returns the Pinnacle bookmaker dict."""
        httpx_mock.add_response(status_code=200, json=PINNACLE_RESPONSE)
        from bip.clv.client import OddsApiClient
        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.fetch_pinnacle_closing_odds(
                sport_key="soccer_epl",
                event_id="event-uuid-123",
                market_key="h2h",
            )
        assert result is not None
        assert result["key"] == "pinnacle"

    async def test_returns_none_when_pinnacle_not_in_bookmakers(
        self, httpx_mock: HTTPXMock, settings
    ):
        """Returns None when Pinnacle is not in the bookmakers list."""
        httpx_mock.add_response(
            status_code=200,
            json={"id": "event-uuid-123", "bookmakers": []},
        )
        from bip.clv.client import OddsApiClient
        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.fetch_pinnacle_closing_odds(
                sport_key="soccer_epl",
                event_id="event-uuid-123",
                market_key="h2h",
            )
        assert result is None

    async def test_api_key_sent_as_query_param(
        self, httpx_mock: HTTPXMock, settings
    ):
        """apiKey query parameter must be present — Odds API v4 auth."""
        httpx_mock.add_response(status_code=200, json=PINNACLE_RESPONSE)
        from bip.clv.client import OddsApiClient
        async with OddsApiClient(api_key="test-odds-key-xyz") as client:
            await client.fetch_pinnacle_closing_odds(
                sport_key="soccer_epl",
                event_id="event-uuid-123",
                market_key="h2h",
            )
        request = httpx_mock.get_requests()[0]
        assert "apiKey=test-odds-key-xyz" in str(request.url)

    async def test_retries_on_429(self, httpx_mock: HTTPXMock, settings):
        """429 responses trigger retry with exponential backoff."""
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=200, json=PINNACLE_RESPONSE)
        from bip.clv.client import OddsApiClient
        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.fetch_pinnacle_closing_odds(
                sport_key="soccer_epl",
                event_id="event-uuid-123",
                market_key="h2h",
            )
        assert result is not None


class TestFindEventByFixture:
    """find_event_by_fixture — resolve Odds API event_id from fixture metadata."""

    KICKOFF = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)

    def _events_payload(
        self,
        *,
        home: str = "Arsenal",
        away: str = "Chelsea",
        commence: datetime | None = None,
        event_id: str = "event-uuid-123",
    ):
        commence = commence or self.KICKOFF
        return [
            {
                "id": event_id,
                "sport_key": "soccer_epl",
                "commence_time": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_team": home,
                "away_team": away,
            }
        ]

    async def test_returns_event_id_on_team_and_time_match(
        self, httpx_mock: HTTPXMock, settings
    ):
        """Happy path: home/away/time all match → returns event id."""
        httpx_mock.add_response(status_code=200, json=self._events_payload())
        from bip.clv.client import OddsApiClient

        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.find_event_by_fixture(
                sport_key="soccer_epl",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff_utc=self.KICKOFF,
            )
        assert result == "event-uuid-123"

    async def test_returns_none_when_no_team_match(
        self, httpx_mock: HTTPXMock, settings
    ):
        """No team-name match → returns None."""
        httpx_mock.add_response(
            status_code=200,
            json=self._events_payload(home="Liverpool", away="Manchester City"),
        )
        from bip.clv.client import OddsApiClient

        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.find_event_by_fixture(
                sport_key="soccer_epl",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff_utc=self.KICKOFF,
            )
        assert result is None

    async def test_returns_none_when_time_outside_window(
        self, httpx_mock: HTTPXMock, settings
    ):
        """Team match but kickoff differs by 4h → returns None."""
        skewed = self.KICKOFF + timedelta(hours=4)
        httpx_mock.add_response(
            status_code=200, json=self._events_payload(commence=skewed)
        )
        from bip.clv.client import OddsApiClient

        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.find_event_by_fixture(
                sport_key="soccer_epl",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff_utc=self.KICKOFF,
            )
        assert result is None

    async def test_team_match_is_case_insensitive(
        self, httpx_mock: HTTPXMock, settings
    ):
        """Case-insensitive team name matching."""
        httpx_mock.add_response(status_code=200, json=self._events_payload())
        from bip.clv.client import OddsApiClient

        async with OddsApiClient(api_key=settings.odds_api_key) as client:
            result = await client.find_event_by_fixture(
                sport_key="soccer_epl",
                home_team="arsenal",
                away_team="CHELSEA",
                kickoff_utc=self.KICKOFF,
            )
        assert result == "event-uuid-123"

    async def test_api_key_sent_as_query_param(
        self, httpx_mock: HTTPXMock, settings
    ):
        """apiKey query parameter must be present on the /events call."""
        httpx_mock.add_response(status_code=200, json=self._events_payload())
        from bip.clv.client import OddsApiClient

        async with OddsApiClient(api_key="test-odds-key-events") as client:
            await client.find_event_by_fixture(
                sport_key="soccer_epl",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff_utc=self.KICKOFF,
            )
        request = httpx_mock.get_requests()[0]
        assert "apiKey=test-odds-key-events" in str(request.url)
        assert "/v4/sports/soccer_epl/events" in str(request.url)
