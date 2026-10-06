"""The Odds API v4 async client for Pinnacle closing odds.

Used at kickoff - 1 minute for CLV snapshot (closing-line capture).
Rookie tier: 500 requests/month — use sparingly.
"""


from datetime import UTC, datetime, timedelta

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from bip.core.errors import ApiError
from bip.core.types import MarketKey

logger = structlog.get_logger(__name__)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Retry on 429 (rate limit), 500, 502, 503, 504."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


class OddsApiClient:
    """Async Odds API v4 client for Pinnacle closing odds."""

    BASE_URL = "https://api.the-odds-api.com"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(30.0),
        )

    async def __aenter__(self) -> "OddsApiClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def fetch_pinnacle_closing_odds(
        self,
        sport_key: str,
        event_id: str,
        market_key: str,
    ) -> dict | None:
        """Fetch Pinnacle closing odds for a specific event.

        Returns:
            Pinnacle bookmaker dict or None if Pinnacle not available.

        Raises:
            ApiError: After 3 retries if The Odds API is unreachable.
        """
        logger.info(
            "fetch_pinnacle_odds",
            sport_key=sport_key,
            event_id=event_id,
            market_key=market_key,
        )
        try:
            response = await self._client.get(
                f"/v4/sports/{sport_key}/events/{event_id}/odds",
                params={
                    "apiKey": self._api_key,
                    "bookmakers": "pinnacle",
                    "markets": market_key,
                    "oddsFormat": "decimal",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {429, 500, 502, 503, 504}:
                raise
            raise ApiError(
                f"Odds API returned {exc.response.status_code} for event {event_id}"
            ) from exc

        data = response.json()
        for bookmaker in data.get("bookmakers", []):
            if bookmaker["key"] == "pinnacle":
                return bookmaker
        return None

    def map_market_key(self, internal_key: str) -> str | None:
        """Convert internal market key to Odds API market key.

        Returns None for markets not supported by The Odds API
        (e.g., corners) or for any unrecognized key. Delegates the
        canonical normalization to ``MarketKey.from_str`` (G-MAINT-05).
        """
        try:
            key = MarketKey.from_str(internal_key)
        except ValueError:
            return None
        odds_key = key.to_odds_api()
        return odds_key or None  # CORNERS -> "" -> None

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def find_event_by_fixture(
        self,
        *,
        sport_key: str,
        home_team: str,
        away_team: str,
        kickoff_utc: datetime,
        window_hours: int = 2,
    ) -> str | None:
        """Resolve The Odds API event_id from API-Football fixture metadata.

        Calls /v4/sports/{sport_key}/events and matches on:
          - case-insensitive contains on home_team AND away_team
          - abs(commence_time - kickoff_utc) <= window_hours

        Args:
            sport_key: Odds API sport identifier (e.g., "soccer_epl").
            home_team: Home team name from the fixture.
            away_team: Away team name from the fixture.
            kickoff_utc: Fixture kickoff (timezone-aware UTC).
            window_hours: Symmetric tolerance for commence_time matching (default 2).

        Returns:
            Event id of the first matching event, or None when no event matches.

        Raises:
            ApiError: After 3 retries on a non-retryable HTTP failure.
        """
        logger.info(
            "find_event_by_fixture",
            sport_key=sport_key,
            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc.isoformat(),
        )
        try:
            response = await self._client.get(
                f"/v4/sports/{sport_key}/events",
                params={"apiKey": self._api_key},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {429, 500, 502, 503, 504}:
                raise
            raise ApiError(
                f"Odds API /events returned {exc.response.status_code} for sport {sport_key}"
            ) from exc

        events = response.json() or []
        home_lc = home_team.strip().lower()
        away_lc = away_team.strip().lower()
        # Ensure kickoff is UTC-aware for comparison
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=UTC)
        window = timedelta(hours=window_hours)

        for event in events:
            api_home = (event.get("home_team") or "").strip().lower()
            api_away = (event.get("away_team") or "").strip().lower()
            commence_raw = event.get("commence_time")
            if not commence_raw:
                continue
            try:
                # Odds API returns "YYYY-MM-DDTHH:MM:SSZ"
                commence = datetime.strptime(
                    commence_raw, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=UTC)
            except ValueError:
                continue

            home_match = home_lc in api_home or api_home in home_lc
            away_match = away_lc in api_away or api_away in away_lc
            time_match = abs(commence - kickoff_utc) <= window

            if home_match and away_match and time_match:
                event_id = event.get("id")
                logger.info(
                    "find_event_match",
                    sport_key=sport_key,
                    event_id=event_id,
                )
                return event_id

        logger.warning(
            "find_event_no_match",
            sport_key=sport_key,
            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc.isoformat(),
            event_count=len(events),
        )
        return None

    async def fetch_historical_closing(
        self,
        event_id: str,
        sport: str = "soccer_epl",
    ) -> dict | None:
        """Fetch historical Pinnacle closing odds at a specific snapshot timestamp.

        PLUMBING STUB — not exercised in Phase 02.1 (D-01). Real implementation
        deferred to a follow-up task after API-Football CLV numbers are validated.
        Endpoint target: /v4/historical/sports/{sport}/events/{event_id}/odds

        Returns None so callers can handle the missing-Pinnacle path gracefully
        without needing a try/except around every call site.
        """
        logger.warning(
            "pinnacle_historical_not_implemented",
            event_id=event_id,
            sport=sport,
            note="fetch_historical_closing is a stub — D-01 deferred",
        )
        return None
