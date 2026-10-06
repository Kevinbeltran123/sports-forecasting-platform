"""Async API-Football v3 client with tenacity retry.

All methods use @retry with exponential backoff covering 429, 500, 502, 503, 504.
Authentication: x-apisports-key header set at client construction (not per-request).

DATA-01: Retry/rate-limit handling via tenacity.
T-05-04: API key loaded from settings env var — never logged.
"""

from __future__ import annotations

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

BASE_URL = "https://v3.football.api-sports.io"


def is_retryable_http_error(exc: BaseException) -> bool:
    """Predicate: retry on rate-limit, server errors, and network failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


class ApiFootballClient:
    """Async API-Football v3 client.

    Usage::

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            raw = await client.get_fixtures(league_id=39, date="2026-04-22")

    All 5 endpoint methods are decorated with @retry (tenacity) covering:
    - HTTP 429 (rate limit)
    - HTTP 500/502/503/504 (server errors)
    - httpx.ConnectError / httpx.TimeoutException (network failures)

    Retry strategy: exponential backoff, min=2s, max=60s, up to 5 attempts.
    On exhaustion, re-raises the original exception.
    """

    def __init__(self, api_key: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-apisports-key": api_key},
            timeout=httpx.Timeout(30.0),
        )

    async def __aenter__(self) -> ApiFootballClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_fixtures(self, league_id: int, date: str) -> dict:
        """GET /fixtures?league={id}&date={YYYY-MM-DD}.

        Args:
            league_id: API-Football league ID (e.g., 39 for Premier League).
            date: Date string in YYYY-MM-DD format.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"league": league_id, "date": date}
        logger.info("api_football_request", endpoint="/fixtures", params=params)
        response = await self._client.get("/fixtures", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_fixtures_by_season(self, league_id: int, season: int) -> dict:
        """GET /fixtures?league={id}&season={yr}.

        Bulk pull of all fixtures for a league/season. Used by the spike
        qualifying_loader to grab every qualifier match in one call.

        Args:
            league_id: API-Football league ID.
            season: Season start year.

        Returns:
            Parsed JSON response dict with all fixtures in the season.
        """
        params = {"league": league_id, "season": season}
        logger.info("api_football_request", endpoint="/fixtures", params=params)
        response = await self._client.get("/fixtures", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_lineups(self, fixture_id: int) -> dict:
        """GET /fixtures/lineups?fixture={id}.

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"fixture": fixture_id}
        logger.info("api_football_request", endpoint="/fixtures/lineups", params=params)
        response = await self._client.get("/fixtures/lineups", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_statistics(self, fixture_id: int) -> dict:
        """GET /fixtures/statistics?fixture={id}.

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"fixture": fixture_id}
        logger.info("api_football_request", endpoint="/fixtures/statistics", params=params)
        response = await self._client.get("/fixtures/statistics", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_injuries(self, league_id: int, season: int) -> dict:
        """GET /injuries?league={id}&season={season}.

        Args:
            league_id: API-Football league ID.
            season: Season year (e.g., 2025 for 2025/26 season).

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"league": league_id, "season": season}
        logger.info("api_football_request", endpoint="/injuries", params=params)
        response = await self._client.get("/injuries", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_h2h(self, team1_id: int, team2_id: int) -> dict:
        """GET /fixtures/headtohead?h2h={team1_id}-{team2_id}.

        Args:
            team1_id: Home team ID.
            team2_id: Away team ID.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"h2h": f"{team1_id}-{team2_id}"}
        logger.info("api_football_request", endpoint="/fixtures/headtohead", params=params)
        response = await self._client.get("/fixtures/headtohead", params=params)
        response.raise_for_status()
        return response.json()

    # ─────────────────────────────────────────────────────────────────
    # Player-level endpoints — added by spike/national-team-tournament-evaluator
    # (Phase 1.1, see internal planning notes §3.2)
    # ─────────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_player_statistics(
        self,
        player_id: int,
        season: int,
        team_id: int | None = None,
    ) -> dict:
        """GET /players?id={pid}&season={yr}[&team={tid}].

        Player season aggregates per club. Required for player recent-form
        baseline (last-N rolling computed downstream from /fixtures/players).

        Args:
            player_id: API-Football player ID.
            season: Season start year (e.g., 2025 for 2025/26).
            team_id: Optional team filter (if a player has multiple clubs in season).

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params: dict[str, int] = {"id": player_id, "season": season}
        if team_id is not None:
            params["team"] = team_id
        logger.info("api_football_request", endpoint="/players", params=params)
        response = await self._client.get("/players", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_fixture_players(self, fixture_id: int) -> dict:
        """GET /fixtures/players?fixture={fid}.

        Per-match player stats (shots, SoT, fouls, key passes, minutes).
        Source for last-N rolling form computation in Phase 1.3.

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"fixture": fixture_id}
        logger.info("api_football_request", endpoint="/fixtures/players", params=params)
        response = await self._client.get("/fixtures/players", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_squad(self, team_id: int) -> dict:
        """GET /players/squads?team={tid}.

        Current squad for a team. For national teams returns the most-recently
        called-up roster (typically 23-30 players including reserves).

        Args:
            team_id: API-Football team ID.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"team": team_id}
        logger.info("api_football_request", endpoint="/players/squads", params=params)
        response = await self._client.get("/players/squads", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_team_statistics(
        self,
        league_id: int,
        season: int,
        team_id: int,
    ) -> dict:
        """GET /teams/statistics?league={lid}&season={yr}&team={tid}.

        Team aggregates over a league/season: fixtures, goals_for/against,
        clean_sheets, cards, penalties. NOTE: corners are NOT in this
        endpoint — must be aggregated via /fixtures/statistics per match.

        Args:
            league_id: API-Football league ID (qualifier league).
            season: Season start year.
            team_id: API-Football team ID.

        Returns:
            Parsed JSON response dict from API-Football.
        """
        params = {"league": league_id, "season": season, "team": team_id}
        logger.info("api_football_request", endpoint="/teams/statistics", params=params)
        response = await self._client.get("/teams/statistics", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_league(
        self,
        league_id: int | None = None,
        country: str | None = None,
        season: int | None = None,
    ) -> dict:
        """GET /leagues with id, country, or season filters.

        At least one filter must be provided. Used for league metadata
        resolution (qualifier league IDs, type=Cup vs League, country code).

        Args:
            league_id: API-Football league ID.
            country: Country name (e.g., "Brazil").
            season: Season start year.

        Returns:
            Parsed JSON response dict from API-Football.

        Raises:
            ValueError: If no filter is provided.
        """
        params: dict[str, int | str] = {}
        if league_id is not None:
            params["id"] = league_id
        if country is not None:
            params["country"] = country
        if season is not None:
            params["season"] = season
        if not params:
            raise ValueError("get_league: at least one of league_id/country/season required")
        logger.info("api_football_request", endpoint="/leagues", params=params)
        response = await self._client.get("/leagues", params=params)
        response.raise_for_status()
        return response.json()

    # ─────────────────────────────────────────────────────────────────
    # Odds — pre-existing
    # ─────────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_odds(
        self,
        fixture_id: int | None = None,
        league_id: int | None = None,
        season: int | None = None,
        bookmaker: str = "Betano",
    ) -> dict:
        """GET /odds — per-fixture or bulk-per-season dispatch.

        Exactly one of (fixture_id) or (league_id + season) must be provided.
        Per-fixture mode has a 7-day historical lookback (API-Football policy);
        use league_id + season for historical backfill via bulk response.

        The ``bookmaker`` param is accepted for caller clarity but is NOT sent
        to the API — API-Football returns all bookmakers; caller filters
        downstream.

        Args:
            fixture_id: API-Football fixture ID (live / recent-history mode).
            league_id: API-Football league ID (bulk-season mode).
            season: Season start year, e.g. 2024 for the 2024-2025 season.
            bookmaker: Bookmaker name (informational; not sent in request).

        Returns:
            Parsed JSON response dict from API-Football.

        Raises:
            ValueError: If neither fixture_id nor (league_id + season) are provided.
        """
        if fixture_id is not None:
            params: dict[str, int | str] = {"fixture": fixture_id}
        elif league_id is not None and season is not None:
            params = {"league": league_id, "season": season}
        else:
            raise ValueError(
                "get_odds: pass either fixture_id or (league_id, season)"
            )
        # SECURITY: log params dict only (fixture/league/season IDs); api_key
        # lives in x-apisports-key header set during __init__, never in logs.
        logger.info("api_football_request", endpoint="/odds", params=params)
        response = await self._client.get("/odds", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_fixtures_by_team_season(
        self,
        team_id: int,
        season: int,
    ) -> dict:
        """GET /fixtures?team={tid}&season={yr}.

        Returns ALL fixtures for a team in the given season year, across
        all leagues/competitions. For national teams "season" is the
        calendar year (Jan-Dec).

        API-Football REQUIRES `season` when querying by team; from/to
        alone returns an error. Caller must loop over seasons to cover
        a multi-year window.

        Used by Team Style Profiler to pull every match of a national
        team with its current coach.

        Args:
            team_id: API-Football team ID.
            season: Calendar year (e.g. 2024 for matches Jan-Dec 2024).

        Returns:
            Parsed JSON response dict with fixtures for that team/season.
        """
        params: dict[str, int | str] = {"team": team_id, "season": season}
        logger.info(
            "api_football_request",
            endpoint="/fixtures",
            params=params,
        )
        response = await self._client.get("/fixtures", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_fixture_events(self, fixture_id: int) -> dict:
        """GET /fixtures/events?fixture={fixture_id}.

        Returns timestamped events (goals, cards, substitutions, VAR) for
        a single fixture. Each event has ``time.elapsed`` (minute), team,
        player, and detail (e.g. 'Yellow Card', 'Normal Goal').

        Used by Team Style Profiler to compute goal-distribution-per-15min
        and time-to-first-card type metrics.

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            Parsed JSON response dict with events list.
        """
        params = {"fixture": fixture_id}
        logger.info(
            "api_football_request",
            endpoint="/fixtures/events",
            params=params,
        )
        response = await self._client.get("/fixtures/events", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_fixture_by_id(self, fixture_id: int) -> dict:
        """GET /fixtures?id={fixture_id}.

        Used by TSP's live state tracker. Returns the single fixture
        with current LIVE state (status.short='1H'/'HT'/'2H'/'ET' and
        status.elapsed=current minute).

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            Parsed JSON response dict.
        """
        params = {"id": fixture_id}
        logger.info(
            "api_football_request",
            endpoint="/fixtures",
            params=params,
        )
        response = await self._client.get("/fixtures", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def get_team_coaches(self, team_id: int) -> dict:
        """GET /coachs?team={team_id}.

        Returns the historical list of coaches for a team with career
        start/end dates per spell. Used by coach_history module to
        identify the current head coach and the start date of the
        current spell — the cutoff for TSP's "current-coach-only" filter.

        Args:
            team_id: API-Football team ID.

        Returns:
            Parsed JSON response dict with coaches list.
        """
        params = {"team": team_id}
        logger.info(
            "api_football_request",
            endpoint="/coachs",
            params=params,
        )
        response = await self._client.get("/coachs", params=params)
        response.raise_for_status()
        return response.json()
