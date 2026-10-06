"""The Odds API v4 client wrapper (Wave 2.D).

Minimal sync httpx client for the Pinnacle CLV-capture use case. The
Pinnacle public API shut down 2025-07-23; The Odds API still lists
Pinnacle (https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)
but with explicit "may incur a delay" warning — they scrape the public
TM-style site. ~60s polling lag is acceptable for pre-kickoff CLV
snapshots (operator captures T-5min before kickoff).

Why sync + httpx (not async): the v3 CLV scheduler fires one capture
per fixture per snapshot point. ≤10 concurrent fixtures × 1 request
each = no concurrency benefit from async. Sync is simpler to test,
easier to reason about retries.

Rate limit aware: Rookie tier = 500 req/month. Client tracks
``x-requests-remaining`` and ``x-requests-used`` headers and exposes
them on the response — caller is responsible for stopping at ceiling.

NOT IN SCOPE (deferred to operator's runtime decisions):
- Subscription decision ($20/mo)
- Coverage verification (operator verifies via free-tier ping first)
- API key management (passed in via env or constructor arg; client
  is stateless about secrets storage)
- Scheduler integration (APScheduler) — separate module
- CLV sink writing — separate module, engine_v3 freeze applies

Refs:
- https://the-odds-api.com/liveapi/guides/v4/
- internal research notes §2.D
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3

# Bookmaker keys we care about for international football CLV.
PINNACLE_KEY = "pinnacle"
BETFAIR_EX_EU_KEY = "betfair_ex_eu"

# Standard 1X2 + AH markets per Odds API docs.
H2H_MARKET = "h2h"
SPREADS_MARKET = "spreads"  # Asian handicap

# Sport keys (verify operator-side before subscribing — coverage check).
SOCCER_WC_2026 = "soccer_fifa_world_cup"
SOCCER_EPL = "soccer_epl"  # for coverage testing on a known-live league
SOCCER_LA_LIGA = "soccer_spain_la_liga"


@dataclass(frozen=True)
class RateLimit:
    """Snapshot of The Odds API rate-limit headers from the last response.

    Rookie tier defaults to 500 req/month (refreshes monthly).
    """

    requests_used: int | None
    requests_remaining: int | None
    last_request_cost: int | None  # in API units (varies per endpoint)


@dataclass(frozen=True)
class OddsApiResponse:
    """One API response payload + rate-limit metadata."""

    data: list[dict[str, Any]] | dict[str, Any]
    rate_limit: RateLimit
    status_code: int


class OddsApiError(Exception):
    """All errors from The Odds API surface as this exception."""


class OddsApiRateLimited(OddsApiError):
    """Raised when the monthly request ceiling is hit (HTTP 429)."""


def _parse_rate_limit(headers: httpx.Headers) -> RateLimit:
    def _maybe_int(v: str | None) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return RateLimit(
        requests_used=_maybe_int(headers.get("x-requests-used")),
        requests_remaining=_maybe_int(headers.get("x-requests-remaining")),
        last_request_cost=_maybe_int(headers.get("x-requests-last")),
    )


class OddsApiClient:
    """Sync httpx wrapper over The Odds API v4.

    Usage:
        client = OddsApiClient(api_key="abc123")
        events = client.get_events(SOCCER_WC_2026)
        odds = client.get_event_odds(SOCCER_WC_2026, events[0]["id"])
        print(odds.rate_limit.requests_remaining)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        client_kwargs: dict[str, Any] = {"timeout": timeout_s}
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OddsApiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        reraise=True,
    )
    def _request(
        self, path: str, params: dict[str, Any] | None = None
    ) -> OddsApiResponse:
        url = f"{self.base_url}{path}"
        query: dict[str, Any] = {"apiKey": self.api_key}
        if params:
            query.update(params)
        r = self._client.get(url, params=query)
        if r.status_code == 429:
            raise OddsApiRateLimited(
                f"Rate-limited (HTTP 429): {r.text[:200]} | rate_limit={_parse_rate_limit(r.headers)}"
            )
        if r.status_code >= 400:
            raise OddsApiError(
                f"HTTP {r.status_code} for {path}: {r.text[:200]}"
            )
        return OddsApiResponse(
            data=r.json(),
            rate_limit=_parse_rate_limit(r.headers),
            status_code=r.status_code,
        )

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def get_sports(self, all_sports: bool = False) -> OddsApiResponse:
        """List available sports. ``all_sports=True`` includes out-of-season
        markets. Useful for coverage verification BEFORE subscribing."""
        params: dict[str, Any] = {}
        if all_sports:
            params["all"] = "true"
        return self._request("/sports", params=params)

    def get_events(
        self,
        sport_key: str,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> OddsApiResponse:
        """List upcoming events for a sport. ISO 8601 dates for date filters."""
        params: dict[str, Any] = {}
        if date_from:
            params["commenceTimeFrom"] = date_from
        if date_to:
            params["commenceTimeTo"] = date_to
        return self._request(f"/sports/{sport_key}/events", params=params)

    def get_event_odds(
        self,
        sport_key: str,
        event_id: str,
        *,
        bookmakers: tuple[str, ...] = (PINNACLE_KEY, BETFAIR_EX_EU_KEY),
        markets: tuple[str, ...] = (H2H_MARKET, SPREADS_MARKET),
        regions: str = "eu",
        odds_format: str = "decimal",
    ) -> OddsApiResponse:
        """Fetch one event's odds from the named bookmakers + markets."""
        params = {
            "bookmakers": ",".join(bookmakers),
            "markets": ",".join(markets),
            "regions": regions,
            "oddsFormat": odds_format,
        }
        return self._request(
            f"/sports/{sport_key}/events/{event_id}/odds", params=params
        )

    def get_odds(
        self,
        sport_key: str,
        *,
        bookmakers: tuple[str, ...] = (PINNACLE_KEY, BETFAIR_EX_EU_KEY),
        markets: tuple[str, ...] = (H2H_MARKET,),
        regions: str = "eu",
        odds_format: str = "decimal",
    ) -> OddsApiResponse:
        """Bulk: fetch upcoming events + their odds in one call. Cheaper
        on Rookie quota than enumerating events + per-event odds."""
        params = {
            "bookmakers": ",".join(bookmakers),
            "markets": ",".join(markets),
            "regions": regions,
            "oddsFormat": odds_format,
        }
        return self._request(f"/sports/{sport_key}/odds", params=params)


# ─────────────────────────────────────────────────────────────────
# De-vig + outcome extraction helpers (no API call)
# ─────────────────────────────────────────────────────────────────


def shin_devig_h2h(odds: dict[str, float]) -> dict[str, float]:
    """Shin (1992) de-vigging of a 3-way (1X2) book.

    Pinnacle's ~2% overround is small enough that Shin and basic
    multiplicative de-vig converge to within ~0.1pp. Default to
    multiplicative — simpler and well-documented as the right call
    for Pinnacle specifically per Buchdahl + Cyton.

    Args:
        odds: ``{"home": d_h, "draw": d_d, "away": d_a}`` decimal odds.
    Returns:
        ``{"home": p_h, "draw": p_d, "away": p_a}`` de-vigged probabilities
        summing to 1.0.
    """
    if not all(k in odds for k in ("home", "draw", "away")):
        raise ValueError("h2h odds must have home/draw/away keys")
    inv = {k: 1.0 / float(odds[k]) for k in ("home", "draw", "away")}
    s = sum(inv.values())
    if s <= 0:
        raise ValueError(f"sum of implied probs is non-positive: {s}")
    return {k: v / s for k, v in inv.items()}


def extract_pinnacle_h2h(payload: dict[str, Any]) -> dict[str, float] | None:
    """Pull Pinnacle's h2h decimal odds out of a single-event payload.

    Returns ``{"home": d, "draw": d, "away": d}`` or None if Pinnacle
    didn't post h2h on this event (book bias or pulled lines).
    """
    if "bookmakers" not in payload:
        return None
    for bm in payload["bookmakers"]:
        if bm.get("key") != PINNACLE_KEY:
            continue
        for m in bm.get("markets", []):
            if m.get("key") != H2H_MARKET:
                continue
            outcomes = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
            home_team = payload.get("home_team")
            away_team = payload.get("away_team")
            if home_team is None or away_team is None:
                return None
            home = outcomes.get(home_team)
            away = outcomes.get(away_team)
            draw = outcomes.get("Draw")
            if not all(isinstance(v, (int, float)) for v in (home, away, draw)):
                return None
            return {"home": float(home), "draw": float(draw), "away": float(away)}
    return None
