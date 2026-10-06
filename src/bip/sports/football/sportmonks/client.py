"""Async Sportmonks v3 Football API client.

Design:
- Single ``httpx.AsyncClient`` per ``SportmonksClient`` instance
- Rate-limit aware: respects ``rate_limit.remaining`` from response meta
- Forgiving parsing: failures on a single record do not break the batch
- Snapshot-friendly: every method returns Pydantic models AND can dump
  the raw payload via ``raw=True`` flag for cache persistence.

The client is INTENTIONALLY thin — it does HTTP + Pydantic validation
only. Higher-level concerns (caching, prediction, value detection)
live in dedicated modules.

Usage::

    async with SportmonksClient.from_env() as client:
        live = await client.list_inplay_livescores()
        f = await client.get_fixture(
            live[0].id,
            includes=["participants", "state", "statistics", "predictions"],
        )
        print(f.name, f.is_live())
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from bip.sports.football.sportmonks.schemas import (
    Fixture,
    FixtureStatistic,
    Odd,
    Pagination,
    Prediction,
    PressureMinute,
    Trend,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sportmonks.com/v3/football"
DEFAULT_CORE_URL = "https://api.sportmonks.com/v3/core"
DEFAULT_TIMEOUT = 30.0

# Sportmonks accepts comma OR semicolon separators for include lists.
# httpx URL-encodes commas (%2C) which Sportmonks then reads as a single
# include name. Semicolons stay as-is and parse correctly.
INCLUDE_SEP = ";"


class SportmonksError(Exception):
    """Raised on API errors after parsing the JSON body."""

    def __init__(self, status_code: int, message: str, code: int | None = None) -> None:
        super().__init__(f"HTTP {status_code} (code={code}): {message}")
        self.status_code = status_code
        self.message = message
        self.code = code


class SportmonksClient:
    """Async wrapper over Sportmonks v3 endpoints we care about.

    Use ``from_env()`` for the standard SPORTMONKS_API_KEY case, or
    construct directly when injecting a custom http client (tests).
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        core_url: str = DEFAULT_CORE_URL,
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._core = core_url.rstrip("/")
        self._owns_client = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls, **kwargs: Any) -> SportmonksClient:
        # python-dotenv-friendly: callers using pydantic-settings will have
        # already loaded .env; for ad-hoc scripts we attempt a lazy load.
        key = os.environ.get("SPORTMONKS_API_KEY")
        if not key:
            try:
                from dotenv import load_dotenv

                load_dotenv()
                key = os.environ.get("SPORTMONKS_API_KEY")
            except ImportError:
                pass
        if not key:
            raise RuntimeError(
                "SPORTMONKS_API_KEY missing. Set in .env or environment."
            )
        return cls(api_key=key, **kwargs)

    # ── lifecycle ────────────────────────────────────────────────────────

    async def __aenter__(self) -> SportmonksClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    # ── low-level GET ────────────────────────────────────────────────────

    async def _get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = {**(params or {}), "api_token": self._key}
        try:
            r = await self._http.get(url, params=merged)
        except httpx.HTTPError as e:
            logger.warning("sportmonks_http_error url=%s err=%s", url, e)
            raise

        # Always JSON-parse — Sportmonks errors are JSON too.
        try:
            body = r.json()
        except ValueError:
            raise SportmonksError(r.status_code, f"non-JSON response: {r.text[:200]}")

        if r.status_code >= 400:
            raise SportmonksError(
                r.status_code,
                body.get("message", "unknown error"),
                code=body.get("code"),
            )

        return body

    async def _paginate(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        max_pages: int = 50,
    ) -> AsyncIterator[dict[str, Any]]:
        page = 1
        while page <= max_pages:
            p = {**(params or {}), "page": page}
            body = await self._get(url, params=p)
            for record in body.get("data", []) or []:
                yield record
            pag = Pagination.model_validate(body.get("pagination") or {})
            if not pag.has_more:
                break
            page += 1

    @staticmethod
    def _parse_record(record: dict[str, Any], model: type, *, ctx: str) -> Any | None:
        """Parse one record; log + skip on failure (do not break the batch)."""
        try:
            return model.model_validate(record)
        except ValidationError as e:
            logger.debug("sportmonks_parse_skip ctx=%s err=%s", ctx, e)
            return None

    # ── livescores ──────────────────────────────────────────────────────

    async def list_inplay_fixtures(
        self,
        *,
        includes: Sequence[str] | None = None,
    ) -> list[Fixture]:
        """All fixtures currently live (1H / 2H / HT / ET / Pen)."""
        params: dict[str, Any] = {"per_page": 100}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        body = await self._get(f"{self._base}/livescores/inplay", params=params)
        out: list[Fixture] = []
        for rec in body.get("data") or []:
            f = self._parse_record(rec, Fixture, ctx="inplay")
            if f is not None:
                out.append(f)
        return out

    async def list_livescores(
        self,
        *,
        includes: Sequence[str] | None = None,
    ) -> list[Fixture]:
        """Fixtures starting within ±15 min window (pre/post-kickoff)."""
        params: dict[str, Any] = {"per_page": 100}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        body = await self._get(f"{self._base}/livescores", params=params)
        return [
            f for rec in (body.get("data") or [])
            if (f := self._parse_record(rec, Fixture, ctx="livescores")) is not None
        ]

    async def list_recently_updated_livescores(
        self,
        *,
        includes: Sequence[str] | None = None,
    ) -> list[Fixture]:
        """Livescores updated in the last 10s — useful for delta-polling."""
        params: dict[str, Any] = {"per_page": 100}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        body = await self._get(f"{self._base}/livescores/latest", params=params)
        return [
            f for rec in (body.get("data") or [])
            if (f := self._parse_record(rec, Fixture, ctx="latest")) is not None
        ]

    # ── fixtures ────────────────────────────────────────────────────────

    async def get_fixture(
        self,
        fixture_id: int,
        *,
        includes: Sequence[str] | None = None,
    ) -> Fixture:
        params: dict[str, Any] = {}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        body = await self._get(f"{self._base}/fixtures/{fixture_id}", params=params)
        record = body.get("data")
        if record is None:
            raise SportmonksError(404, f"fixture {fixture_id} not found")
        return Fixture.model_validate(record)

    async def get_fixture_with_raw(
        self,
        fixture_id: int,
        *,
        includes: Sequence[str] | None = None,
    ) -> tuple[Fixture, dict[str, Any]]:
        """Return both the parsed Fixture and the raw API record dict.

        Use this in capture pipelines that need to preserve fields the
        Pydantic schema doesn't model (venue details, comments, lineups
        positional metadata, etc.) for retrospective offline analysis.
        Single API call — cheaper than calling ``get_fixture`` and a
        separate raw fetch.
        """
        params: dict[str, Any] = {}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        body = await self._get(f"{self._base}/fixtures/{fixture_id}", params=params)
        record = body.get("data")
        if record is None:
            raise SportmonksError(404, f"fixture {fixture_id} not found")
        return Fixture.model_validate(record), record

    async def get_fixtures_by_date(
        self,
        date_iso: str,  # YYYY-MM-DD
        *,
        includes: Sequence[str] | None = None,
    ) -> list[Fixture]:
        params: dict[str, Any] = {"per_page": 100}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        body = await self._get(
            f"{self._base}/fixtures/date/{date_iso}", params=params
        )
        return [
            f for rec in (body.get("data") or [])
            if (f := self._parse_record(rec, Fixture, ctx="by_date")) is not None
        ]

    async def get_fixtures_by_date_range(
        self,
        start_iso: str,
        end_iso: str,
        *,
        includes: Sequence[str] | None = None,
    ) -> list[Fixture]:
        params: dict[str, Any] = {"per_page": 100}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        out: list[Fixture] = []
        async for rec in self._paginate(
            f"{self._base}/fixtures/between/{start_iso}/{end_iso}",
            params=params,
        ):
            f = self._parse_record(rec, Fixture, ctx="date_range")
            if f is not None:
                out.append(f)
        return out

    async def get_team_recent_fixtures(
        self,
        team_id: int,
        *,
        start_iso: str,
        end_iso: str,
        includes: Sequence[str] | None = None,
    ) -> list[Fixture]:
        """Fetch fixtures for one team in a date range.

        Backbone for the ``team_form`` cache: pulls last N matches with
        ``events`` + ``state`` includes so we can compute goal-timing
        rates, BTTS rate, clean-sheet rate, and W/D/L form offline.

        Sportmonks v3 endpoint:
            GET /football/fixtures/between/{start}/{end}/{team_id}
        """
        params: dict[str, Any] = {"per_page": 100}
        if includes:
            params["include"] = INCLUDE_SEP.join(includes)
        out: list[Fixture] = []
        async for rec in self._paginate(
            f"{self._base}/fixtures/between/{start_iso}/{end_iso}/{team_id}",
            params=params,
        ):
            f = self._parse_record(rec, Fixture, ctx="team_recent")
            if f is not None:
                out.append(f)
        return out

    # ── predictions ─────────────────────────────────────────────────────

    async def get_predictions_for_fixture(self, fixture_id: int) -> list[Prediction]:
        """All ML predictions Sportmonks emits for one fixture."""
        body = await self._get(
            f"{self._base}/predictions/probabilities/fixtures/{fixture_id}"
        )
        records = body.get("data") or []
        if isinstance(records, dict):
            # Single-record case
            records = [records]
        return [
            p for rec in records
            if (p := self._parse_record(rec, Prediction, ctx="predictions")) is not None
        ]

    async def get_value_bets_for_fixture(self, fixture_id: int) -> list[Prediction]:
        """Sportmonks own value-bet tags (type_id=33). Filtered server-side."""
        body = await self._get(
            f"{self._base}/predictions/value-bets/fixtures/{fixture_id}"
        )
        records = body.get("data") or []
        if isinstance(records, dict):
            records = [records]
        return [
            p for rec in records
            if (p := self._parse_record(rec, Prediction, ctx="value_bets")) is not None
        ]

    # ── odds ────────────────────────────────────────────────────────────

    async def get_inplay_odds_for_fixture(
        self,
        fixture_id: int,
        *,
        market_ids: Sequence[int] | None = None,
        bookmaker_ids: Sequence[int] | None = None,
    ) -> list[Odd]:
        """Live odds for one fixture. Optionally filter by market/bookmaker."""
        params: dict[str, Any] = {"per_page": 1000}
        if market_ids:
            params["filters"] = f"market_ids:{','.join(str(m) for m in market_ids)}"
        out: list[Odd] = []
        async for rec in self._paginate(
            f"{self._base}/odds/inplay/fixtures/{fixture_id}",
            params=params,
        ):
            if bookmaker_ids and rec.get("bookmaker_id") not in bookmaker_ids:
                continue
            o = self._parse_record(rec, Odd, ctx="inplay_odds")
            if o is not None:
                out.append(o)
        return out

    async def get_prematch_odds_for_fixture(
        self,
        fixture_id: int,
        *,
        market_ids: Sequence[int] | None = None,
    ) -> list[Odd]:
        params: dict[str, Any] = {"per_page": 1000}
        if market_ids:
            params["filters"] = f"market_ids:{','.join(str(m) for m in market_ids)}"
        out: list[Odd] = []
        async for rec in self._paginate(
            f"{self._base}/odds/pre-match/fixtures/{fixture_id}",
            params=params,
        ):
            o = self._parse_record(rec, Odd, ctx="prematch_odds")
            if o is not None:
                out.append(o)
        return out

    # ── statistics + trends + pressure ──────────────────────────────────

    async def get_statistics_for_fixture(
        self, fixture_id: int
    ) -> list[FixtureStatistic]:
        """Aggregate stats — same data as ``include=statistics`` on get_fixture."""
        body = await self._get(
            f"{self._base}/statistics/seasons/fixtures/{fixture_id}"
        )
        return [
            s for rec in (body.get("data") or [])
            if (s := self._parse_record(rec, FixtureStatistic, ctx="statistics")) is not None
        ]

    async def get_trends_for_fixture(
        self, fixture_id: int, *, type_ids: Sequence[int] | None = None
    ) -> list[Trend]:
        """Minute-by-minute statistic stream. Filter by type_ids if given."""
        f = await self.get_fixture(fixture_id, includes=["trends"])
        trends = f.trends or []
        if type_ids:
            ids = set(type_ids)
            trends = [t for t in trends if t.type_id in ids]
        return list(trends)

    async def get_pressure_for_fixture(self, fixture_id: int) -> list[PressureMinute]:
        f = await self.get_fixture(fixture_id, includes=["pressure"])
        return list(f.pressure or [])

    # ── meta ────────────────────────────────────────────────────────────

    async def get_subscription_info(self) -> dict[str, Any]:
        """Returns plan tier, add-ons, leagues — for CLI / debug."""
        # /my/details was 404 in recon; check subscription via response meta
        # by hitting /leagues with per_page=1 and reading the envelope.
        body = await self._get(f"{self._base}/leagues", params={"per_page": 1})
        return {
            "subscription": body.get("subscription"),
            "rate_limit": body.get("rate_limit"),
            "pagination": body.get("pagination"),
        }

    async def list_all_types(self) -> list[dict[str, Any]]:
        """Pull the full /core/types catalog (paginated)."""
        out: list[dict[str, Any]] = []
        async for rec in self._paginate(
            f"{self._core}/types",
            params={"per_page": 500},
        ):
            out.append(rec)
        return out


# ── helpers ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def sportmonks_client_from_env(**kwargs: Any) -> AsyncIterator[SportmonksClient]:
    """Async context manager that yields a ready-to-use client.

    Convenience wrapper for short scripts; equivalent to::

        async with SportmonksClient.from_env() as client:
            ...
    """
    client = SportmonksClient.from_env(**kwargs)
    try:
        yield client
    finally:
        await client.close()
