"""Sportmonks-cache-backed ``OddsResolver`` for CLVSampler.

Wires the live spike's snapshot cache to the CLV drift sampler so the
bot can detect line movement on open picks without separate API calls
(everything's already cached by the watch loop).

Resolver contract::

    async (fixture_id, market, selection, bookmaker_id) -> odd | None

The resolver loads the latest snapshot for ``fixture_id``, iterates its
``odds`` array, applies the same market-key → ``(market_id, matcher)``
binding used by ``ValueDetector`` so the matched Odd is guaranteed to be
the same selection we originally backed.

Returns ``None`` when:
- No snapshot exists for the fixture (silent cache miss)
- Market key is not in the bindings table (unsupported market)
- No Odd matches the bookmaker_id + selection (book pulled the market)
- Odd value cannot be parsed (rare — Sportmonks emits weird strings)

Failure-safe: any unexpected exception is logged + None returned. The
sampler treats None as "skip this pick"; the watch loop is never
affected.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from bip.evaluation.live.value_detector import _MARKET_BINDINGS
from bip.sports.football.sportmonks.cache import SportmonksCache
from bip.sports.football.sportmonks.schemas import Fixture, Odd


logger = logging.getLogger(__name__)


# Public alias of the type the sampler expects.
OddsResolver = Callable[[int, str, str, int], Awaitable[float | None]]


def make_cache_odds_resolver(
    cache: SportmonksCache,
) -> OddsResolver:
    """Build an ``OddsResolver`` closure over a ``SportmonksCache``.

    The closure is async (the sampler awaits it), but the underlying
    file I/O is synchronous — no event loop blocking risk since each
    snapshot is one small JSON read.
    """

    async def _resolve(
        fixture_id: int, market: str, selection: str, bookmaker_id: int,
    ) -> float | None:
        try:
            fixture = cache.load_latest_snapshot(fixture_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "clv_resolver_snapshot_load_failed "
                "fixture=%s err=%s", fixture_id, exc,
            )
            return None
        if fixture is None:
            return None
        return resolve_odd_from_fixture(
            fixture, market=market, selection=selection,
            bookmaker_id=bookmaker_id,
        )

    return _resolve


def resolve_odd_from_fixture(
    fixture: Fixture, *, market: str, selection: str, bookmaker_id: int,
) -> float | None:
    """Pure function: extract the current decimal odd for a pick spec.

    Exposed so tests and other tools (analyze_decisions, CLV reports)
    can reuse the same matching logic without going through the cache.
    """
    binding = _MARKET_BINDINGS.get(market)
    if binding is None:
        return None
    market_id, matcher = binding
    if fixture.odds is None:
        return None
    for odd in fixture.odds:
        if odd.bookmaker_id != bookmaker_id:
            continue
        if int(odd.market_id) != int(market_id):
            continue
        if odd.suspended or odd.stopped:
            continue
        try:
            matched_selection = matcher(odd)
        except Exception:  # noqa: BLE001
            continue
        if matched_selection != selection:
            continue
        d = odd.decimal_odd
        if d is None or d <= 1.0:
            continue
        return d
    return None
