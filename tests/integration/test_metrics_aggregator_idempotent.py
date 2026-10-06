"""Integration test for MetricsAggregator idempotency (CLV-04 — D-13)."""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

from bip.core.metrics.aggregator import MetricsAggregator
from bip.core.storage.models import PerformanceMetric


def _zero(sport, league, market, period, start, end):
    return PerformanceMetric(
        sport=sport, league=league, market=market, period=period,
        period_start=start, period_end=end,
        total_picks=0, won=0, lost=0, void=0,
        total_staked=0.0, total_pnl=0.0,
        roi=None, yield_pct=None, avg_clv=None, avg_edge=None,
    )


def test_repeat_run_upserts_same_row():
    """Two identical runs of MetricsAggregator.run() must call upsert with the same key.

    Idempotency contract: the (sport,league,market,period,period_start) tuple is the
    upsert key — Supabase repository's existing upsert() declares on_conflict for this.
    Repeating the run must NOT produce duplicate rows.
    """
    perf_repo = MagicMock()
    perf_repo.compute_period = MagicMock(side_effect=lambda **kw: _zero(
        kw["sport"], kw["league"], kw["market"], kw["period"],
        kw["period_start"], kw["period_end"]
    ))
    perf_repo.upsert = MagicMock(return_value={"id": 1})
    pick_repo = MagicMock()
    agg = MetricsAggregator(perf_repo=perf_repo, pick_repo=pick_repo)

    today = date(2026, 5, 3)
    asyncio.run(agg.run(today=today))
    asyncio.run(agg.run(today=today))

    # Each run produces 4 periods x 5 leagues = 20 upserts. Two runs → 40 upserts total.
    assert perf_repo.upsert.call_count == 40
    # First call vs 21st (same period+league+market) share upsert key.
    first_call = perf_repo.upsert.call_args_list[0].args[0]
    twenty_first_call = perf_repo.upsert.call_args_list[20].args[0]
    assert first_call.period == twenty_first_call.period
    assert first_call.period_start == twenty_first_call.period_start
    assert first_call.league == twenty_first_call.league
    assert first_call.market == twenty_first_call.market


def test_compute_period_called_once_per_triple_per_period():
    """Single MetricsAggregator.run() makes exactly N round trips: 4 periods x |leagues| x |markets|.

    D-16: one Supabase RPC call per (sport,league,market) per period. With 5 leagues x
    1 market x 4 periods = 20 RPC calls.
    """
    perf_repo = MagicMock()
    perf_repo.compute_period = MagicMock(side_effect=lambda **kw: _zero(
        kw["sport"], kw["league"], kw["market"], kw["period"],
        kw["period_start"], kw["period_end"]
    ))
    perf_repo.upsert = MagicMock(return_value={"id": 1})
    pick_repo = MagicMock()
    agg = MetricsAggregator(perf_repo=perf_repo, pick_repo=pick_repo)

    asyncio.run(agg.run(today=date(2026, 5, 3)))
    assert perf_repo.compute_period.call_count == 4 * 5  # 4 periods x 5 leagues
