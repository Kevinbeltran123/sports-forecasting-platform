"""MetricsAggregator + compute_daily_metrics tests (CLV-04 — D-13, D-15, D-16)."""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

from bip.core.metrics.aggregator import (
    MetricsAggregator,
    compute_daily_metrics,
)
from bip.core.storage.models import PerformanceMetric


def _zero_metric(sport, league, market, period, start, end):
    return PerformanceMetric(
        sport=sport, league=league, market=market, period=period,
        period_start=start, period_end=end,
        total_picks=0, won=0, lost=0, void=0,
        total_staked=0.0, total_pnl=0.0,
        roi=None, yield_pct=None, avg_clv=None, avg_edge=None,
    )


def test_compute_period_returns_PerformanceMetric():
    """compute_daily_metrics returns a list of PerformanceMetric."""
    perf_repo = MagicMock()
    perf_repo.compute_period = MagicMock(side_effect=lambda **kw: _zero_metric(
        kw["sport"], kw["league"], kw["market"], kw["period"],
        kw["period_start"], kw["period_end"]
    ))
    out = compute_daily_metrics(
        perf_repo, "football", "premier_league", "onextwo", date(2026, 5, 3)
    )
    assert len(out) == 4  # daily, weekly, monthly, all_time
    assert all(isinstance(m, PerformanceMetric) for m in out)


def test_left_join_keeps_picks_without_clv_record():
    """compute_period delegates to migration 005 LEFT JOIN — picks with no CLV row count.

    This test verifies the REPO CALL shape (SQL contract is tested in
    tests/integration/test_compute_period_rpc.py).
    """
    perf_repo = MagicMock()
    perf_repo.compute_period = MagicMock(side_effect=lambda **kw: _zero_metric(
        kw["sport"], kw["league"], kw["market"], kw["period"],
        kw["period_start"], kw["period_end"]
    ))
    compute_daily_metrics(
        perf_repo, "football", "premier_league", "onextwo", date(2026, 5, 3)
    )
    # Repo was called 4 times (4 periods) for the same triple
    assert perf_repo.compute_period.call_count == 4


def test_idempotent_rerun_upserts_not_inserts():
    """MetricsAggregator.run() calls perf_repo.upsert (NOT insert) so re-run is idempotent."""
    perf_repo = MagicMock()
    perf_repo.compute_period = MagicMock(side_effect=lambda **kw: _zero_metric(
        kw["sport"], kw["league"], kw["market"], kw["period"],
        kw["period_start"], kw["period_end"]
    ))
    perf_repo.upsert = MagicMock(return_value={"id": 1})
    pick_repo = MagicMock()
    agg = MetricsAggregator(perf_repo=perf_repo, pick_repo=pick_repo)

    asyncio.run(agg.run(today=date(2026, 5, 3)))
    asyncio.run(agg.run(today=date(2026, 5, 3)))
    # 2 runs x 4 periods x 5 leagues = 40 upsert calls
    assert perf_repo.upsert.call_count == 40
    # No insert was ever called (we only mocked upsert; insert is bare MagicMock)
    assert not perf_repo.insert.called


def test_all_four_periods_computed_in_one_run():
    """One MetricsAggregator.run() produces 4 metrics per (sport,league,market) — D-13."""
    perf_repo = MagicMock()
    seen_periods: list[str] = []

    def _capture(**kw):
        seen_periods.append(kw["period"])
        return _zero_metric(
            kw["sport"], kw["league"], kw["market"], kw["period"],
            kw["period_start"], kw["period_end"]
        )

    perf_repo.compute_period = MagicMock(side_effect=_capture)
    perf_repo.upsert = MagicMock(return_value={"id": 1})
    pick_repo = MagicMock()
    agg = MetricsAggregator(perf_repo=perf_repo, pick_repo=pick_repo)

    asyncio.run(agg.run(today=date(2026, 5, 3)))
    # Per the 5-league discovery, expect each period 5 times.
    assert seen_periods.count("daily") == 5
    assert seen_periods.count("weekly") == 5
    assert seen_periods.count("monthly") == 5
    assert seen_periods.count("all_time") == 5


def test_metrics_aggregation_complete_log_emitted(capsys):
    """structlog event metrics_aggregation_complete fires with upserted count.

    structlog writes to stdout (not stdlib logging), so use capsys not caplog.
    """
    perf_repo = MagicMock()
    perf_repo.compute_period = MagicMock(side_effect=lambda **kw: _zero_metric(
        kw["sport"], kw["league"], kw["market"], kw["period"],
        kw["period_start"], kw["period_end"]
    ))
    perf_repo.upsert = MagicMock(return_value={"id": 1})
    pick_repo = MagicMock()
    agg = MetricsAggregator(perf_repo=perf_repo, pick_repo=pick_repo)
    asyncio.run(agg.run(today=date(2026, 5, 3)))
    captured = capsys.readouterr()
    assert "metrics_aggregation_complete" in captured.out
    assert "upserted=20" in captured.out  # 4 periods x 5 leagues
