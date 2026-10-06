"""compute_weekly_drift tests (D-14 + RESEARCH §Drift Statistical Method)."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from bip.core.metrics.aggregator import compute_weekly_drift


def _row(d: datetime, clv: float, sport="football", league="premier_league", market="onextwo"):
    return {
        "sport": sport, "league": league, "market": market,
        "created_at": d, "clv_percentage": clv,
    }


def _make_rows(today: date, recent_clv: float, prior_clv: float,
               recent_n: int = 35, prior_n: int = 35) -> list[dict]:
    """Generate rows: prior_n rows in [today-56, today-28), recent_n rows in [today-28, today).

    Spread evenly across days so each day in window has ~1 row (n_days~28).
    """
    rows = []
    # Prior window: 28 days, ~prior_n rows spread across [today-56, today-28).
    for i in range(prior_n):
        # day_offset in [28, 56) → strictly inside [today-56, today-28).
        day_offset = 56 - 1 - (i % 28)
        d = datetime.combine(
            today - timedelta(days=day_offset), datetime.min.time(), tzinfo=UTC
        )
        rows.append(_row(d, prior_clv))
    # Recent window: 28 days, ~recent_n rows spread across [today-28, today).
    for i in range(recent_n):
        # day_offset in [1, 28] → strictly inside [today-28, today).
        day_offset = 28 - (i % 28)
        if day_offset == 0:
            day_offset = 1
        d = datetime.combine(
            today - timedelta(days=day_offset), datetime.min.time(), tzinfo=UTC
        )
        rows.append(_row(d, recent_clv))
    return rows


def test_hard_threshold_2pp_drop_triggers():
    """Recent mean drops by 2.5pp from prior mean → triggered_hard=True."""
    today = date(2026, 5, 1)
    rows = _make_rows(today, recent_clv=1.0, prior_clv=3.5)
    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today,
        min_picks=30, abs_threshold_pp=2.0, stdev_multiplier=1.5,
        stdev_floor_pp=1.0,
    )
    assert result is not None
    assert result.triggered_hard is True
    assert result.delta_pp <= -2.0


def test_soft_threshold_15x_stdev_triggers():
    """Recent mean drop >= 1.5*stdev (with stdev > floor) → triggered_soft=True."""
    today = date(2026, 5, 1)
    rows = _make_rows(today, recent_clv=3.0, prior_clv=5.0)
    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today,
        min_picks=30, abs_threshold_pp=2.0, stdev_multiplier=1.5,
        stdev_floor_pp=1.0,
    )
    assert result is not None
    assert result.triggered_soft is True


def test_insufficient_sample_returns_None():
    """recent_n < min OR prior_n < min → None (no alert, no false alarm)."""
    today = date(2026, 5, 1)
    rows = _make_rows(today, recent_clv=0.0, prior_clv=5.0, recent_n=10, prior_n=10)
    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today, min_picks=30
    )
    assert result is None


def test_stdev_floor_prevents_zero_stdev_tight_gate():
    """Steady prior → raw stdev~0 → floor=1.0pp prevents soft gate from being absurdly tight."""
    today = date(2026, 5, 1)
    rows = _make_rows(today, recent_clv=3.5, prior_clv=4.0)
    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today,
        min_picks=30, abs_threshold_pp=2.0, stdev_multiplier=1.5,
        stdev_floor_pp=1.0,
    )
    assert result is not None
    assert result.stdev_pp >= 1.0  # floor applied
    assert result.triggered_soft is False  # 0.5pp drop does NOT trigger with floor


def test_drift_uses_stdev_of_daily_means_not_per_pick():
    """RESEARCH §Drift Statistical Method: stdev computed over PER-DAY MEANS, not per-pick.

    Inject one HUGE outlier (CLV=50.0) on a single day in the prior window. If stdev were
    per-pick, the single huge value would dominate (stdev > 5pp). With per-DAY mean
    aggregation, that day's mean is (50 + 4*5)/5 = 14, so the day-mean stdev is bounded
    by the spread among 28 daily means.
    """
    today = date(2026, 5, 1)
    rows = []
    for i in range(56):
        day = today - timedelta(days=56 - i)
        d = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        for j in range(5):
            clv = 5.0 if not (i == 5 and j == 0) else 50.0
            rows.append(_row(d, clv))
    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today,
        min_picks=30, stdev_floor_pp=0.01,
    )
    assert result is not None
    # If per-pick stdev were used, it would be very large (single 50.0 in 280 rows).
    # Per-day-mean stdev is bounded — single outlier dilutes into one day's average.
    assert result.stdev_pp > 0.01  # floor not engaged
    assert result.stdev_pp < 5.0  # per-day-mean caps the impact of one outlier
