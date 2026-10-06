"""Integration test for Polars datetime parsing of Supabase ISO8601 strings (WARNING 4 fix).

Without this test, a regression to `cast(pl.Datetime)` would silently NULL all
created_at values, causing the window filters to produce 0 rows → return None,
and drift would never alert in production.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from bip.core.metrics.aggregator import compute_weekly_drift


def _iso(d: datetime) -> str:
    """Format datetime as Supabase ISO8601 string with +00:00 offset."""
    # Supabase format: "2026-05-01T12:34:56.789+00:00"
    return d.astimezone(timezone.utc).isoformat()


def test_compute_weekly_drift_parses_supabase_iso8601_strings():
    """ISO8601 strings (with +00:00 offset) from Supabase parse cleanly into Polars Datetime."""
    today = date(2026, 5, 1)
    rows = []
    # Prior window — 35 rows in [today-56, today-28).
    for i in range(35):
        day_offset = 55 - (i % 28)  # spread across [28, 55]
        d = datetime.combine(
            today - timedelta(days=day_offset), datetime.min.time(), tzinfo=timezone.utc
        )
        rows.append({
            "sport": "football", "league": "premier_league", "market": "onextwo",
            "created_at": _iso(d),  # STRING — exactly as Supabase returns it
            "clv_percentage": 5.0,
        })
    # Recent window — 35 rows in [today-28, today).
    for i in range(35):
        day_offset = max(1, 28 - (i % 28))
        d = datetime.combine(
            today - timedelta(days=day_offset), datetime.min.time(), tzinfo=timezone.utc
        )
        rows.append({
            "sport": "football", "league": "premier_league", "market": "onextwo",
            "created_at": _iso(d),  # STRING
            "clv_percentage": 1.0,  # 4pp drop — should trigger HARD
        })

    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today,
        min_picks=30, abs_threshold_pp=2.0,
    )
    assert result is not None, (
        "WARNING 4 regression: ISO8601 strings did NOT parse — windows filtered to 0 rows. "
        "Confirm aggregator.py uses str.to_datetime (NOT cast(pl.Datetime))."
    )
    assert result.recent_n >= 30
    assert result.prior_n >= 30
    assert result.triggered_hard is True


def test_compute_weekly_drift_handles_native_datetime_objects():
    """Native datetime objects (in-process tests) still work — to_datetime is no-op on Datetime dtype."""
    today = date(2026, 5, 1)
    rows = []
    for i in range(35):
        day_offset = 55 - (i % 28)
        d = datetime.combine(
            today - timedelta(days=day_offset), datetime.min.time(), tzinfo=timezone.utc
        )
        rows.append({
            "sport": "football", "league": "premier_league", "market": "onextwo",
            "created_at": d,  # native datetime
            "clv_percentage": 5.0,
        })
    for i in range(35):
        day_offset = max(1, 28 - (i % 28))
        d = datetime.combine(
            today - timedelta(days=day_offset), datetime.min.time(), tzinfo=timezone.utc
        )
        rows.append({
            "sport": "football", "league": "premier_league", "market": "onextwo",
            "created_at": d,  # native datetime
            "clv_percentage": 5.0,
        })
    # Steady CLV — no drift expected, but the function must NOT crash on native datetimes.
    result = compute_weekly_drift(
        rows, "football", "premier_league", "onextwo", today,
        min_picks=30, abs_threshold_pp=2.0,
    )
    assert result is not None  # parsed successfully, just no drift trigger
