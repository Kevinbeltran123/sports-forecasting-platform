"""Tests for the v3 stake-caps (liquidity) audit.

Covers CSV parsing, per-market median aggregation, per-family bucket
distribution, recommended-min-cap recovery for a target coverage, and
the end-to-end run_audit happy + degenerate paths.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from bip.evaluation.live.engine_v3.runtime.liquidity_audit import (
    DEFAULT_STAKE_CAP_BUCKETS_EUR,
    REQUIRED_CSV_COLUMNS,
    aggregate_by_market,
    compute_family_distribution,
    load_stake_caps_csv,
    run_audit,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _write_caps_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _make_pick(family: str, market_id: str) -> dict:
    return {
        "fixture_id": 1,
        "timestamp_utc": "2026-05-11T12:00:00Z",
        "thesis_id": "t",
        "thesis_layer": "rule",
        "market_id": market_id,
        "family": family,
    }


# ──────────────────────────────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────────────────────────────


def test_load_caps_missing_file_returns_empty(tmp_path):
    df = load_stake_caps_csv(tmp_path / "nonexistent.csv")
    assert df.is_empty()


def test_load_caps_filters_to_betano_only(tmp_path):
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "goals", "market_id": "m1", "bookmaker": "betano",
         "stake_cap_eur": 100.0, "sampled_at_utc": "2026-05-11T12:00:00Z",
         "notes": ""},
        {"market_family": "goals", "market_id": "m1", "bookmaker": "bet365",
         "stake_cap_eur": 5000.0, "sampled_at_utc": "2026-05-11T12:00:00Z",
         "notes": ""},
    ])
    df = load_stake_caps_csv(p)
    assert df.height == 1
    assert df["bookmaker"][0].lower() == "betano"


def test_load_caps_missing_required_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pl.DataFrame([
        {"market_id": "m1", "stake_cap_eur": 100.0}
    ]).write_csv(p)
    with pytest.raises(ValueError, match="missing required columns"):
        load_stake_caps_csv(p)


# ──────────────────────────────────────────────────────────────────────
# Per-market aggregation (median)
# ──────────────────────────────────────────────────────────────────────


def test_aggregate_takes_median_per_market(tmp_path):
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "corners", "market_id": "c1", "bookmaker": "betano",
         "stake_cap_eur": 50.0, "sampled_at_utc": "2026-05-11T10:00:00Z",
         "notes": ""},
        {"market_family": "corners", "market_id": "c1", "bookmaker": "betano",
         "stake_cap_eur": 100.0, "sampled_at_utc": "2026-05-11T11:00:00Z",
         "notes": ""},
        {"market_family": "corners", "market_id": "c1", "bookmaker": "betano",
         "stake_cap_eur": 200.0, "sampled_at_utc": "2026-05-11T12:00:00Z",
         "notes": ""},
    ])
    caps = load_stake_caps_csv(p)
    by_market = aggregate_by_market(caps)
    assert len(by_market) == 1
    m = by_market[0]
    assert m.market_id == "c1"
    assert m.median_cap_eur == 100.0  # median of (50, 100, 200)
    assert m.min_cap_eur == 50.0
    assert m.max_cap_eur == 200.0
    assert m.n_observations == 3


def test_aggregate_empty_returns_empty_list():
    out = aggregate_by_market(pl.DataFrame())
    assert out == []


# ──────────────────────────────────────────────────────────────────────
# Family distribution + recommendation
# ──────────────────────────────────────────────────────────────────────


def test_family_distribution_buckets_picks(tmp_path):
    """5 picks, all on a market with median cap 75 → bucket 50-100 = 100%."""
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "goals", "market_id": "g1", "bookmaker": "betano",
         "stake_cap_eur": 75.0, "sampled_at_utc": "2026-05-11T10:00:00Z",
         "notes": ""},
    ])
    caps = load_stake_caps_csv(p)
    by_market = {m.market_id: m for m in aggregate_by_market(caps)}
    picks = pl.DataFrame([_make_pick("goals", "g1") for _ in range(5)])
    fd = compute_family_distribution("goals", picks, by_market)
    assert fd.bucket_counts["50-100"] == 5
    assert fd.bucket_pct["50-100"] == pytest.approx(1.0)


def test_recommended_min_cap_keeps_target_coverage(tmp_path):
    """5 picks at cap 75 (50-100 bucket), 5 picks at cap 300 (250-500 bucket).

    Target coverage 0.80 → walk buckets high→low; 250-500 contributes
    50%, 100-250 contributes 0%, 50-100 contributes 50%. Cumulative
    high→low: 50% (still <80%), then 50% + 0% = 50% (5000+ first), then
    accumulating... Actually with target 0.80 we need at least 80% in
    aggregate from a bucket boundary upward. The recommended threshold
    walks down until cumulative >= target.
    """
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "goals", "market_id": "g1", "bookmaker": "betano",
         "stake_cap_eur": 75.0, "sampled_at_utc": "2026-05-11T10:00:00Z",
         "notes": ""},
        {"market_family": "goals", "market_id": "g2", "bookmaker": "betano",
         "stake_cap_eur": 300.0, "sampled_at_utc": "2026-05-11T10:00:00Z",
         "notes": ""},
    ])
    caps = load_stake_caps_csv(p)
    by_market = {m.market_id: m for m in aggregate_by_market(caps)}
    picks_rows = (
        [_make_pick("goals", "g1") for _ in range(5)] +
        [_make_pick("goals", "g2") for _ in range(5)]
    )
    picks = pl.DataFrame(picks_rows)
    fd = compute_family_distribution(
        "goals", picks, by_market, target_coverage=0.80
    )
    # 50-100 bucket has 5 picks (50%), 250-500 has 5 picks (50%).
    # Walking high→low: 5000+(0%), 1000-5000(0%), 500-1000(0%),
    # 250-500(50%), 100-250(0%), 50-100(50%). Cumulative reaches 80% at
    # the 50-100 bucket → recommended boundary = 50.
    assert fd.recommended_min_cap_eur == 50.0


def test_family_distribution_handles_unobserved_markets(tmp_path):
    """A pick whose market_id has no cap measurement → counted as unobserved."""
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "goals", "market_id": "known_market",
         "bookmaker": "betano", "stake_cap_eur": 100.0,
         "sampled_at_utc": "2026-05-11T10:00:00Z", "notes": ""},
    ])
    caps = load_stake_caps_csv(p)
    by_market = {m.market_id: m for m in aggregate_by_market(caps)}
    picks = pl.DataFrame([
        _make_pick("goals", "known_market"),
        _make_pick("goals", "unknown_market"),
    ])
    fd = compute_family_distribution("goals", picks, by_market)
    assert fd.n_candidates == 2
    assert fd.n_markets_observed == 1
    assert fd.n_markets_unobserved == 1
    # only 1 of 2 candidates has a cap → only 1 enters bucket counts
    assert sum(fd.bucket_counts.values()) == 1


# ──────────────────────────────────────────────────────────────────────
# End-to-end run_audit
# ──────────────────────────────────────────────────────────────────────


def test_run_audit_empty_picks_returns_only_market_summary(tmp_path):
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "goals", "market_id": "g1", "bookmaker": "betano",
         "stake_cap_eur": 100.0, "sampled_at_utc": "2026-05-11T10:00:00Z",
         "notes": ""},
    ])
    report = run_audit(
        shadow_picks=pl.DataFrame(),
        stake_caps_csv=p,
    )
    assert report.overall_n_candidates == 0
    assert report.by_family == []
    assert len(report.by_market) == 1


def test_run_audit_full_path(tmp_path):
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "corners", "market_id": "c1",
         "bookmaker": "betano", "stake_cap_eur": 80.0,
         "sampled_at_utc": "2026-05-11T10:00:00Z", "notes": ""},
        {"market_family": "goals", "market_id": "g1",
         "bookmaker": "betano", "stake_cap_eur": 200.0,
         "sampled_at_utc": "2026-05-11T10:00:00Z", "notes": ""},
    ])
    picks = pl.DataFrame([
        _make_pick("corners", "c1"),
        _make_pick("corners", "c1"),
        _make_pick("goals", "g1"),
    ])
    report = run_audit(shadow_picks=picks, stake_caps_csv=p)

    assert report.overall_n_candidates == 3
    assert report.overall_n_with_cap == 3
    assert report.overall_observed_pct == pytest.approx(1.0)
    families = {fd.family for fd in report.by_family}
    assert families == {"corners", "goals"}


def test_run_audit_partial_observation_coverage(tmp_path):
    p = tmp_path / "caps.csv"
    _write_caps_csv(p, [
        {"market_family": "goals", "market_id": "g1",
         "bookmaker": "betano", "stake_cap_eur": 200.0,
         "sampled_at_utc": "2026-05-11T10:00:00Z", "notes": ""},
    ])
    picks = pl.DataFrame([
        _make_pick("goals", "g1"),
        _make_pick("goals", "g_unmeasured"),
    ])
    report = run_audit(shadow_picks=picks, stake_caps_csv=p)
    assert report.overall_n_candidates == 2
    assert report.overall_n_with_cap == 1
    assert report.overall_observed_pct == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────
# Bucket label sanity
# ──────────────────────────────────────────────────────────────────────


def test_default_buckets_cover_relevant_range():
    """Sanity: default buckets span 0 to 5000+ EUR which matches the
    1/4-Kelly notional stake range."""
    assert DEFAULT_STAKE_CAP_BUCKETS_EUR[0] == 0.0
    assert DEFAULT_STAKE_CAP_BUCKETS_EUR[-1] == 5000.0
    assert all(
        DEFAULT_STAKE_CAP_BUCKETS_EUR[i] < DEFAULT_STAKE_CAP_BUCKETS_EUR[i + 1]
        for i in range(len(DEFAULT_STAKE_CAP_BUCKETS_EUR) - 1)
    )


def test_required_csv_columns_lock_schema():
    """If we add a column to REQUIRED_CSV_COLUMNS, downstream callers
    + the operator template must update too. Test pins the contract."""
    assert REQUIRED_CSV_COLUMNS == (
        "market_family", "market_id", "bookmaker",
        "stake_cap_eur", "sampled_at_utc",
    )
