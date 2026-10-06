"""Regime bucketing + isotonic calibration + transfer matrix tests."""
from __future__ import annotations

import numpy as np

from bip.evaluation.live.engine_v3.phase3 import (
    IsotonicCell,
    TimeBucketedIsotonic,
    bucket_gsv,
    minute_bucket,
    transfer_calibration,
)
from bip.evaluation.live.engine_v3.thesis import MarketFamily


# ──────────────────────────────────────────────────────────────────────
# Bucketing
# ──────────────────────────────────────────────────────────────────────


def test_minute_bucket_ranges():
    assert minute_bucket(0) == "00-15"
    assert minute_bucket(14) == "00-15"
    assert minute_bucket(15) == "15-30"
    assert minute_bucket(44) == "30-45"
    assert minute_bucket(45) == "45-60"
    assert minute_bucket(60) == "60-75"
    assert minute_bucket(75) == "75+"
    assert minute_bucket(90) == "75+"


def test_bucket_gsv_napoli_archetype(napoli_gsv):
    key = bucket_gsv(napoli_gsv)
    # The Napoli archetype is detected → archetype field carries it.
    assert key.archetype == "dominant_losing_napoli"
    assert key.minute_bucket == "30-45"


def test_bucket_gsv_none_archetype(base_gsv):
    """A bland base state shouldn't activate any archetype."""
    key = bucket_gsv(base_gsv)
    assert key.archetype == "none"
    # base_gsv fixture is minute=60 → falls in "60-75" bucket (lo-inclusive).
    assert key.minute_bucket == "60-75"


# ──────────────────────────────────────────────────────────────────────
# Isotonic cell + table
# ──────────────────────────────────────────────────────────────────────


def test_isotonic_cell_recovers_monotone_mapping():
    """Synthetic well-calibrated data → fit recovers ≈ identity."""
    rng = np.random.default_rng(42)
    preds = rng.uniform(0, 1, size=400)
    outcomes = (rng.uniform(0, 1, size=400) < preds).astype(float)
    cell = IsotonicCell.fit(preds, outcomes)
    assert cell.n == 400
    # The fitted function on points {0.2, 0.5, 0.8} should be approximately
    # the input — well-calibrated synthetic data.
    for x in (0.2, 0.5, 0.8):
        assert abs(cell.apply(x) - x) < 0.10


def test_isotonic_cell_empty_returns_identity():
    cell = IsotonicCell.fit([], [])
    assert cell.n == 0
    assert cell.apply(0.42) == 0.42


def test_time_bucketed_falls_back_to_family_cell():
    """When the regime cell is empty, family-level cell handles it."""
    cal = TimeBucketedIsotonic(min_samples=10)
    # Family cell with strong calibration shift: predicted 0.5 → empirical 0.7.
    preds = [0.5] * 50
    outs = [0.7] * 50
    cal.fit_family_cell(MarketFamily.GOALS, "60-75", preds, outs)

    # Regime cell not fitted; family fallback should apply.
    p = cal.calibrate(
        0.5, regime="arch-none:min-60-75:phase-open_attacking",
        family=MarketFamily.GOALS, minute_bucket="60-75",
    )
    assert abs(p - 0.7) < 0.05


def test_time_bucketed_falls_back_to_identity_when_no_cells():
    cal = TimeBucketedIsotonic(min_samples=10)
    p = cal.calibrate(
        0.42, regime="arch-none:min-30-45:phase-open_attacking",
        family=MarketFamily.GOALS, minute_bucket="30-45",
    )
    assert p == 0.42


def test_time_bucketed_save_load_roundtrip(tmp_path):
    cal = TimeBucketedIsotonic(min_samples=5)
    preds = [0.4, 0.5, 0.6, 0.7]
    outs = [0.42, 0.55, 0.58, 0.72]
    cal.fit_family_cell(MarketFamily.CORNERS, "60-75", preds, outs)
    path = tmp_path / "cal.json"
    cal.save(path)
    cal2 = TimeBucketedIsotonic.load(path)
    expected = cal.calibrate(0.5, regime="x", family=MarketFamily.CORNERS, minute_bucket="60-75")
    actual = cal2.calibrate(0.5, regime="x", family=MarketFamily.CORNERS, minute_bucket="60-75")
    assert abs(actual - expected) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Transfer matrix
# ──────────────────────────────────────────────────────────────────────


def test_transfer_corners_to_next_corner():
    out = transfer_calibration(
        src=MarketFamily.CORNERS, dst=MarketFamily.NEXT_CORNER, src_p=0.60,
    )
    assert out is not None
    assert 0.0 <= out <= 1.0


def test_transfer_btts_to_goals():
    out = transfer_calibration(
        src=MarketFamily.BTTS, dst=MarketFamily.GOALS, src_p=0.50,
    )
    assert out is not None
    # The transfer carries a +0.07 shift → expect ≈ 0.57.
    assert abs(out - 0.57) < 0.05


def test_transfer_unknown_pair_returns_none():
    out = transfer_calibration(
        src=MarketFamily.PROPS, dst=MarketFamily.CARDS, src_p=0.50,
    )
    assert out is None
