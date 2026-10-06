"""Tests for the Phase-2 IsotonicCalibrator (sec 5.4)."""
from __future__ import annotations

import numpy as np
import pytest

from bip.evaluation.live.engine_v3.calibrator import (
    CalibrationSample,
    IsotonicCalibrator,
    map_v2_market_to_family,
    minute_bucket,
)
from bip.evaluation.live.engine_v3.thesis import MarketFamily


def test_minute_bucket_boundaries():
    assert minute_bucket(0) == "0-15"
    assert minute_bucket(14) == "0-15"
    assert minute_bucket(15) == "15-30"
    assert minute_bucket(44) == "30-45"
    assert minute_bucket(45) == "45-60"
    assert minute_bucket(74) == "60-75"
    assert minute_bucket(89) == "75-90"
    assert minute_bucket(90) == "90+"
    assert minute_bucket(120) == "90+"
    # Defensive: negative / weird minutes
    assert minute_bucket(-5) == "0-15"


def test_map_v2_market_to_family_corner_btts_goals():
    assert map_v2_market_to_family("match_corners_over_10.5") == MarketFamily.CORNERS
    assert map_v2_market_to_family("btts_second_half") == MarketFamily.BTTS
    assert map_v2_market_to_family("home_ou_1_5") == MarketFamily.GOALS
    assert map_v2_market_to_family("cards_total_4_5") == MarketFamily.CARDS
    assert map_v2_market_to_family("team_to_score_first") == MarketFamily.NEXT_GOAL
    assert map_v2_market_to_family("draw_no_bet") == MarketFamily.RESULT_1X2
    assert map_v2_market_to_family("xyz_unknown") is None


def test_calibrator_corrects_overconfident_predictions():
    """The whole point: when predicted is consistently higher than
    actual, the calibrator pulls predictions down toward the empirical
    rate."""
    rng = np.random.default_rng(42)
    # Synthesise 200 settled BTTS picks where the predictor says 0.85 but
    # actual win rate is 0.55 — exactly the v2 BTTS miscalibration.
    samples: list[CalibrationSample] = []
    for _ in range(200):
        pred = float(rng.normal(0.85, 0.05))
        pred = max(0.01, min(0.99, pred))
        outcome = 1 if rng.random() < 0.55 else 0
        samples.append(
            CalibrationSample(
                family=MarketFamily.BTTS,
                minute=45,
                predicted_p=pred,
                outcome=outcome,
            )
        )
    cal = IsotonicCalibrator().fit(samples)
    # A prediction at the centre of the overconfident region should be
    # pulled down to roughly the empirical rate.
    p_cal = cal.transform(0.85, MarketFamily.BTTS, 45)
    assert 0.45 <= p_cal <= 0.65, f"expected ≈0.55, got {p_cal}"


def test_calibrator_falls_back_to_family_when_cell_sparse():
    """A cell with <30 samples uses the family-global fit instead."""
    rng = np.random.default_rng(7)
    # 50 GOALS samples all in 30-45 bucket
    bucket_samples = [
        CalibrationSample(
            family=MarketFamily.GOALS,
            minute=35,
            predicted_p=float(rng.uniform(0.4, 0.9)),
            outcome=int(rng.random() < 0.5),
        )
        for _ in range(50)
    ]
    cal = IsotonicCalibrator().fit(bucket_samples)
    # Querying an unbucketed minute → should fall through to family fit.
    p_cal_15 = cal.transform(0.7, MarketFamily.GOALS, 15)  # 0-15 bucket
    p_cal_35 = cal.transform(0.7, MarketFamily.GOALS, 35)  # 30-45 bucket
    # Both must return a valid probability.
    assert 0.0 <= p_cal_15 <= 1.0
    assert 0.0 <= p_cal_35 <= 1.0


def test_calibrator_identity_when_family_below_threshold():
    """With fewer than _MIN_FAMILY_SAMPLES (30), the family is not
    calibrated and the transform must return p_raw unchanged."""
    cal = IsotonicCalibrator().fit(
        [
            CalibrationSample(
                family=MarketFamily.CARDS,
                minute=60,
                predicted_p=0.7,
                outcome=1,
            )
            for _ in range(5)  # too few
        ]
    )
    assert cal.transform(0.7, MarketFamily.CARDS, 60) == pytest.approx(0.7)


def test_calibrator_persistence_roundtrip(tmp_path):
    """save+load preserves fitted state."""
    samples = [
        CalibrationSample(
            family=MarketFamily.GOALS,
            minute=45,
            predicted_p=0.8,
            outcome=1 if i % 3 != 0 else 0,
        )
        for i in range(40)
    ]
    cal = IsotonicCalibrator().fit(samples)
    out = cal.save(tmp_path / "cal.pkl")
    cal2 = IsotonicCalibrator.load(out)
    assert cal2.is_fitted == cal.is_fitted
    assert cal2.n_per_family == cal.n_per_family
    p_orig = cal.transform(0.8, MarketFamily.GOALS, 45)
    p_load = cal2.transform(0.8, MarketFamily.GOALS, 45)
    assert p_orig == pytest.approx(p_load)


def test_calibrator_unfitted_is_identity():
    cal = IsotonicCalibrator()
    assert cal.transform(0.42, MarketFamily.GOALS, 30) == pytest.approx(0.42)
