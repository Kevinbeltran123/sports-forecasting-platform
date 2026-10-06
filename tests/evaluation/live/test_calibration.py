"""Tests for IsotonicProbabilityCalibrator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bip.evaluation.live.calibration import (
    IsotonicProbabilityCalibrator,
    PerMarketCalibrator,
    _ece,
)


class TestConstruction:
    def test_basic_fit_from_arrays(self):
        # Predictions match outcomes: well-calibrated input
        rng = np.random.default_rng(0)
        probs = np.linspace(0.05, 0.95, 200)
        outcomes = (rng.uniform(size=200) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        assert cal.n_train == 200
        assert cal.ece_after <= cal.ece_before + 1e-6  # calibration can only help

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            IsotonicProbabilityCalibrator.fit(
                np.array([0.5, 0.6]), np.array([1.0]),
            )

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match=">=10"):
            IsotonicProbabilityCalibrator.fit(
                np.array([0.5, 0.6, 0.7]), np.array([1.0, 0.0, 1.0]),
            )

    def test_rejects_bad_threshold_lengths(self):
        with pytest.raises(ValueError, match="Mismatched threshold"):
            IsotonicProbabilityCalibrator([0.0, 0.5, 1.0], [0.0, 1.0])

    def test_rejects_too_few_thresholds(self):
        with pytest.raises(ValueError, match="at least 2"):
            IsotonicProbabilityCalibrator([0.5], [0.5])

    def test_rejects_non_monotone_x(self):
        with pytest.raises(ValueError, match="x_thresholds"):
            IsotonicProbabilityCalibrator([0.5, 0.4, 0.9], [0.0, 0.5, 1.0])

    def test_rejects_non_monotone_y(self):
        with pytest.raises(ValueError, match="y_thresholds"):
            IsotonicProbabilityCalibrator([0.0, 0.5, 1.0], [0.0, 0.8, 0.3])


class TestTransform:
    def test_identity_on_perfectly_calibrated(self):
        # If raw probs == empirical, isotonic should be ~identity
        rng = np.random.default_rng(42)
        probs = np.linspace(0.1, 0.9, 500)
        outcomes = (rng.uniform(size=500) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        # At an interior point, calibration should not move probability dramatically
        out = cal.transform(0.5)
        assert 0.3 <= out <= 0.7

    def test_corrects_overconfidence(self):
        # Predicted ~0.95 but actual win-rate 0.75 → calibrator should map down
        rng = np.random.default_rng(1)
        n = 200
        # Low probs: well-calibrated
        low_probs = np.full(n, 0.2)
        low_out = (rng.uniform(size=n) < 0.2).astype(float)
        # High probs: overconfident — predicted 0.95, actual 0.75
        high_probs = np.full(n, 0.95)
        high_out = (rng.uniform(size=n) < 0.75).astype(float)
        probs = np.concatenate([low_probs, high_probs])
        outcomes = np.concatenate([low_out, high_out])
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        # Calibrated 0.95 should map down significantly
        calibrated_high = cal.transform(0.95)
        assert calibrated_high < 0.90, (
            f"expected calibrator to pull 0.95 down, got {calibrated_high}"
        )

    def test_clips_out_of_range_inputs(self):
        # Input below smallest training x → maps to first y
        rng = np.random.default_rng(2)
        probs = np.linspace(0.2, 0.8, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        # Way out of range
        assert 0.0 <= cal.transform(0.0) <= 1.0
        assert 0.0 <= cal.transform(1.0) <= 1.0
        assert 0.0 <= cal.transform(-0.5) <= 1.0  # clipped
        assert 0.0 <= cal.transform(1.5) <= 1.0   # clipped

    def test_transform_batch_matches_pointwise(self):
        rng = np.random.default_rng(3)
        probs = np.linspace(0.1, 0.9, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        test_points = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
        batch = cal.transform_batch(test_points)
        pointwise = np.array([cal.transform(p) for p in test_points])
        np.testing.assert_allclose(batch, pointwise)


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        rng = np.random.default_rng(4)
        probs = np.linspace(0.1, 0.9, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        path = tmp_path / "cal.json"
        cal.save_json(path)
        # Roundtrip preserves transform behavior
        loaded = IsotonicProbabilityCalibrator.load_json(path)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert loaded.transform(p) == pytest.approx(cal.transform(p))
        assert loaded.n_train == cal.n_train

    def test_load_rejects_unknown_type(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "type": "platt", "x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0],
        }))
        with pytest.raises(ValueError, match="type='isotonic'"):
            IsotonicProbabilityCalibrator.load_json(path)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        rng = np.random.default_rng(5)
        probs = np.linspace(0.1, 0.9, 50)
        outcomes = (rng.uniform(size=50) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        nested = tmp_path / "deeply" / "nested" / "cal.json"
        cal.save_json(nested)
        assert nested.exists()


class TestECE:
    def test_perfect_calibration_zero_ece(self):
        # When predictions exactly equal empirical, ECE = 0
        probs = np.array([0.5] * 100)
        outcomes = np.array([1.0] * 50 + [0.0] * 50)  # exactly 50% win rate
        ece = _ece(probs, outcomes, n_bins=10)
        assert ece < 0.01

    def test_overconfidence_positive_ece(self):
        # Predict 0.9, actual 0.5 → high ECE
        probs = np.array([0.9] * 100)
        outcomes = np.array([1.0] * 50 + [0.0] * 50)
        ece = _ece(probs, outcomes, n_bins=10)
        assert ece > 0.3

    def test_empty_input_returns_zero(self):
        assert _ece(np.array([]), np.array([])) == 0.0


# ── PerMarketCalibrator ─────────────────────────────────────────────────────


def _gen_market_dataset(seed: int = 0):
    """Generate a multi-market dataset where each market has a distinct bias.

    market_A (n=200): well-calibrated (probs uniform [0.1, 0.9])
    market_B (n=200): overconfident (probs in [0.6, 0.95], empirical wr ~0.30)
    market_C (n=10): too few for per-market fit (uses fallback)
    """
    rng = np.random.default_rng(seed)
    a_probs = np.linspace(0.1, 0.9, 200)
    a_out = (rng.uniform(size=200) < a_probs).astype(float)
    # market_B: predicted [0.6, 0.95] but empirical wr ~0.30 (severely overconfident)
    b_probs = np.linspace(0.6, 0.95, 200)
    b_out = (rng.uniform(size=200) < 0.30).astype(float)
    c_probs = np.linspace(0.4, 0.7, 10)
    c_out = (rng.uniform(size=10) < c_probs).astype(float)
    probs = np.concatenate([a_probs, b_probs, c_probs])
    outcomes = np.concatenate([a_out, b_out, c_out])
    markets = ["market_A"] * 200 + ["market_B"] * 200 + ["market_C"] * 10
    return probs, outcomes, markets


class TestPerMarketFit:
    def test_fits_per_market_when_sufficient_samples(self):
        probs, outcomes, markets = _gen_market_dataset(seed=1)
        cal = PerMarketCalibrator.fit(
            probs, outcomes, markets, min_samples_per_market=25,
        )
        assert "market_A" in cal.per_market
        assert "market_B" in cal.per_market
        # market_C has only 10 samples — should fall back
        assert "market_C" not in cal.per_market
        assert "market_C" in cal.markets_using_fallback

    def test_overconfident_market_pulled_down(self):
        # market_B predicted [0.6, 0.95] but empirical wr ~0.30 →
        # per-market fit should map mid-range B probs well below 0.50
        probs, outcomes, markets = _gen_market_dataset(seed=2)
        cal = PerMarketCalibrator.fit(probs, outcomes, markets)
        b_calibrated = cal.transform(0.80, market="market_B")
        assert b_calibrated < 0.50, (
            f"expected market_B's 0.80 to map well below 0.50, got {b_calibrated}"
        )

    def test_well_calibrated_market_remains_near_identity(self):
        probs, outcomes, markets = _gen_market_dataset(seed=3)
        cal = PerMarketCalibrator.fit(probs, outcomes, markets)
        # market_A is well-calibrated — 0.5 should map near 0.5
        a_calibrated = cal.transform(0.5, market="market_A")
        assert 0.35 < a_calibrated < 0.65

    def test_unseen_market_uses_fallback(self):
        probs, outcomes, markets = _gen_market_dataset(seed=4)
        cal = PerMarketCalibrator.fit(probs, outcomes, markets)
        # Never-seen market → fallback transform
        unseen = cal.transform(0.5, market="market_Z_never_seen")
        fallback_val = cal.fallback.transform(0.5)
        assert unseen == pytest.approx(fallback_val)

    def test_no_market_uses_fallback(self):
        probs, outcomes, markets = _gen_market_dataset(seed=5)
        cal = PerMarketCalibrator.fit(probs, outcomes, markets)
        # market=None → fallback
        no_market = cal.transform(0.5)
        assert no_market == pytest.approx(cal.fallback.transform(0.5))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            PerMarketCalibrator.fit(
                np.array([0.5, 0.6]), np.array([1.0]), ["m1", "m1"],
            )

    def test_respects_custom_min_samples_threshold(self):
        probs, outcomes, markets = _gen_market_dataset(seed=6)
        # Raise threshold to 250 → no market qualifies
        cal = PerMarketCalibrator.fit(
            probs, outcomes, markets, min_samples_per_market=250,
        )
        assert len(cal.per_market) == 0
        assert len(cal.markets_using_fallback) >= 2


class TestPerMarketPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        probs, outcomes, markets = _gen_market_dataset(seed=7)
        cal = PerMarketCalibrator.fit(probs, outcomes, markets)
        path = tmp_path / "per_market.json"
        cal.save_json(path)
        loaded = PerMarketCalibrator.load_json(path)
        # Same transforms on a few test points + markets
        for mkt in ["market_A", "market_B", "market_Z"]:
            for p in [0.2, 0.5, 0.8]:
                assert loaded.transform(p, market=mkt) == pytest.approx(
                    cal.transform(p, market=mkt)
                )
        assert loaded.n_train_total == cal.n_train_total
        assert sorted(loaded.markets_with_own_fit) == sorted(cal.markets_with_own_fit)

    def test_load_rejects_wrong_type(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"type": "isotonic"}))
        with pytest.raises(ValueError, match="type='per_market'"):
            PerMarketCalibrator.load_json(path)


class TestPerMarketInterfaceCompat:
    """PerMarketCalibrator must be interchangeable with IsotonicProbabilityCalibrator
    wherever the consumer calls `.transform(p, market=...)`."""

    def test_isotonic_ignores_market_kwarg(self):
        rng = np.random.default_rng(8)
        probs = np.linspace(0.1, 0.9, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        # Passing a market kwarg should not change behavior (interface compat)
        assert cal.transform(0.5, market="anything") == pytest.approx(
            cal.transform(0.5)
        )
