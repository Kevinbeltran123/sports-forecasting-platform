"""Tests for isotonic calibrator."""
from __future__ import annotations

import numpy as np
import pytest

from bip.evaluation.tournaments.team_style_profiler.backtest.isotonic_calibrator import (
    IsotonicCalibrator,
)


class TestIsotonicCalibrator:
    def test_passes_through_when_not_fit(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.calibrate("BTTS_yes", 0.62) == 0.62

    def test_fit_and_calibrate_monotonic(self) -> None:
        cal = IsotonicCalibrator()
        # Simulate over-prediction: model says 0.7 but actual is 0.5
        # Fit data: predictions evenly spaced; outcomes shifted down 0.2
        rng = np.random.default_rng(0)
        n = 200
        true_p = rng.uniform(0.1, 0.9, n)
        # Over-predicted: predicted = true_p + 0.15 (clamped)
        predicted = np.clip(true_p + 0.15, 0, 1)
        outcomes = (rng.random(n) < true_p).astype(int)
        cal.fit("BTTS_yes", predicted, outcomes)
        # Calibrated 0.7 should be lower (closer to true_p ~0.55)
        cal_out = cal.calibrate("BTTS_yes", 0.7)
        assert cal_out < 0.7

    def test_fit_requires_n_ge_10(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError):
            cal.fit("X", np.array([0.5] * 5), np.array([1] * 5))

    def test_fit_mismatched_lengths_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError):
            cal.fit("X", np.array([0.5] * 10), np.array([1] * 5))

    def test_batch_calibration(self) -> None:
        cal = IsotonicCalibrator()
        # No fit yet -> passes through
        probs = np.array([0.3, 0.5, 0.7])
        result = cal.calibrate_batch("X", probs)
        np.testing.assert_array_equal(result, probs)

    def test_is_fit(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.is_fit("X") is False
        rng = np.random.default_rng(42)
        cal.fit("X", rng.uniform(0, 1, 20), (rng.random(20) > 0.5).astype(int))
        assert cal.is_fit("X") is True
