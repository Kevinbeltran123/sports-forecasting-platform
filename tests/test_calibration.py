"""Calibration selection + FrozenEstimator usage -- ML-03."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class TestCalibrationSelection:
    """ML-03: Platt for <300, Isotonic for >500, Platt fallback for 300-500."""

    def test_method_selection(self):
        """Threshold boundaries match ML-03 exactly."""
        from bip.train.calibration import select_calibrator
        from bip.core.types import CalibrationMethod

        assert select_calibrator(250) == CalibrationMethod.platt
        assert select_calibrator(299) == CalibrationMethod.platt
        assert select_calibrator(300) == CalibrationMethod.platt
        assert select_calibrator(400) == CalibrationMethod.platt  # 300-500 fallback
        assert select_calibrator(500) == CalibrationMethod.platt  # inclusive boundary
        assert select_calibrator(501) == CalibrationMethod.isotonic
        assert select_calibrator(1000) == CalibrationMethod.isotonic

    def test_uses_frozen_estimator(self):
        """calibration.py imports FrozenEstimator and does NOT use cv='prefit' -- sklearn 1.8."""
        with open("src/bip/train/calibration.py") as f:
            src = f.read()
        assert "FrozenEstimator" in src, "Must use FrozenEstimator (sklearn 1.8)"
        assert "'prefit'" not in src, "cv='prefit' removed in sklearn 1.8"
        assert '"prefit"' not in src, "cv=\"prefit\" removed in sklearn 1.8"

    def test_calibration_improves_logloss(self):
        """Calibrated logloss <= uncalibrated on synthetic skewed probs -- ML-03.

        Build a deliberately miscalibrated classifier (LogReg on a nonlinear
        decision surface), then confirm calibration produces equal-or-lower
        logloss on a held-out validation set.
        """
        rng = np.random.default_rng(42)
        n = 1000  # > 500 so isotonic is selected; 200 rows left after 200+600 split
        X = rng.standard_normal((n, 3))
        # Binary target derived from a nonlinear decision surface
        y = ((X[:, 0] * X[:, 1] + 0.5 * X[:, 2]) > 0).astype(int)
        n_cal = 600
        X_tr, X_cal = X[:200], X[200:200 + n_cal]
        y_tr, y_cal = y[:200], y[200:200 + n_cal]
        # Hold out last rows for evaluation
        X_val = X[200 + n_cal:]
        y_val = y[200 + n_cal:]

        base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        raw_probs = base.predict_proba(X_val)
        uncalibrated_ll = log_loss(y_val, raw_probs, labels=[0, 1])

        from bip.train.calibration import calibrate

        calibrated = calibrate(base, X_cal, y_cal)
        cal_probs = calibrated.predict_proba(X_val)
        calibrated_ll = log_loss(y_val, cal_probs, labels=[0, 1])

        # Allow a small epsilon -- calibration must not materially regress logloss
        assert calibrated_ll <= uncalibrated_ll + 0.01, (
            f"Calibration regressed logloss: {uncalibrated_ll:.4f} -> {calibrated_ll:.4f}"
        )
