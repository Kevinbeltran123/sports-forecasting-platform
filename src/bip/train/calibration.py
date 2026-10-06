"""Calibration module -- ML-03.

sklearn 1.8 REMOVED the pre-1.8 ``cv=<legacy-string>`` sentinel for
CalibratedClassifierCV. Replacement:
    from sklearn.frozen import FrozenEstimator
    CalibratedClassifierCV(estimator=FrozenEstimator(fitted), cv=None)

ML-03 sample-size thresholds:
  n < 300        -> Platt (sigmoid)
  300 <= n <= 500 -> Platt fallback (safer on intermediate samples;
                    RESEARCH.md Pitfall 7 -- isotonic overfits <500 samples)
  n > 500        -> Isotonic regression

References:
  - RESEARCH.md Pattern 2: Calibration in sklearn 1.8 (no legacy cv sentinel)
  - RESEARCH.md Pitfall 2: Using the removed cv sentinel from old reference code
  - RESEARCH.md Pitfall 7: 300-500 sample gap -> Platt fallback
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from bip.core.types import CalibrationMethod

logger = structlog.get_logger(__name__)


def select_calibrator(n_samples: int) -> CalibrationMethod:
    """ML-03 threshold selection.

    Returns:
      CalibrationMethod.platt for n_samples <= 500
          (covers n<300 primary rule AND the 300-500 fallback gap)
      CalibrationMethod.isotonic for n_samples > 500
    """
    if n_samples > 500:
        return CalibrationMethod.isotonic
    return CalibrationMethod.platt


def _to_sklearn_method(method: CalibrationMethod) -> str:
    """Map our enum to sklearn's 'method' kwarg vocabulary.

    sklearn expects 'sigmoid' or 'isotonic'; our enum uses 'platt' for
    the Platt-scaling value. Map platt -> sigmoid at the call site.
    """
    if method == CalibrationMethod.platt:
        return "sigmoid"
    if method == CalibrationMethod.isotonic:
        return "isotonic"
    raise ValueError(f"Unsupported calibration method: {method}")


def calibrate(
    ensemble: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> CalibratedClassifierCV:
    """Calibrate a fitted classifier's probability output (ML-03).

    Uses the sklearn 1.8 FrozenEstimator pattern -- NEVER pass the removed
    cv sentinel (the pre-1.8 legacy string constant). The FrozenEstimator
    wrapper prevents accidental refit of the base ensemble, and cv=None
    skips the refit code path entirely.

    Args:
      ensemble: a fitted classifier exposing predict_proba (meta-learner
                output from the Stacking pipeline in Plan 02-04).
      X_cal: calibration features, shape (n_samples, n_features).
      y_cal: calibration labels, shape (n_samples,).

    Returns:
      A fitted CalibratedClassifierCV whose predict_proba rows sum to 1.
    """
    method = select_calibrator(len(y_cal))
    logger.info(
        "calibration_selected",
        method=method.value,
        n_samples=len(y_cal),
    )
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(ensemble),
        method=_to_sklearn_method(method),
        cv=None,  # does not re-fit; uses the frozen estimator as-is
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated
