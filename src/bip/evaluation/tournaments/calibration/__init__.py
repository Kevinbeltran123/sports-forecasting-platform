"""Calibration subpackage for the WC2026 tournament-evaluator spike.

Exports the logit-space calibrator (Ojeda 2023) as the recommended default
and `_IsotonicCalibrator` as a deprecated shim.

Typical usage::

    from bip.evaluation.tournaments.calibration import LogisticLogitCalibrator

    cal = LogisticLogitCalibrator()
    cal.fit(raw_probs, outcomes)          # list/array of float or [[p0,p1,p2],...]
    calibrated = cal.transform(new_probs) # same shape as input
"""

from bip.evaluation.tournaments.calibration.logit_calibrator import (
    LogisticLogitCalibrator,
    _IsotonicCalibrator,
)

__all__ = ["LogisticLogitCalibrator", "_IsotonicCalibrator"]
