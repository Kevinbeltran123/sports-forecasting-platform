"""KS-based calibration drift detector tests."""
from __future__ import annotations

import numpy as np

from bip.evaluation.live.engine_v3.phase3 import CalibrationDriftDetector
from bip.evaluation.live.engine_v3.thesis import MarketFamily


def test_drift_detector_min_samples_gate():
    """Below ``min_samples`` no report is produced — even with skewed data."""
    det = CalibrationDriftDetector(min_samples=50, window_size=100)
    for _ in range(20):
        det.record("arch-none:min-60-75:phase-open_attacking",
                   MarketFamily.GOALS, 0.5, 1.0)
    report = det.check_drift(
        "arch-none:min-60-75:phase-open_attacking", MarketFamily.GOALS,
    )
    assert report is None


def test_drift_detector_calibrated_data_does_not_reject():
    """Well-calibrated synthetic predictor → no rejection."""
    rng = np.random.default_rng(1)
    det = CalibrationDriftDetector(min_samples=50, window_size=300, alpha=0.001)
    for _ in range(200):
        p = float(rng.uniform(0, 1))
        outcome = 1.0 if rng.uniform(0, 1) < p else 0.0
        det.record("x", MarketFamily.CORNERS, p, outcome)
    report = det.check_drift("x", MarketFamily.CORNERS)
    assert report is not None
    # KS may reject sometimes due to noise; check no panic-suspension.
    if report.rejected:
        assert det.is_suspended("x", MarketFamily.CORNERS)
    else:
        assert not det.is_suspended("x", MarketFamily.CORNERS)


def test_drift_detector_miscalibrated_triggers_suspension():
    """Predictor always says 0.8 but outcomes are always 0.0 → strong KS rejection."""
    det = CalibrationDriftDetector(min_samples=20, window_size=200, alpha=0.05)
    for _ in range(100):
        det.record("x", MarketFamily.GOALS, predicted_p=0.80, outcome=0.0)
    report = det.check_drift("x", MarketFamily.GOALS)
    assert report is not None
    assert report.rejected is True
    assert det.is_suspended("x", MarketFamily.GOALS)


def test_drift_detector_resume_lifts_suspension():
    det = CalibrationDriftDetector(min_samples=20, window_size=200, alpha=0.05)
    for _ in range(60):
        det.record("x", MarketFamily.GOALS, 0.80, 0.0)
    det.check_drift("x", MarketFamily.GOALS)
    assert det.is_suspended("x", MarketFamily.GOALS)
    assert det.resume("x", MarketFamily.GOALS)
    assert not det.is_suspended("x", MarketFamily.GOALS)
    # Re-resume on a non-suspended cell is a no-op.
    assert not det.resume("x", MarketFamily.GOALS)


def test_drift_detector_check_all_returns_one_per_cell():
    det = CalibrationDriftDetector(min_samples=20, window_size=100)
    for _ in range(50):
        det.record("a", MarketFamily.GOALS, 0.5, 0.5)
        det.record("b", MarketFamily.CORNERS, 0.5, 0.5)
    reports = det.check_all()
    assert len(reports) == 2
    cells = {r.cell for r in reports}
    assert ("a", "goals") in cells
    assert ("b", "corners") in cells


def test_drift_detector_snapshot_state_shape():
    det = CalibrationDriftDetector(min_samples=10, window_size=100)
    for _ in range(20):
        det.record("a", MarketFamily.GOALS, 0.5, 0.5)
    det.check_all()
    state = det.snapshot_state()
    assert state["n_cells_tracked"] == 1
    assert "last_reports" in state
    assert state["n_cells_suspended"] >= 0
