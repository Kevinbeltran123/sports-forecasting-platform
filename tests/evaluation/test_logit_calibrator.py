"""Layer-1 tests for LogisticLogitCalibrator (Phase 2 of calibration-lock spike).

All tests are synthetic — no real predictor output required.

Synthetic data design:
    - Wide uniform distributions (p_true ~ U(0.1, 0.9)) rather than
      football-shaped concentrations (0.5–0.70). This ensures enough
      probability spread to get reliable ECE estimates across bins —
      the "does calibration improve ECE" assertion needs measurable before/after.
    - Overconfidence is introduced by stretching: p_raw = 0.5 + k*(p_true − 0.5)
      where k > 1 pushes predictions away from 0.5. k=2.5 produces strong,
      reliable miscalibration detectable with n=300 samples.
    - Fixed seeds make tests deterministic across runs.

Layer-2 tests (requires real predictor output) are marked xfail with
# requires-real-data and will be activated in Phase 5 backtest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from bip.evaluation.tournaments.backtest.calibration_metrics import classwise_ece
from bip.evaluation.tournaments.calibration import (
    LogisticLogitCalibrator,
    _IsotonicCalibrator,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_overconfident_binary(
    n: int = 300, overconfidence: float = 2.5, seed: int = 42
) -> tuple[list[float], list[int]]:
    """Binary probs systematically stretched away from 0.5.

    p_true ~ U(0.1, 0.9) → true DGP
    p_raw  = clip(0.5 + overconfidence * (p_true − 0.5), 0.05, 0.95)
    outcomes sampled from Bernoulli(p_true).
    """
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.1, 0.9, n)
    outcomes = rng.binomial(1, p_true).tolist()
    p_raw = np.clip(0.5 + overconfidence * (p_true - 0.5), 0.05, 0.95)
    return p_raw.tolist(), outcomes


def _make_overconfident_1x2(
    n: int = 300, overconfidence: float = 2.5, seed: int = 42
) -> tuple[list[list[float]], list[int]]:
    """Three-class 1X2 probs stretched away from 1/3.

    True distribution: Dirichlet(2, 1, 2) → favour home/away symmetrically.
    Overconfident prediction: stretch each class toward 0/1.
    """
    rng = np.random.default_rng(seed)
    true_probs = rng.dirichlet([2.0, 1.0, 2.0], size=n)
    outcomes = [
        int(rng.choice(3, p=row)) for row in true_probs
    ]

    center = np.array([1 / 3.0, 1 / 3.0, 1 / 3.0])
    raw = center + overconfidence * (true_probs - center)
    raw = np.clip(raw, 0.02, 0.97)
    row_sums = raw.sum(axis=1, keepdims=True)
    raw /= row_sums
    return [row.tolist() for row in raw], outcomes


def _ece_binary(probs: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    """Classwise ECE for a binary market (convenience wrapper)."""
    matrix = [[1.0 - p, p] for p in probs]
    return classwise_ece(matrix, outcomes, n_bins=n_bins).classwise_ece


def _ece_multiclass(
    probs: list[list[float]], outcomes: list[int], n_bins: int = 10
) -> float:
    return classwise_ece(probs, outcomes, n_bins=n_bins).classwise_ece


# ---------------------------------------------------------------------------
# Binary calibration tests
# ---------------------------------------------------------------------------


def test_binary_fit_reduces_ece() -> None:
    """Calibrating overconfident binary probs improves classwise ECE."""
    probs, outcomes = _make_overconfident_binary()
    ece_before = _ece_binary(probs, outcomes)

    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)

    ece_after = _ece_binary(calibrated.tolist(), outcomes)
    assert ece_after < ece_before, (
        f"Expected ECE to improve after calibration; "
        f"before={ece_before:.4f}, after={ece_after:.4f}"
    )


def test_binary_output_bounded() -> None:
    probs, outcomes = _make_overconfident_binary()
    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_binary_output_length() -> None:
    probs, outcomes = _make_overconfident_binary(n=150)
    cal = LogisticLogitCalibrator()
    calibrated = cal.fit(probs, outcomes).transform(probs)
    assert len(calibrated) == 150


def test_binary_identity_on_perfectly_calibrated() -> None:
    """On perfectly calibrated data β ≈ 1, α ≈ 0 — output ≈ input."""
    rng = np.random.default_rng(7)
    p_true = rng.uniform(0.1, 0.9, 300)
    outcomes = rng.binomial(1, p_true).tolist()
    probs = p_true.tolist()  # raw == true

    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)

    # ECE should stay near zero — not strictly smaller because we start at near-zero.
    ece = _ece_binary(calibrated.tolist(), outcomes)
    assert ece < 0.06, f"ECE on perfectly calibrated data should stay low, got {ece:.4f}"


def test_binary_fallback_identity_on_small_n() -> None:
    """n=20 per class triggers identity fallback — α=0, β=1."""
    rng = np.random.default_rng(13)
    n = 40  # intentionally below min_samples_per_class=50
    p = rng.uniform(0.1, 0.9, n).tolist()
    outcomes = [int(v > 0.5) for v in p]

    cal = LogisticLogitCalibrator()
    cal.fit(p, outcomes)

    assert cal._alphas == (0.0, 0.0)
    assert cal._betas == (1.0, 1.0)


def test_binary_not_fitted_raises() -> None:
    cal = LogisticLogitCalibrator()
    with pytest.raises(RuntimeError, match="not been fitted"):
        cal.transform([0.3, 0.5, 0.7])


# ---------------------------------------------------------------------------
# Multiclass calibration tests
# ---------------------------------------------------------------------------


def test_multiclass_rows_sum_to_one() -> None:
    probs, outcomes = _make_overconfident_1x2()
    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)

    row_sums = calibrated.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_multiclass_output_bounded() -> None:
    probs, outcomes = _make_overconfident_1x2()
    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_multiclass_shape_preserved() -> None:
    probs, outcomes = _make_overconfident_1x2(n=100)
    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)
    assert calibrated.shape == (100, 3)


def test_multiclass_fit_reduces_ece() -> None:
    probs, outcomes = _make_overconfident_1x2()
    ece_before = _ece_multiclass(probs, outcomes)

    cal = LogisticLogitCalibrator()
    calibrated = cal.fit_transform(probs, outcomes)

    ece_after = _ece_multiclass(calibrated.tolist(), outcomes)
    assert ece_after < ece_before, (
        f"Multiclass ECE should improve; before={ece_before:.4f}, after={ece_after:.4f}"
    )


def test_multiclass_wrong_column_count_raises() -> None:
    probs, outcomes = _make_overconfident_1x2(n=200)
    cal = LogisticLogitCalibrator()
    cal.fit(probs, outcomes)

    wrong_shape = [[0.5, 0.5] for _ in range(10)]
    with pytest.raises(ValueError, match="columns"):
        cal.transform(wrong_shape)


# ---------------------------------------------------------------------------
# fit_transform vs fit + transform equivalence
# ---------------------------------------------------------------------------


def test_fit_transform_equivalent_binary() -> None:
    probs, outcomes = _make_overconfident_binary(seed=1)

    cal_a = LogisticLogitCalibrator()
    result_a = cal_a.fit_transform(probs, outcomes)

    cal_b = LogisticLogitCalibrator()
    cal_b.fit(probs, outcomes)
    result_b = cal_b.transform(probs)

    np.testing.assert_array_equal(result_a, result_b)


def test_fit_transform_equivalent_multiclass() -> None:
    probs, outcomes = _make_overconfident_1x2(seed=2)

    cal_a = LogisticLogitCalibrator()
    result_a = cal_a.fit_transform(probs, outcomes)

    cal_b = LogisticLogitCalibrator()
    cal_b.fit(probs, outcomes)
    result_b = cal_b.transform(probs)

    np.testing.assert_array_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_persistence_round_trip_binary() -> None:
    probs, outcomes = _make_overconfident_binary(seed=5)
    cal = LogisticLogitCalibrator()
    cal.fit(probs, outcomes)
    original_output = cal.transform(probs)

    restored = LogisticLogitCalibrator.from_dict(cal.to_dict())
    restored_output = restored.transform(probs)

    np.testing.assert_allclose(original_output, restored_output, atol=1e-12)


def test_persistence_round_trip_multiclass() -> None:
    probs, outcomes = _make_overconfident_1x2(seed=6)
    cal = LogisticLogitCalibrator()
    cal.fit(probs, outcomes)
    original_output = cal.transform(probs)

    restored = LogisticLogitCalibrator.from_dict(cal.to_dict())
    restored_output = restored.transform(probs)

    np.testing.assert_allclose(original_output, restored_output, atol=1e-12)


def test_to_dict_not_fitted_raises() -> None:
    cal = LogisticLogitCalibrator()
    with pytest.raises(RuntimeError, match="not been fitted"):
        cal.to_dict()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_not_fitted() -> None:
    assert "not fitted" in repr(LogisticLogitCalibrator())


def test_repr_fitted() -> None:
    probs, outcomes = _make_overconfident_binary(n=100, seed=9)
    cal = LogisticLogitCalibrator()
    cal.fit(probs, outcomes)
    r = repr(cal)
    assert "cls0" in r
    assert "cls1" in r


# ---------------------------------------------------------------------------
# _IsotonicCalibrator deprecation shim
# ---------------------------------------------------------------------------


def test_isotonic_shim_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _IsotonicCalibrator()
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "LogisticLogitCalibrator" in str(w[0].message)


def test_isotonic_shim_still_functional() -> None:
    """The shim still fits and transforms so callers don't break before migration."""
    probs, outcomes = _make_overconfident_binary(n=200, seed=11)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        iso = _IsotonicCalibrator()
        iso.fit(probs, outcomes)
        calibrated = iso.transform(probs)

    assert len(calibrated) == 200
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


# ---------------------------------------------------------------------------
# Layer-2 tests (require real predictor output — deferred to Phase 5 backtest)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="requires-real-data: real BivariatePoissonModel output")
def test_layer2_logistic_beats_isotonic_on_real_bivariate_poisson() -> None:
    """LogisticLogitCalibrator ECE < _IsotonicCalibrator ECE on walk-forward 1X2 probs.

    Activation condition: walk_forward.run_backtest() output available for
    WC 2018 / Euro 2024 / Copa 2024 historical fixtures (Phase 5).
    """
    raise NotImplementedError  # pragma: no cover


@pytest.mark.xfail(reason="requires-real-data: calibrated probs re-evaluated on Phase 5 gate")
def test_layer2_calibrated_1x2_passes_lock_gate() -> None:
    """After LogisticLogitCalibrator.transform(), classwise-ECE ≤ 5% AND Brier ≤ 0.21.

    Activation condition: real WC/Euro/Copa backtested 1X2 predictions available.
    """
    raise NotImplementedError  # pragma: no cover
