"""Bootstrap-CI gate for Brier + ROI — Sprint 1 Ola B.

Replaces the spec's arbitrary "CLV > +3%" threshold (which has never been
hit — as measured on the v2 baseline)
with a STATISTICALLY GROUNDED gate:

  - Brier score with non-parametric bootstrap CI on the holdout
  - ROI with non-parametric bootstrap CI on the realized picks

Verdict policy (3 tiers, parametrizable):

  PASS    (production stake): Brier CI upper < 0.25 AND ROI CI lower > +0.03
  SHADOW  (shadow / 0.5u):    ROI CI lower > -0.005 (i.e. not significantly
                              negative at the configured alpha)
  FAIL                       : everything else (drop the model)

The gate is pure-function; no IO, no state. Sprint 2 wires it into the
delivery worker decision tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class GateVerdict(str, Enum):
    PASS = "PASS"          # production-grade — full stake
    SHADOW = "SHADOW"      # shadow-mode / minimum-stake only
    FAIL = "FAIL"          # drop the model — do not deliver picks


@dataclass(frozen=True)
class GateDecision:
    """Outcome of evaluate_gate() with all stats + reasoning."""

    verdict: GateVerdict
    reasons: tuple[str, ...]
    brier_ci: tuple[float, float, float]  # (lower, point, upper)
    roi_ci: tuple[float, float, float] | None  # None if no picks
    n_picks: int


# ──────────────────────────────────────────────────────────────────────
# Bootstrap CI utilities
# ──────────────────────────────────────────────────────────────────────


def _bootstrap_indices(
    n: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Return array of shape (n_bootstrap, n) with bootstrap-sampled indices."""
    return rng.integers(low=0, high=n, size=(n_bootstrap, n))


def _ci_bounds(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return lo, hi


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between one-hot y and probas.

    For binary problems pass y_proba shape (n, 2) and y_true with values
    {0, 1}; for 3-class 1x2 pass y_proba shape (n, 3) and y_true ∈ {0,1,2}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n, k = y_proba.shape
    if len(y_true) != n:
        raise ValueError(f"y/proba length mismatch: {len(y_true)} vs {n}")
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def bootstrap_brier_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """(lower, point, upper) for multiclass Brier on the holdout.

    Lower-is-better; CI upper bound is the gate-relevant quantile.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    point = brier_score_multiclass(y_true, y_proba)

    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = _bootstrap_indices(n, n_bootstrap, rng)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        samples[i] = brier_score_multiclass(y_true[idx[i]], y_proba[idx[i]])
    lo, hi = _ci_bounds(samples, alpha)
    return lo, point, hi


def bootstrap_roi_ci(
    picks: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI on ROI = mean(picks_pl_units).

    `picks` is an array of per-pick P/L (in units; e.g. -1.0 for loss at
    1u stake, +0.95 for win at 1.95 odds). ROI is the mean.

    Returns (lower, point, upper). Empty input → (nan, nan, nan).
    """
    arr = np.asarray(list(picks), dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    idx = _bootstrap_indices(n, n_bootstrap, rng)
    samples = arr[idx].mean(axis=1)
    lo, hi = _ci_bounds(samples, alpha)
    return float(lo), point, float(hi)


# ──────────────────────────────────────────────────────────────────────
# Gate evaluation
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GatePolicy:
    """Parametric thresholds used by evaluate_gate."""

    brier_upper_for_pass: float = 0.25
    roi_lower_for_pass: float = 0.03
    roi_lower_for_shadow: float = -0.005
    min_picks_for_pass: int = 100  # need enough power to trust the bootstrap
    min_picks_for_shadow: int = 30


DEFAULT_POLICY = GatePolicy()


def evaluate_gate(
    *,
    brier_ci: tuple[float, float, float],
    roi_ci: tuple[float, float, float] | None,
    n_picks: int,
    policy: GatePolicy = DEFAULT_POLICY,
) -> GateDecision:
    """Decide PASS / SHADOW / FAIL given the two CIs and pick count.

    Algorithm:
      1. If Brier CI is NaN (empty holdout) → FAIL.
      2. If Brier upper >= brier_upper_for_pass → cannot PASS (poor cal).
      3. If no picks (roi_ci=None or n=0) → SHADOW iff calibration is
         acceptable, else FAIL. Models with no picks can run in shadow
         to collect data — they don't deliver, so no risk.
      4. If ROI lower > roi_lower_for_pass AND Brier upper < brier_upper
         AND n_picks >= min_picks_for_pass → PASS.
      5. Else if ROI lower > roi_lower_for_shadow AND n_picks >=
         min_picks_for_shadow → SHADOW.
      6. Else FAIL.
    """
    reasons: list[str] = []
    brier_lo, _, brier_hi = brier_ci

    if np.isnan(brier_hi):
        reasons.append("empty_holdout")
        return GateDecision(GateVerdict.FAIL, tuple(reasons), brier_ci, roi_ci, n_picks)

    calibration_ok = brier_hi < policy.brier_upper_for_pass
    if not calibration_ok:
        reasons.append(
            f"brier_ci_upper={brier_hi:.4f} >= {policy.brier_upper_for_pass}"
        )

    if roi_ci is None or n_picks == 0:
        # No picks — model didn't emit. SHADOW only if calibration is OK.
        if calibration_ok:
            reasons.append("no_picks_emitted_but_calibration_ok")
            return GateDecision(
                GateVerdict.SHADOW, tuple(reasons), brier_ci, roi_ci, n_picks
            )
        reasons.append("no_picks_and_poor_calibration")
        return GateDecision(GateVerdict.FAIL, tuple(reasons), brier_ci, roi_ci, n_picks)

    roi_lo, roi_pt, roi_hi = roi_ci

    if (
        calibration_ok
        and roi_lo > policy.roi_lower_for_pass
        and n_picks >= policy.min_picks_for_pass
    ):
        reasons.append(
            f"PASS: brier_upper={brier_hi:.4f} < {policy.brier_upper_for_pass} "
            f"AND roi_lower={roi_lo:.4f} > {policy.roi_lower_for_pass} "
            f"AND n={n_picks} >= {policy.min_picks_for_pass}"
        )
        return GateDecision(
            GateVerdict.PASS, tuple(reasons), brier_ci, roi_ci, n_picks
        )

    if (
        roi_lo > policy.roi_lower_for_shadow
        and n_picks >= policy.min_picks_for_shadow
    ):
        reasons.append(
            f"SHADOW: roi_lower={roi_lo:.4f} > {policy.roi_lower_for_shadow} "
            f"AND n={n_picks} >= {policy.min_picks_for_shadow}"
        )
        return GateDecision(
            GateVerdict.SHADOW, tuple(reasons), brier_ci, roi_ci, n_picks
        )

    reasons.append(
        f"FAIL: roi_lower={roi_lo:.4f} below shadow threshold "
        f"{policy.roi_lower_for_shadow} OR n={n_picks} below "
        f"{policy.min_picks_for_shadow}"
    )
    return GateDecision(GateVerdict.FAIL, tuple(reasons), brier_ci, roi_ci, n_picks)


__all__ = [
    "DEFAULT_POLICY",
    "GateDecision",
    "GatePolicy",
    "GateVerdict",
    "bootstrap_brier_ci",
    "bootstrap_roi_ci",
    "brier_score_multiclass",
    "evaluate_gate",
]
