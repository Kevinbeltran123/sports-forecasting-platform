"""Walsh & Joshi 2024 classwise calibration metrics.

Phase 1 of the WC2026 calibration-lock spike adds these on top of the
existing `metrics.py`. The motivation comes from SYNTHESIS.md Conclusion 1:
calibration is not optional, and *marginal* ECE hides per-class miscalibration
that the literature (and the operator's bankroll) cares about.

Why classwise ECE rather than marginal:
    A 1X2 model that predicts Win and Loss perfectly but is 5pp off on Draw
    can still post a low marginal ECE because the Draw bin is a fraction of
    the total mass. Walsh & Joshi 2024 show that Kelly applied to such a
    model goes from +37% expected ROI to −76% real ROI — selecting on
    classwise ECE preserved $11k of edge in their NBA case study.

Why 80% bin-fill:
    With 20 bins and small validation sets (n<200, common at the
    international-tournament level), probability concentrations leave most
    bins empty. ECE computed over <80% filled bins systematically
    underestimates miscalibration — the curve looks smooth precisely because
    the data is missing. The flag goes in the report so a downstream
    consumer can downgrade a "passing" ECE that came from an unreliable bin
    setup.

References:
    Walsh & Joshi 2024 — *Calibration of probabilistic predictions for
    sports betting* (internal research notes)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bip.evaluation.tournaments.backtest.metrics import (
    CalibrationCurve,
    expected_calibration_error,
)

DEFAULT_N_BINS = 20
DEFAULT_MIN_BIN_FILL = 0.8


@dataclass(frozen=True)
class BinFillReport:
    """Result of the 80% bin-fill diagnostic.

    `passes=True` means the ECE computed under this bin setup is statistically
    reliable per Walsh & Joshi 2024. False means the operator should either
    collect more data or reduce `n_bins`.
    """

    n_bins: int
    n_filled: int
    fill_pct: float
    min_fill_required: float
    passes: bool


@dataclass(frozen=True)
class ClasswiseCalibration:
    """Output of `classwise_ece` — averaged ECE plus per-class breakdown."""

    classwise_ece: float
    per_class_ece: tuple[float, ...]
    n_classes: int
    n_bins: int


# ---------------------------------------------------------------------------
# Bin-fill diagnostic
# ---------------------------------------------------------------------------


def bin_fill_check(
    probs: Sequence[float],
    *,
    n_bins: int = DEFAULT_N_BINS,
    min_fill: float = DEFAULT_MIN_BIN_FILL,
) -> BinFillReport:
    """Return how many of `n_bins` uniform bins over [0,1] have ≥1 sample.

    Args:
        probs: 1-D iterable of probabilities (any market, single class).
        n_bins: Number of equal-width bins (default 20 per Walsh & Joshi).
        min_fill: Minimum fraction of bins that must be non-empty for the
            diagnostic to pass (default 0.8 per Walsh & Joshi).

    Returns:
        `BinFillReport` with `passes=True` iff `n_filled / n_bins >= min_fill`.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    arr = np.asarray(list(probs), dtype=float)
    if arr.size == 0:
        return BinFillReport(
            n_bins=n_bins,
            n_filled=0,
            fill_pct=0.0,
            min_fill_required=min_fill,
            passes=False,
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_filled = 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (arr >= lo) & (arr <= hi)
        else:
            mask = (arr >= lo) & (arr < hi)
        if mask.any():
            n_filled += 1

    fill_pct = n_filled / n_bins
    return BinFillReport(
        n_bins=n_bins,
        n_filled=n_filled,
        fill_pct=fill_pct,
        min_fill_required=min_fill,
        passes=fill_pct >= min_fill,
    )


# ---------------------------------------------------------------------------
# Classwise ECE
# ---------------------------------------------------------------------------


def classwise_ece(
    probs_matrix: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> ClasswiseCalibration:
    """Walsh & Joshi 2024 classwise ECE for K-class probabilistic predictions.

    For each class `k`, treat the column `probs_matrix[:, k]` as a binary
    calibration problem ("did class k actually occur N% of the time when
    we said P(class=k) = N%?"), compute ECE for that column, then average
    across classes.

    Args:
        probs_matrix: shape (N, K) — each row is a probability vector.
        outcomes: shape (N,) — observed class index in [0, K).
        n_bins: Number of equal-width bins per class (default 20).

    Returns:
        `ClasswiseCalibration` with the average and the per-class breakdown.
    """
    if len(probs_matrix) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probs_matrix)} vs {len(outcomes)}")
    if not probs_matrix:
        return ClasswiseCalibration(
            classwise_ece=0.0,
            per_class_ece=(),
            n_classes=0,
            n_bins=n_bins,
        )

    matrix = np.asarray([list(row) for row in probs_matrix], dtype=float)
    n_classes = matrix.shape[1]
    outs = np.asarray(list(outcomes), dtype=int)

    if (outs < 0).any() or (outs >= n_classes).any():
        raise ValueError(
            f"outcomes must be in [0, {n_classes}); got "
            f"min={outs.min()} max={outs.max()}"
        )

    per_class: list[float] = []
    for k in range(n_classes):
        binary_indicator = (outs == k).astype(int).tolist()
        per_class.append(
            expected_calibration_error(
                matrix[:, k].tolist(),
                binary_indicator,
                n_buckets=n_bins,
            )
        )

    avg = float(sum(per_class) / n_classes) if per_class else 0.0
    return ClasswiseCalibration(
        classwise_ece=avg,
        per_class_ece=tuple(per_class),
        n_classes=n_classes,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Per-class reliability curves (one per class — for plotting / inspection)
# ---------------------------------------------------------------------------


def per_class_reliability(
    probs_matrix: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[CalibrationCurve, ...]:
    """Build one reliability curve per class for K-class predictions.

    Returns a tuple of length K. Curve `k` contains buckets over the
    distribution of `P(class=k)` predictions, with bucket frequency
    measured as the empirical rate of the actual outcome being class `k`.

    Empty buckets are dropped (consistent with `metrics.reliability_curve`).
    """
    from bip.evaluation.tournaments.backtest.metrics import reliability_curve

    if len(probs_matrix) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs_matrix:
        return ()

    matrix = np.asarray([list(row) for row in probs_matrix], dtype=float)
    n_classes = matrix.shape[1]
    outs = np.asarray(list(outcomes), dtype=int)

    curves: list[CalibrationCurve] = []
    for k in range(n_classes):
        binary_indicator = (outs == k).astype(int).tolist()
        curves.append(
            reliability_curve(
                matrix[:, k].tolist(),
                binary_indicator,
                n_buckets=n_bins,
            )
        )
    return tuple(curves)
