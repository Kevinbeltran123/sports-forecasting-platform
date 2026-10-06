"""Calibration metrics for backtest evaluation.

Metrics implemented:
- log_loss(probs, outcomes)         — proper score for probabilistic 1X2 / BTTS / O/U
- brier_score(probs, outcomes)      — mean squared error against the indicator
- mae(predicted, actual)            — for count markets (goals, corners, SoT)
- poisson_deviance(λ, k)            — proper score for Poisson counts
- expected_calibration_error(...)   — ECE: bucket predictions, measure |bucket_mean - bucket_freq|
- reliability_curve(...)            — (bucket_mean_pred, bucket_freq_actual) tuples for plotting

All metrics return floats (or tuples of arrays). No matplotlib — Phase 4
report can convert reliability_curve output to a plot if desired.

References:
- Brier 1950 — Verification of forecasts expressed in terms of probability
- Naeini, Cooper, Hauskrecht 2015 — Obtaining Well Calibrated Probabilities
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EPS = 1e-15  # log-loss clip to avoid log(0)


@dataclass(frozen=True)
class CalibrationCurve:
    """Reliability curve data: bucket_mean_pred[i] vs bucket_freq_actual[i]."""

    bucket_mean_pred: tuple[float, ...]
    bucket_freq_actual: tuple[float, ...]
    bucket_count: tuple[int, ...]


# ─────────────────────────────────────────────────────────────────
# Probabilistic markets
# ─────────────────────────────────────────────────────────────────


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Binary log-loss: -mean(y · log(p) + (1-y) · log(1-p)).

    Args:
        probs: Predicted P(outcome=1) per match.
        outcomes: Observed binary outcome (0 or 1).

    Returns:
        Mean log-loss across the sequence. Lower is better; perfect = 0.
    """
    if len(probs) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probs)} vs {len(outcomes)}")
    if not probs:
        return 0.0

    total = 0.0
    for p, y in zip(probs, outcomes):
        p_clip = max(EPS, min(1 - EPS, p))
        if y == 1:
            total += -math.log(p_clip)
        else:
            total += -math.log(1 - p_clip)
    return total / len(probs)


def multiclass_log_loss(
    probs: Sequence[Sequence[float]], outcomes: Sequence[int]
) -> float:
    """Multiclass log-loss for 1X2 (3 classes: 0=home, 1=draw, 2=away).

    Args:
        probs: Each inner sequence is (p_home, p_draw, p_away) — must sum ~1.
        outcomes: Observed class index (0, 1, or 2).
    """
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs:
        return 0.0

    total = 0.0
    for p_row, y in zip(probs, outcomes):
        if y < 0 or y >= len(p_row):
            raise ValueError(f"outcome {y} out of range for {len(p_row)} classes")
        p = max(EPS, min(1 - EPS, p_row[y]))
        total += -math.log(p)
    return total / len(probs)


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Brier score: mean squared error of probability vs binary indicator.

    Range [0, 1]. Lower is better; perfect = 0; uniformly random binary = 0.25.
    """
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def multiclass_brier_score(
    probs: Sequence[Sequence[float]], outcomes: Sequence[int]
) -> float:
    """Brier score for 1X2 (or any K-class). Sum of per-class Brier averaged.

    For each row: Σ_k (p_k - 1[y=k])^2.
    Averaged across rows. Range [0, K-1] for K classes.
    """
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs:
        return 0.0

    total = 0.0
    for p_row, y in zip(probs, outcomes):
        for k, p in enumerate(p_row):
            indicator = 1.0 if k == y else 0.0
            total += (p - indicator) ** 2
    return total / len(probs)


# ─────────────────────────────────────────────────────────────────
# Count markets
# ─────────────────────────────────────────────────────────────────


def mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean absolute error for count markets (goals, corners, SoT)."""
    if len(predicted) != len(actual):
        raise ValueError("length mismatch")
    if not predicted:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


def poisson_deviance(lambdas: Sequence[float], counts: Sequence[int]) -> float:
    """Poisson deviance: 2 · Σ(k · log(k/λ) − (k − λ)).

    Proper score for Poisson observations under predicted rate λ.
    By convention 0 · log(0) = 0. Lower is better; perfect (k=λ everywhere) = 0.
    """
    if len(lambdas) != len(counts):
        raise ValueError("length mismatch")
    if not lambdas:
        return 0.0

    total = 0.0
    for lam, k in zip(lambdas, counts):
        lam_clip = max(EPS, lam)
        if k == 0:
            term = -(0 - lam_clip)
        else:
            term = k * math.log(k / lam_clip) - (k - lam_clip)
        total += 2 * term
    return total / len(lambdas)


# ─────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────


def expected_calibration_error(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_buckets: int = 10,
) -> float:
    """ECE: weighted mean of |bucket_mean_pred - bucket_freq_actual|.

    Buckets are equal-width over [0, 1]. Empty buckets contribute 0.

    Args:
        probs: Predicted P(outcome=1).
        outcomes: Observed binary (0 or 1).
        n_buckets: Number of equal-width buckets (default 10).

    Returns:
        ECE in [0, 1]. Lower is better; 0 = perfectly calibrated.
    """
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs:
        return 0.0

    arr_p = np.array(probs)
    arr_y = np.array(outcomes)
    n = len(arr_p)

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    ece = 0.0
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        # Right-inclusive on the final bucket so p=1.0 falls in the last bucket.
        if i == n_buckets - 1:
            mask = (arr_p >= lo) & (arr_p <= hi)
        else:
            mask = (arr_p >= lo) & (arr_p < hi)
        bucket_n = int(mask.sum())
        if bucket_n == 0:
            continue
        bucket_pred = float(arr_p[mask].mean())
        bucket_actual = float(arr_y[mask].mean())
        ece += (bucket_n / n) * abs(bucket_pred - bucket_actual)
    return ece


def reliability_curve(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_buckets: int = 10,
) -> CalibrationCurve:
    """Build a reliability curve for plotting / inspection."""
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")

    arr_p = np.array(probs) if probs else np.array([])
    arr_y = np.array(outcomes) if outcomes else np.array([])

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    means_pred: list[float] = []
    freqs_actual: list[float] = []
    counts: list[int] = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        if i == n_buckets - 1:
            mask = (arr_p >= lo) & (arr_p <= hi)
        else:
            mask = (arr_p >= lo) & (arr_p < hi)
        bucket_n = int(mask.sum()) if arr_p.size else 0
        if bucket_n == 0:
            continue
        means_pred.append(float(arr_p[mask].mean()))
        freqs_actual.append(float(arr_y[mask].mean()))
        counts.append(bucket_n)

    return CalibrationCurve(
        bucket_mean_pred=tuple(means_pred),
        bucket_freq_actual=tuple(freqs_actual),
        bucket_count=tuple(counts),
    )
