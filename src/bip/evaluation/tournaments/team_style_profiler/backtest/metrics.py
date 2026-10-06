"""Backtest metrics — Brier, ECE, hit rate, coverage.

All functions operate on aligned numpy arrays: ``probs`` (predicted
probability of the positive class) and ``outcomes`` (0/1 ground truth).

Per-market gates (operator decision):
  - BTTS: Brier <= 0.235 -> PASS
  - O/U 2.5: Brier <= 0.230 -> PASS
  - O/U 3.5: Brier <= 0.235 -> PASS
  - Corners O/U: Brier <= 0.250 -> PASS
  - Cards O/U: Brier <= 0.250 -> PASS
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── Core metric primitives ─────────────────────────────────────────────


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Brier score = mean((p - y)^2). 0 = perfect, 0.25 = uninformative."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: weighted-by-bin-count gap between bin's mean predicted prob
    and bin's actual fraction positive.

    Lower is better. 0 = perfectly calibrated.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        bin_avg = float(p[mask].mean())
        bin_pos = float(y[mask].mean())
        bin_n = int(mask.sum())
        ece += (bin_n / n) * abs(bin_avg - bin_pos)
    return float(ece)


def classwise_ece(
    probs_3class: np.ndarray, outcomes_3class: np.ndarray, n_bins: int = 10
) -> float:
    """ECE generalization for 3-class problems (used for 1X2 even though
    TSP doesn't emit 1X2 — kept for completeness)."""
    if probs_3class.size == 0:
        return float("nan")
    n_classes = probs_3class.shape[1]
    ece_sum = 0.0
    for c in range(n_classes):
        p = probs_3class[:, c]
        y = (outcomes_3class == c).astype(float)
        ece_sum += expected_calibration_error(p, y, n_bins=n_bins)
    return ece_sum / n_classes


def hit_rate(probs: np.ndarray, outcomes: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of cases where (probs >= threshold) matches the outcome.

    For O/U: probs is P(over), outcome is 1 if FT exceeded. hit_rate
    measures how often the model's directional call was right.
    """
    if probs.size == 0:
        return float("nan")
    preds = (probs >= threshold).astype(int)
    return float(np.mean(preds == outcomes.astype(int)))


# ─── Per-market gates ───────────────────────────────────────────────────


MARKET_BRIER_GATES: dict[str, float] = {
    "BTTS_yes": 0.235,
    "BTTS_no": 0.235,
    "O2.5": 0.230,
    "U2.5": 0.230,
    "O3.5": 0.235,
    "corners_O9.5": 0.250,
    "cards_O4.5": 0.250,
    "AH_home_-0.5": 0.235,
    "AH_home_-1.5": 0.250,
}


def market_passes_brier_gate(market: str, brier: float) -> bool:
    """True iff brier <= the configured gate for this market."""
    gate = MARKET_BRIER_GATES.get(market)
    if gate is None:
        return False  # unknown market = fail by default (conservative)
    return brier <= gate


# ─── Container types ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketMetrics:
    """Per-market backtest summary."""

    market: str
    n: int
    """Number of predictions evaluated for this market."""

    brier: float
    ece: float
    hit_rate: float
    mean_predicted: float
    mean_outcome: float
    passes_brier_gate: bool


@dataclass(frozen=True)
class BacktestMetrics:
    """Top-level container of backtest results across all markets."""

    n_fixtures: int
    """Total fixtures attempted (some markets may be missing per fixture)."""

    n_with_predictions: int
    """Fixtures where TSP produced at least one prediction (had usable profile)."""

    coverage_pct: float
    """n_with_predictions / n_fixtures * 100."""

    market_metrics: dict[str, MarketMetrics]

    @property
    def all_markets_pass(self) -> bool:
        if not self.market_metrics:
            return False
        return all(m.passes_brier_gate for m in self.market_metrics.values())

    def summary_lines(self) -> list[str]:
        out = [
            f"Total fixtures: {self.n_fixtures}",
            f"With predictions: {self.n_with_predictions} ({self.coverage_pct:.1f}% coverage)",
            "",
            "Per-market:",
        ]
        for market, mm in sorted(self.market_metrics.items()):
            status = "PASS" if mm.passes_brier_gate else "FAIL"
            out.append(
                f"  {market}: n={mm.n}, Brier={mm.brier:.4f}, "
                f"ECE={mm.ece:.4f}, hit_rate={mm.hit_rate:.2%}, "
                f"E[p]={mm.mean_predicted:.3f}, E[y]={mm.mean_outcome:.3f} → {status}"
            )
        out.append("")
        out.append(f"ALL MARKETS PASS: {self.all_markets_pass}")
        return out
