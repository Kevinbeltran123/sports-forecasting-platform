"""Backtest infrastructure for Team Style Profiler.

Validates TSP predictions against historical outcomes from
AFCON23 + Copa24 + Euro24 (n~135 matches, post-2023 era so DT-filter
is informative).

OPERATOR DECISIONS (locked 2026-05-24):
  - Scope: AFCON23 + Copa24 + Euro24 only
  - Odds: Brier-implied approximation (no real bookmaker odds available
    historically without separate subscription)
  - Calibration: Isotonic regression per market (non-parametric)
  - Success: ROI flat >=+5% on edge>=3% picks AND Brier <= 0.235 BTTS

LIMITATION: with Brier-implied odds, "ROI" is not real — it's a
self-referential metric. The honest signal is Brier + ECE per market.
ROI requires real historical Pinnacle odds (out of scope per
operator decision).

Modules:
  metrics            Brier / ECE / coverage / hit-rate primitives
  isotonic_calibrator Per-market isotonic regression
  runner             End-to-end orchestrator (loads outcomes, calls TSP,
                     produces backtest results)
  report             Markdown report generator
"""

from bip.evaluation.tournaments.team_style_profiler.backtest.metrics import (
    BacktestMetrics,
    MarketMetrics,
    brier_score,
    classwise_ece,
    expected_calibration_error,
    hit_rate,
)

__all__ = [
    "BacktestMetrics",
    "MarketMetrics",
    "brier_score",
    "classwise_ece",
    "expected_calibration_error",
    "hit_rate",
]
