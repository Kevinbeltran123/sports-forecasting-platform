"""Common interface for goal-prediction models + the GoalsDistribution dataclass.

All match-level goal predictors take two BlendedRates (home, away) and return
a GoalsDistribution exposing the probabilities of common markets.

The GoalsDistribution is the canonical handoff to the match output generator
(Phase 3) — every market price downstream derives from it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bip.evaluation.tournaments.predict.blender import BlendedRates

# Score grid extent — covers >99.9% of football match outcomes.
# (P(X=8) under λ=2.5 is ~3e-5; tails truncated.)
MAX_GOALS = 8


@dataclass(frozen=True)
class GoalsDistribution:
    """Per-match goal-count distribution + derived market probabilities."""

    lambda_home: float
    lambda_away: float

    # Score grid: P(home=i, away=j) for i,j in [0, MAX_GOALS]
    # Stored as immutable tuple-of-tuples (frozen-friendly) — convert to ndarray on demand.
    score_grid: tuple[tuple[float, ...], ...]

    # 1X2 markets
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Derived markets
    p_btts: float
    p_over_2_5: float
    p_under_2_5: float

    # Diagnostics
    expected_total_goals: float
    model_name: str

    def score_grid_as_array(self) -> np.ndarray:
        return np.array(self.score_grid)


class GoalsModel(ABC):
    """ABC for any model that produces a GoalsDistribution from blended rates."""

    name: str

    @abstractmethod
    def predict(
        self,
        home_blend: BlendedRates,
        away_blend: BlendedRates,
    ) -> GoalsDistribution:
        """Return the per-match distribution of goal counts and derived markets."""
        ...


def derive_markets_from_grid(
    grid: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    model_name: str,
) -> GoalsDistribution:
    """Common derivation: take a score grid and emit a GoalsDistribution.

    Args:
        grid: 2D array of shape (MAX_GOALS+1, MAX_GOALS+1) where grid[i,j]
            is P(home=i, away=j). Should sum to ~1.0 (truncation tail
            ignored).
        lambda_home, lambda_away: marginal expected goals (for diagnostics).
        model_name: tag for the resulting distribution.
    """
    n = grid.shape[0]
    p_home = float(np.sum(np.tril(grid, k=-1)))  # i > j  -> home > away
    p_away = float(np.sum(np.triu(grid, k=1)))   # i < j
    p_draw = float(np.sum(np.diag(grid)))

    # Renormalize 1X2 to handle truncation tail (P(score>=N) ~ 0).
    total_1x2 = p_home + p_draw + p_away
    if total_1x2 > 0:
        p_home /= total_1x2
        p_draw /= total_1x2
        p_away /= total_1x2

    # P(BTTS) = 1 - P(home=0) - P(away=0) + P(home=0, away=0)
    p_home_zero = float(np.sum(grid[0, :]))
    p_away_zero = float(np.sum(grid[:, 0]))
    p_both_zero = float(grid[0, 0])
    p_btts_yes = 1.0 - p_home_zero - p_away_zero + p_both_zero
    p_btts_yes = max(0.0, min(1.0, p_btts_yes))

    # Over/Under 2.5: total goals = i + j > 2  (i.e. >= 3)
    over_mass = 0.0
    for i in range(n):
        for j in range(n):
            if i + j >= 3:
                over_mass += grid[i, j]
    p_over = float(over_mass)
    p_under = max(0.0, 1.0 - p_over)

    return GoalsDistribution(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        score_grid=tuple(tuple(row) for row in grid.tolist()),
        p_home_win=p_home,
        p_draw=p_draw,
        p_away_win=p_away,
        p_btts=p_btts_yes,
        p_over_2_5=p_over,
        p_under_2_5=p_under,
        expected_total_goals=lambda_home + lambda_away,
        model_name=model_name,
    )
