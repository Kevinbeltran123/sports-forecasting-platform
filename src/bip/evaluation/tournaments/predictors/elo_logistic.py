"""ELO + logistic baseline for goal predictions.

Sanity baseline for Phase 4 backtest. Uses pre-loaded national-team Elo
ratings (from eloratings.net or a similar source) to derive expected
goal supremacy, then applies a logistic transform.

Calibration:
- Elo difference of 100 -> ~0.5 goal supremacy (roughly the empirical
  rule for international football per eloratings.net 2024 calibration).
- Total goals baseline = 2.7 (median for top international tournaments
  per FIFA aggregate stats 2010-2022).

Output: GoalsDistribution via independent Poisson with the derived λs.

This is intentionally minimalist — it's a SANITY baseline, not a serious
model. The goal of including it in Phase 4 is to verify the Bivariate
Poisson + blended-rates approach beats a simple Elo-driven Poisson.
If it doesn't, the spike's value proposition is undermined and the
project should pivot.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from bip.evaluation.tournaments.predict.blender import BlendedRates
from bip.evaluation.tournaments.predictors.base import (
    MAX_GOALS,
    GoalsDistribution,
    GoalsModel,
    derive_markets_from_grid,
)

NAME = "elo_logistic"

# Empirical international-football constants (eloratings.net documentation).
ELO_PER_GOAL = 100.0  # ~100 Elo difference = 0.5 goal supremacy
DEFAULT_TOTAL_GOALS_BASELINE = 2.55  # WC2018-2022 average + Euro/Copa adjustment


class EloLogisticModel(GoalsModel):
    """Elo-derived expected goals + independent Poisson.

    Constructed with per-team Elo ratings; ignores the BlendedRates input
    EXCEPT to read team_id (uses Elo lookup instead). This means the
    EloLogisticModel is NOT a drop-in replacement for the Bivariate model
    in production paths — it's only for backtest comparison.
    """

    name = NAME

    def __init__(
        self,
        elo_ratings: dict[int, float],
        *,
        total_goals_baseline: float = DEFAULT_TOTAL_GOALS_BASELINE,
        elo_per_goal: float = ELO_PER_GOAL,
    ) -> None:
        if total_goals_baseline <= 0:
            raise ValueError("total_goals_baseline must be > 0")
        if elo_per_goal <= 0:
            raise ValueError("elo_per_goal must be > 0")
        self._elo = elo_ratings
        self._total_goals = total_goals_baseline
        self._elo_per_goal = elo_per_goal

    def predict(
        self,
        home_blend: BlendedRates,
        away_blend: BlendedRates,
    ) -> GoalsDistribution:
        home_elo = self._elo.get(home_blend.team_id)
        away_elo = self._elo.get(away_blend.team_id)
        if home_elo is None or away_elo is None:
            raise KeyError(
                f"Missing Elo for team(s): home={home_blend.team_id} "
                f"({home_elo}), away={away_blend.team_id} ({away_elo})"
            )

        # Goal supremacy from Elo diff.
        elo_diff = home_elo - away_elo
        supremacy = elo_diff / self._elo_per_goal

        # Split total goals into λ_home, λ_away preserving the sum and supremacy.
        # λ_home + λ_away = total_goals; λ_home - λ_away = supremacy.
        lam_h = max(0.05, (self._total_goals + supremacy) / 2.0)
        lam_a = max(0.05, (self._total_goals - supremacy) / 2.0)

        ks = np.arange(MAX_GOALS + 1)
        p_home = stats.poisson.pmf(ks, lam_h)
        p_away = stats.poisson.pmf(ks, lam_a)
        grid = np.outer(p_home, p_away)
        return derive_markets_from_grid(grid, lam_h, lam_a, NAME)


def elo_win_probability(home_elo: float, away_elo: float) -> float:
    """Standard Elo win-probability formula (no draw): 1 / (1 + 10^((Ra - Rb) / 400))."""
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))


def elo_to_goal_supremacy(home_elo: float, away_elo: float, *, elo_per_goal: float = ELO_PER_GOAL) -> float:
    """Convert Elo diff to expected goal supremacy (positive = home favored)."""
    return (home_elo - away_elo) / elo_per_goal


__all__ = [
    "EloLogisticModel",
    "elo_win_probability",
    "elo_to_goal_supremacy",
    "DEFAULT_TOTAL_GOALS_BASELINE",
    "ELO_PER_GOAL",
    "NAME",
]
