"""Independent-Poisson goals model.

Assumes home and away goal counts are independent Poissons with rates from
the blender's expected_goals_per90. The match length is 90 minutes so
λ = expected_goals_per90 directly (no scaling).

This is the simplest possible model and serves as the baseline against
which BivariatePoisson and ELO+logistic are compared in Phase 4.

Independence is a known wrong assumption for football — pairs of low-scoring
games (0-0, 1-1) cluster more than independence implies. Dixon-Coles applies
a correction, BivariatePoisson uses a shared rate parameter. Both are
typically 1-3% better than independent on log-loss for top European leagues.
For national-team tournaments where sample is sparse, the difference may be
smaller — Phase 4 will tell us.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from bip.evaluation.tournaments.predictors.base import (
    MAX_GOALS,
    GoalsDistribution,
    GoalsModel,
    derive_markets_from_grid,
)
from bip.evaluation.tournaments.predict.blender import BlendedRates

NAME = "independent_poisson"


class IndependentPoissonModel(GoalsModel):
    """Goals as independent Poisson(λ_home), Poisson(λ_away)."""

    name = NAME

    def predict(
        self,
        home_blend: BlendedRates,
        away_blend: BlendedRates,
    ) -> GoalsDistribution:
        lam_h = max(home_blend.expected_goals_per90, 1e-6)
        lam_a = max(away_blend.expected_goals_per90, 1e-6)

        ks = np.arange(MAX_GOALS + 1)
        p_home = stats.poisson.pmf(ks, lam_h)
        p_away = stats.poisson.pmf(ks, lam_a)

        # Outer product -> joint independent grid.
        grid = np.outer(p_home, p_away)
        return derive_markets_from_grid(grid, lam_h, lam_a, NAME)


def expected_count_to_pmf(lam: float, max_n: int = 12) -> np.ndarray:
    """Helper for non-goals count markets (corners, SoT).

    Returns the Poisson PMF over [0, max_n]. Used by Phase 3 output generator
    to derive Over/Under prices for any count market given a single λ.
    """
    if lam <= 0:
        out = np.zeros(max_n + 1)
        out[0] = 1.0
        return out
    ks = np.arange(max_n + 1)
    return stats.poisson.pmf(ks, lam)
