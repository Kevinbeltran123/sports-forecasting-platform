"""Bivariate-Poisson goals model with closed-form correlation parameter.

The Bivariate Poisson distribution captures positive correlation between
home and away goal counts via a shared component λ12::

    X = X1 + X12,  Y = X2 + X12   where X1, X2, X12 ~ independent Poisson

So:
- mean(X) = λ1 + λ12      (= our target λ_home)
- mean(Y) = λ2 + λ12      (= our target λ_away)
- cov(X, Y) = λ12         (correlation parameter)

Given target marginal means μ_h, μ_a from the blender and a Pearson
correlation ρ, we set::

    λ12 = ρ · sqrt(μ_h · μ_a)
    λ1  = μ_h − λ12
    λ2  = μ_a − λ12

with the constraint λ12 ≤ min(μ_h, μ_a) (else λ1 or λ2 would go negative).

For football, Dixon & Coles (1997) found ρ ≈ 0.05 — small but non-zero.
The default ρ = 0.04 is conservative; backtest may tune it. ρ = 0
collapses to independent Poisson, so this model is a strict generalization.

PMF::

    P(X=x, Y=y) =
      e^{-(λ1+λ2+λ12)}
      · (λ1^x / x!)
      · (λ2^y / y!)
      · Σ_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (λ12 / (λ1·λ2))^k

(See Karlis & Ntzoufras 2003.)
"""

from __future__ import annotations

import math

import numpy as np

from bip.evaluation.tournaments.predictors.base import (
    MAX_GOALS,
    GoalsDistribution,
    GoalsModel,
    derive_markets_from_grid,
)
from bip.evaluation.tournaments.predict.blender import BlendedRates

NAME = "bivariate_poisson"
DEFAULT_RHO = 0.04


class BivariatePoissonModel(GoalsModel):
    """Bivariate Poisson with shared-rate correlation parameter."""

    name = NAME

    def __init__(self, rho: float = DEFAULT_RHO) -> None:
        if not -0.5 <= rho <= 0.5:
            raise ValueError(f"rho must be in [-0.5, 0.5], got {rho}")
        self._rho = rho

    @property
    def rho(self) -> float:
        return self._rho

    def predict(
        self,
        home_blend: BlendedRates,
        away_blend: BlendedRates,
    ) -> GoalsDistribution:
        mu_h = max(home_blend.expected_goals_per90, 1e-6)
        mu_a = max(away_blend.expected_goals_per90, 1e-6)

        # Compute λ12 = ρ * sqrt(μ_h * μ_a), clamped so λ1, λ2 stay non-negative.
        lam12 = self._rho * math.sqrt(mu_h * mu_a)
        lam12 = max(0.0, min(lam12, min(mu_h, mu_a) - 1e-9))
        lam1 = mu_h - lam12
        lam2 = mu_a - lam12

        grid = _bivariate_poisson_grid(lam1, lam2, lam12, MAX_GOALS)
        return derive_markets_from_grid(grid, mu_h, mu_a, NAME)


def _bivariate_poisson_grid(
    lam1: float,
    lam2: float,
    lam12: float,
    max_n: int,
) -> np.ndarray:
    """Build the (max_n+1) × (max_n+1) score-grid of the Bivariate Poisson.

    Implementation note: the inner sum can be unstable when λ12 is small
    (the (λ12/(λ1·λ2))^k factor is tiny). We compute term-by-term with
    iterative ratios to avoid factorial blow-up — manageable for max_n ≤ 8.
    """
    base_exp = math.exp(-(lam1 + lam2 + lam12))
    grid = np.zeros((max_n + 1, max_n + 1))
    correction_ratio = lam12 / (lam1 * lam2) if lam1 > 0 and lam2 > 0 else 0.0

    # Pre-compute lam1^x / x! and lam2^y / y! tables.
    lam1_pow_over_fact = [1.0]
    lam2_pow_over_fact = [1.0]
    for k in range(1, max_n + 1):
        lam1_pow_over_fact.append(lam1_pow_over_fact[-1] * lam1 / k)
        lam2_pow_over_fact.append(lam2_pow_over_fact[-1] * lam2 / k)

    for x in range(max_n + 1):
        for y in range(max_n + 1):
            # Inner sum: Σ_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (λ12/(λ1·λ2))^k
            inner = 0.0
            for k in range(min(x, y) + 1):
                term = (
                    math.comb(x, k)
                    * math.comb(y, k)
                    * math.factorial(k)
                    * (correction_ratio**k)
                )
                inner += term
            grid[x, y] = (
                base_exp
                * lam1_pow_over_fact[x]
                * lam2_pow_over_fact[y]
                * inner
            )

    return grid
