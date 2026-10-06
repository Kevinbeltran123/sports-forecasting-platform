"""Diagonal-Inflated Bivariate Poisson (DIBP) — Karlis & Ntzoufras (2003).

Implements

    P(X=x, Y=y) =
        (1 − π) · BivPois(x, y; λ1, λ2, λ12)              if x ≠ y
        (1 − π) · BivPois(k, k; λ1, λ2, λ12) + π · J(k; θ) if x = y = k

with J(k; θ) = (1 − θ) · θ^k, the geometric-tail diagonal mass. The
existing ``BivariatePoissonModel`` in ``predictors/bivariate_poisson.py``
is the π = 0 case; DIBP adds two parameters (π, θ) that absorb the
excess draw mass football scorelines exhibit relative to plain BP.

Reference: Karlis & Ntzoufras (2003), *Analysis of sports data by using
bivariate Poisson models*. JRSS-D 52(3): 381–393.
DOI: 10.1111/1467-9884.00366.

Why a standalone module rather than extending ``BivariatePoissonModel``:
- Existing model is wrapped by the lock_v1 emitter and the live engine;
  D-09 freeze rules prohibit touching it.
- DIBP is parameterised on top of the *same* BP grid, so we reuse
  ``_bivariate_poisson_grid`` from the existing module rather than
  reimplementing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bip.evaluation.tournaments.predictors.bivariate_poisson import _bivariate_poisson_grid


@dataclass(frozen=True)
class DIBPParams:
    """Three parameters of the DIBP family.

    - ``rho`` ∈ [-0.5, 0.5]: BP correlation (λ12 = ρ · sqrt(μ_h · μ_a)).
      ρ = 0 makes the underlying BP independent.
    - ``pi_diag`` ∈ [0, 1]: weight on the diagonal-mass mixture component.
      0 collapses DIBP to plain BP.
    - ``theta_diag`` ∈ [0, 1): geometric decay of the diagonal mass.
      ``J(k; θ) = (1 − θ) · θ^k`` — higher θ pushes mass to higher draws
      (1-1, 2-2 ...), lower θ concentrates it on 0-0.
    """

    rho: float
    pi_diag: float
    theta_diag: float

    def __post_init__(self) -> None:
        if not -0.5 <= self.rho <= 0.5:
            raise ValueError(f"rho must be in [-0.5, 0.5], got {self.rho}")
        if not 0.0 <= self.pi_diag <= 1.0:
            raise ValueError(f"pi_diag must be in [0, 1], got {self.pi_diag}")
        if not 0.0 <= self.theta_diag < 1.0:
            raise ValueError(f"theta_diag must be in [0, 1), got {self.theta_diag}")


def _diagonal_mass(max_n: int, theta: float) -> np.ndarray:
    """Geometric diagonal distribution J(k; θ) = (1 − θ) · θ^k for k=0..max_n.

    Truncated to max_n; we renormalise the whole DIBP grid downstream so
    truncation error is absorbed there.
    """

    powers = np.arange(max_n + 1, dtype=np.float64)
    return (1.0 - theta) * (theta**powers)


def compute_dibp_grid(
    mu_h: float,
    mu_a: float,
    params: DIBPParams,
    max_goals: int = 8,
) -> np.ndarray:
    """Build the (max_goals+1) × (max_goals+1) DIBP score grid.

    Returns a 2-D ndarray with grid[x, y] = P(home=x, away=y) under the
    DIBP family. Renormalised so it sums exactly to 1.0 to absorb the
    grid-truncation error.
    """

    mu_h = max(mu_h, 1e-6)
    mu_a = max(mu_a, 1e-6)

    lam12 = params.rho * math.sqrt(mu_h * mu_a)
    lam12 = max(0.0, min(lam12, min(mu_h, mu_a) - 1e-9))
    lam1 = mu_h - lam12
    lam2 = mu_a - lam12

    bp_grid = _bivariate_poisson_grid(lam1, lam2, lam12, max_goals)
    diag = _diagonal_mass(max_goals, params.theta_diag)

    grid = (1.0 - params.pi_diag) * bp_grid
    # Add inflation on the diagonal.
    for k in range(max_goals + 1):
        grid[k, k] += params.pi_diag * diag[k]

    # Renormalise — both BP truncation and J truncation can leave a tiny gap.
    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


@dataclass(frozen=True)
class DIBPMarkets:
    """Standard market probabilities derived from a DIBP score grid."""

    p_home_win: float
    p_draw: float
    p_away_win: float
    p_btts: float
    p_over_2_5: float
    p_under_2_5: float
    expected_total_goals: float

    def as_dict(self) -> dict[str, float]:
        return {
            "p_home_win": self.p_home_win,
            "p_draw": self.p_draw,
            "p_away_win": self.p_away_win,
            "p_btts": self.p_btts,
            "p_over_2_5": self.p_over_2_5,
            "p_under_2_5": self.p_under_2_5,
            "expected_total_goals": self.expected_total_goals,
        }


def markets_from_grid(grid: np.ndarray) -> DIBPMarkets:
    """Derive 1X2 / BTTS / OU2.5 probabilities from a score grid.

    Conventions:
    - grid[x, y] = P(home = x, away = y)
    - home win = strict upper triangle (x > y); draw = main diagonal;
      away win = strict lower triangle (x < y).
    - BTTS = sum over x ≥ 1 ∧ y ≥ 1.
    - Over 2.5 = sum over x + y ≥ 3.
    """

    n_rows, n_cols = grid.shape
    x_idx = np.arange(n_rows).reshape(-1, 1)
    y_idx = np.arange(n_cols).reshape(1, -1)

    home_win_mask = x_idx > y_idx
    draw_mask = x_idx == y_idx
    away_win_mask = x_idx < y_idx
    btts_mask = (x_idx >= 1) & (y_idx >= 1)
    over_mask = (x_idx + y_idx) >= 3

    p_home = float(grid[home_win_mask].sum())
    p_draw = float(grid[draw_mask].sum())
    p_away = float(grid[away_win_mask].sum())
    p_btts = float(grid[btts_mask].sum())
    p_over = float(grid[over_mask].sum())
    p_under = 1.0 - p_over

    # Expected total goals = sum over (x+y) · P(x, y).
    total_grid = (x_idx + y_idx) * grid
    expected_total = float(total_grid.sum())

    return DIBPMarkets(
        p_home_win=p_home,
        p_draw=p_draw,
        p_away_win=p_away,
        p_btts=p_btts,
        p_over_2_5=p_over,
        p_under_2_5=p_under,
        expected_total_goals=expected_total,
    )
