"""MLE fitter for DIBP parameters (ρ, π_diag, θ_diag) on a calibration corpus.

Takes a validation/calibration DataFrame of matches with known team
strengths from the Ola 1 ``WeightedMLEResult`` and finds the (ρ, π_diag,
θ_diag) that maximise the per-match Poisson likelihood under the DIBP
family.

Important boundary handling:
- L-BFGS-B bounds: ρ ∈ [-0.5, 0.5], π_diag ∈ [0, 0.95], θ_diag ∈ [0, 0.95].
  We use 0.95 not 1.0 on the upper bounds because the geometric J(·;θ)
  becomes ill-conditioned as θ → 1.
- The fit only uses rows where *both* teams appear in the strength prior.
  Matches with a cold-start team are skipped (the predictor falls back to
  cohort priors at predict time — out of scope for the fitter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .dibp import DIBPParams, compute_dibp_grid
from .weighted_strength import WeightedMLEResult


@dataclass(frozen=True)
class DIBPFitResult:
    """Output of fit_dibp."""

    params: DIBPParams
    n_matches_used: int
    log_likelihood: float
    converged: bool

    def __repr__(self) -> str:
        return (
            "DIBPFitResult("
            f"rho={self.params.rho:.4f}, "
            f"pi_diag={self.params.pi_diag:.4f}, "
            f"theta_diag={self.params.theta_diag:.4f}, "
            f"n={self.n_matches_used}, "
            f"ll={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )


def _per_match_lambdas(
    matches: pl.DataFrame,
    strengths: WeightedMLEResult,
    home_col: str,
    away_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """For each match, look up (μ_h, μ_a) from the weighted-MLE prior.

    Returns (mu_h, mu_a, home_goals, away_goals, n_used). Drops matches
    where either team is missing from the prior.
    """

    mu_h_list: list[float] = []
    mu_a_list: list[float] = []
    hg_list: list[int] = []
    ag_list: list[int] = []
    rows = matches.iter_rows(named=True)
    for row in rows:
        h, a = row[home_col], row[away_col]
        if h not in strengths.strengths or a not in strengths.strengths:
            continue
        try:
            lh, la = strengths.predict_lambdas(h, a)
        except KeyError:
            continue
        mu_h_list.append(lh)
        mu_a_list.append(la)
        hg_list.append(int(row["home_goals"]))
        ag_list.append(int(row["away_goals"]))
    return (
        np.array(mu_h_list, dtype=np.float64),
        np.array(mu_a_list, dtype=np.float64),
        np.array(hg_list, dtype=np.int64),
        np.array(ag_list, dtype=np.int64),
        len(mu_h_list),
    )


def fit_dibp(
    matches: pl.DataFrame,
    strengths: WeightedMLEResult,
    *,
    home_col: str = "home_team",
    away_col: str = "away_team",
    max_goals: int = 8,
    init_params: DIBPParams = DIBPParams(rho=0.0, pi_diag=0.05, theta_diag=0.5),
    max_iter: int = 100,
) -> DIBPFitResult:
    """Maximum likelihood estimate of (ρ, π_diag, θ_diag).

    Matches without both teams in ``strengths`` are dropped; we report
    the n_matches_used so the caller can verify the fit corpus was large
    enough. Below 30 matches the result should not be used in production.
    """

    mu_h, mu_a, home_goals, away_goals, n_used = _per_match_lambdas(
        matches, strengths, home_col=home_col, away_col=away_col
    )

    if n_used == 0:
        raise ValueError(
            "No matches in the calibration corpus had both teams in the strength prior."
        )

    # Cap goals at max_goals so they're addressable in the score grid.
    home_goals = np.clip(home_goals, 0, max_goals)
    away_goals = np.clip(away_goals, 0, max_goals)

    def neg_log_likelihood(theta: np.ndarray) -> float:
        rho, pi_diag, theta_diag = float(theta[0]), float(theta[1]), float(theta[2])
        try:
            params = DIBPParams(rho=rho, pi_diag=pi_diag, theta_diag=theta_diag)
        except ValueError:
            return 1e9
        total = 0.0
        for i in range(n_used):
            grid = compute_dibp_grid(mu_h[i], mu_a[i], params, max_goals=max_goals)
            p_ij = grid[home_goals[i], away_goals[i]]
            if p_ij <= 0:
                return 1e9
            total += math.log(p_ij)
        return -total

    x0 = np.array([init_params.rho, init_params.pi_diag, init_params.theta_diag], dtype=np.float64)
    # Note on ρ bounds: the underlying BP grid clamps λ12 = max(0, ρ·sqrt(μ_h·μ_a)),
    # so ρ < 0 produces an identical grid to ρ = 0. We restrict the search to
    # ρ ≥ 0 to prevent L-BFGS-B from wandering across the flat gradient region.
    # Anti-correlation between home and away goals would need a copula-style
    # formulation (Sarmanov family) — out of scope for this spike.
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=[(0.0, 0.5), (0.0, 0.95), (0.0, 0.95)],
        options={"maxiter": max_iter, "ftol": 1e-7},
    )

    final_params = DIBPParams(
        rho=float(result.x[0]),
        pi_diag=float(result.x[1]),
        theta_diag=float(result.x[2]),
    )
    return DIBPFitResult(
        params=final_params,
        n_matches_used=n_used,
        log_likelihood=-float(result.fun),
        converged=bool(result.success),
    )
