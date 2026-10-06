"""Numerical helpers for the cross-team predictor.

Two responsibilities:
  1. ``weighted_combine``: merge general + sub-profile DistributionStats
     using inverse-CI-width weighting (per operator decision: narrower CI
     means more reliable estimate, gets more weight).
  2. Poisson + Bivariate Poisson PMF utilities — used to derive market
     probabilities (BTTS, O/U, AH) from per-team goal rates.
"""
from __future__ import annotations

from math import exp, factorial, lgamma

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat


# ─── Weighted combine (inverse CI width) ────────────────────────────────


def weighted_combine(
    general: DistributionStat,
    sub: DistributionStat | None,
    epsilon: float = 1e-6,
) -> DistributionStat:
    """Combine a general TSV stat with a sub-profile stat by inverse CI width.

    Operator decision (locked 2026-05-24): narrower CI gets more weight
    because narrower CI = more samples / more consistent estimate.

    Algorithm:
      w_general = 1 / (general.ci_width + epsilon)
      w_sub     = 1 / (sub.ci_width + epsilon)
      mean      = (w_general * general.mean + w_sub * sub.mean) / (w_general + w_sub)
      n         = general.n + sub.n  (combined effective sample)
      ci_*      = weighted mix of the CI bounds

    When sub is None, returns general unchanged.

    >>> from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat
    >>> g = DistributionStat(mean=1.5, ci_low=1.0, ci_high=2.0, n=20)  # width=1.0
    >>> s = DistributionStat(mean=2.0, ci_low=1.8, ci_high=2.2, n=5)   # width=0.4 (narrower)
    >>> c = weighted_combine(g, s)
    >>> # sub should pull mean toward 2.0 because its CI is narrower
    >>> c.mean > 1.75
    True
    """
    if sub is None or sub.n == 0:
        return general
    if general.n == 0:
        return sub

    w_g = 1.0 / (general.ci_width + epsilon)
    w_s = 1.0 / (sub.ci_width + epsilon)
    total_w = w_g + w_s

    mean = (w_g * general.mean + w_s * sub.mean) / total_w
    ci_low = (w_g * general.ci_low + w_s * sub.ci_low) / total_w
    ci_high = (w_g * general.ci_high + w_s * sub.ci_high) / total_w

    return DistributionStat(
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        n=general.n + sub.n,
    )


# ─── Poisson PMF helpers ────────────────────────────────────────────────


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    # Use log-gamma for numerical stability with larger k
    return float(np.exp(-lam + k * np.log(lam) - lgamma(k + 1)))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)."""
    if k < 0:
        return 0.0
    return float(sum(poisson_pmf(i, lam) for i in range(k + 1)))


def poisson_sf(k: int, lam: float) -> float:
    """P(X > k) = 1 - CDF(k)."""
    return 1.0 - poisson_cdf(k, lam)


# ─── Bivariate Poisson grid ─────────────────────────────────────────────


def bivariate_poisson_grid(
    lam_home: float,
    lam_away: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """Return an (max_goals+1) x (max_goals+1) matrix P[h, a] for the
    joint distribution of (home_goals, away_goals).

    rho=0.0 -> simple independent Poisson product (matches lock_v1).
    rho>0.0 -> uses the Karlis-Ntzoufras bivariate Poisson:
        X = X1 + X3, Y = X2 + X3
        X1 ~ Poisson(λ_h - λ_3)
        X2 ~ Poisson(λ_a - λ_3)
        X3 ~ Poisson(λ_3) where λ_3 = rho * sqrt(λ_h * λ_a)
    rho<0.0 -> equivalent of a negative correlation; we DO NOT implement
        the Sarmanov form here (out of scope per v3 spike findings).
        Clamped to 0.0.

    Args:
        lam_home: home team goal rate (mean goals scored by home).
        lam_away: away team goal rate.
        rho: correlation parameter. Default 0.0 = independent.
        max_goals: matrix dimension. 10 covers >99.9% of realistic
                   football scores.

    Returns:
        2D numpy array of shape (max_goals+1, max_goals+1) summing to ~1.
    """
    if rho < 0.0:
        rho = 0.0  # NOT supported; clamp
    lam_home = max(lam_home, 0.001)
    lam_away = max(lam_away, 0.001)

    if rho == 0.0:
        h_probs = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)])
        a_probs = np.array([poisson_pmf(j, lam_away) for j in range(max_goals + 1)])
        return np.outer(h_probs, a_probs)

    # Karlis-Ntzoufras bivariate Poisson with positive correlation.
    lam3 = rho * (lam_home * lam_away) ** 0.5
    lam3 = min(lam3, lam_home - 0.001, lam_away - 0.001)  # constraint
    lam1 = lam_home - lam3
    lam2 = lam_away - lam3

    grid = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            # P(X1=h-k) * P(X2=a-k) * P(X3=k) summed over valid k
            total = 0.0
            for k in range(min(h, a) + 1):
                total += (
                    poisson_pmf(h - k, lam1)
                    * poisson_pmf(a - k, lam2)
                    * poisson_pmf(k, lam3)
                )
            grid[h, a] = total
    return grid


# ─── Market probability from grid ───────────────────────────────────────


def p_over_total(grid: np.ndarray, line: float) -> float:
    """P(total_goals > line) from a bivariate scoreline grid."""
    n = grid.shape[0]
    p = 0.0
    for h in range(n):
        for a in range(n):
            if (h + a) > line:
                p += grid[h, a]
    return float(p)


def p_btts(grid: np.ndarray) -> float:
    """P(both teams score >= 1) from a bivariate scoreline grid."""
    n = grid.shape[0]
    p = 0.0
    for h in range(1, n):
        for a in range(1, n):
            p += grid[h, a]
    return float(p)


def p_home_covers_ah(grid: np.ndarray, line: float) -> float:
    """P(home_goals - away_goals > line) for Asian handicap.

    line is from the home team's perspective (negative line means home
    is favorite needing to win by that margin).

    For half-line AH (e.g., -1.5, -2.5), the bet either fully wins or fully
    loses — no push. We compute the strict-greater probability.

    For whole-line AH (e.g., -1.0), the bet pushes when home_goals -
    away_goals == abs(line). This function returns the WIN probability
    (excludes push); ``p_home_pushes_ah`` returns the push.
    """
    n = grid.shape[0]
    p = 0.0
    for h in range(n):
        for a in range(n):
            if (h - a) > line:
                p += grid[h, a]
    return float(p)


def p_home_pushes_ah(grid: np.ndarray, line: float) -> float:
    """P(home_goals - away_goals == line) — only meaningful for whole-line AH."""
    if line != int(line):
        return 0.0  # half-line AH never pushes
    n = grid.shape[0]
    p = 0.0
    for h in range(n):
        for a in range(n):
            if (h - a) == int(line):
                p += grid[h, a]
    return float(p)
