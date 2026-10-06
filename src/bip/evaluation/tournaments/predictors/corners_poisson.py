"""Corners prediction model — Independent Poisson per team.

Corners between teams are less correlated than goals, so independent
Poisson per side is a sound first approximation.

Input : BlendedRates.expected_corners_for_per90 from each team.
Output: CornersDistribution with O/U, team-superiority, Asian handicap,
        and naive first-half markets (40% of full-match λ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from bip.evaluation.tournaments.predict.blender import BlendedRates

MAX_CORNERS = 25  # covers >99.9% of corner totals under λ ≤ 12
FIRST_HALF_FRACTION = 0.40  # historical: ~40% of corners in first half
MIN_LAMBDA = 0.5  # floor so PMF is always well-defined


@dataclass(frozen=True)
class CornersDistribution:
    """Per-match corners distribution and derived market probabilities."""

    lambda_home: float
    lambda_away: float
    lambda_total: float

    # Standard O/U markets
    p_over_8_5: float
    p_over_9_5: float
    p_over_10_5: float
    p_over_11_5: float
    p_under_8_5: float
    p_under_9_5: float
    p_under_10_5: float
    p_under_11_5: float

    # Team superiority
    p_home_more: float
    p_away_more: float
    p_equal_corners: float

    # Asian handicap corners (home perspective)
    p_home_ahc_minus_1_5: float   # home wins corners by 2+
    p_away_ahc_minus_1_5: float   # away wins corners by 2+
    p_home_ahc_plus_1_5: float    # home doesn't lose corners by 2+

    # First half (naive: FIRST_HALF_FRACTION * λ_full)
    lambda_home_fh: float
    lambda_away_fh: float
    p_total_fh_over_4_5: float
    p_total_fh_over_5_5: float


class CornersPoissonModel:
    """Independent Poisson model for corners.

    Uses each team's blended expected_corners_for_per90 as λ.
    Independent Poisson is justified because inter-team corner correlation
    is much weaker than for goals.
    """

    name = "corners_independent_poisson"

    def predict(
        self,
        home_blend: BlendedRates,
        away_blend: BlendedRates,
    ) -> CornersDistribution:
        λ_h = max(MIN_LAMBDA, home_blend.expected_corners_for_per90)
        λ_a = max(MIN_LAMBDA, away_blend.expected_corners_for_per90)
        λ_t = λ_h + λ_a

        total_pmf = _poisson_pmf(λ_t, MAX_CORNERS)
        grid = np.outer(_poisson_pmf(λ_h, MAX_CORNERS), _poisson_pmf(λ_a, MAX_CORNERS))
        n = grid.shape[0]

        def p_over(line: float) -> float:
            # line is always X.5 → threshold = int(line) + 1
            threshold = int(line) + 1
            if threshold > MAX_CORNERS:
                return 0.0
            return float(np.sum(total_pmf[threshold:]))

        # Team superiority from joint grid
        p_h_more = float(np.sum(np.tril(grid, k=-1)))  # home > away
        p_a_more = float(np.sum(np.triu(grid, k=1)))   # away > home
        p_equal = float(np.sum(np.diag(grid)))

        # Asian handicap
        p_home_ahc = float(sum(
            grid[i, j] for i in range(n) for j in range(n) if i - j >= 2
        ))
        p_away_ahc = float(sum(
            grid[i, j] for i in range(n) for j in range(n) if j - i >= 2
        ))
        # Home +1.5: home doesn't lose corners by 2+ (complement of away -1.5)
        p_home_plus_1_5 = 1.0 - p_away_ahc

        # First half
        λ_h_fh = λ_h * FIRST_HALF_FRACTION
        λ_a_fh = λ_a * FIRST_HALF_FRACTION
        fh_pmf = _poisson_pmf(λ_h_fh + λ_a_fh, MAX_CORNERS)

        def p_over_fh(line: float) -> float:
            threshold = int(line) + 1
            if threshold > MAX_CORNERS:
                return 0.0
            return float(np.sum(fh_pmf[threshold:]))

        return CornersDistribution(
            lambda_home=λ_h,
            lambda_away=λ_a,
            lambda_total=λ_t,
            p_over_8_5=p_over(8.5),
            p_over_9_5=p_over(9.5),
            p_over_10_5=p_over(10.5),
            p_over_11_5=p_over(11.5),
            p_under_8_5=1.0 - p_over(8.5),
            p_under_9_5=1.0 - p_over(9.5),
            p_under_10_5=1.0 - p_over(10.5),
            p_under_11_5=1.0 - p_over(11.5),
            p_home_more=p_h_more,
            p_away_more=p_a_more,
            p_equal_corners=p_equal,
            p_home_ahc_minus_1_5=p_home_ahc,
            p_away_ahc_minus_1_5=p_away_ahc,
            p_home_ahc_plus_1_5=p_home_plus_1_5,
            lambda_home_fh=λ_h_fh,
            lambda_away_fh=λ_a_fh,
            p_total_fh_over_4_5=p_over_fh(4.5),
            p_total_fh_over_5_5=p_over_fh(5.5),
        )


def _poisson_pmf(lam: float, max_k: int) -> np.ndarray:
    """PMF array P(X=k) for k in [0, max_k] under Poisson(lam)."""
    return np.array([poisson.pmf(k, lam) for k in range(max_k + 1)])
