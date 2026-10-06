"""Tests for predictor_helpers — weighted_combine + Poisson + bivariate grid."""
from __future__ import annotations

import numpy as np
import pytest

from bip.evaluation.tournaments.team_style_profiler.predictor_helpers import (
    bivariate_poisson_grid,
    p_btts,
    p_home_covers_ah,
    p_home_pushes_ah,
    p_over_total,
    poisson_cdf,
    poisson_pmf,
    poisson_sf,
    weighted_combine,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat


def _d(mean: float, width: float, n: int = 20) -> DistributionStat:
    return DistributionStat(
        mean=mean, ci_low=mean - width / 2, ci_high=mean + width / 2, n=n
    )


class TestWeightedCombine:
    def test_no_sub_returns_general(self) -> None:
        g = _d(1.5, 0.6)
        assert weighted_combine(g, None) == g

    def test_zero_n_sub_returns_general(self) -> None:
        g = _d(1.5, 0.6)
        s = DistributionStat(mean=99.0, ci_low=98, ci_high=100, n=0)
        assert weighted_combine(g, s).mean == 1.5

    def test_narrower_sub_pulls_mean(self) -> None:
        g = _d(mean=1.0, width=1.0)  # wide CI
        s = _d(mean=2.0, width=0.2)  # narrow CI (5x weight)
        c = weighted_combine(g, s)
        # narrow sub dominates
        assert c.mean > 1.5
        assert c.mean < 2.0

    def test_wider_sub_pulled_by_general(self) -> None:
        g = _d(mean=1.0, width=0.2)  # narrow
        s = _d(mean=3.0, width=1.0)  # wide
        c = weighted_combine(g, s)
        # general dominates because it's tighter
        assert c.mean < 2.0
        assert c.mean > 1.0

    def test_n_is_sum(self) -> None:
        g = _d(1, 0.5, n=10)
        s = _d(2, 0.5, n=5)
        c = weighted_combine(g, s)
        assert c.n == 15

    def test_equal_widths_equal_means(self) -> None:
        g = _d(1.0, 0.5, n=20)
        s = _d(3.0, 0.5, n=20)
        c = weighted_combine(g, s)
        # equal widths -> equal weights -> midpoint
        assert c.mean == pytest.approx(2.0)


class TestPoissonPmf:
    def test_zero_lambda(self) -> None:
        assert poisson_pmf(0, 0.0) == 1.0
        assert poisson_pmf(1, 0.0) == 0.0

    def test_at_lambda(self) -> None:
        # P(X=2) for X~Poisson(2) ~ 0.2707
        assert poisson_pmf(2, 2.0) == pytest.approx(0.2707, abs=0.001)

    def test_negative_k_zero(self) -> None:
        assert poisson_pmf(-1, 1.0) == 0.0

    def test_pmf_sums_to_one(self) -> None:
        s = sum(poisson_pmf(k, 2.5) for k in range(20))
        assert s == pytest.approx(1.0, abs=0.0001)


class TestPoissonCdfSf:
    def test_cdf_complementary_sf(self) -> None:
        for lam in [0.5, 1.5, 2.5]:
            for k in range(5):
                assert poisson_cdf(k, lam) + poisson_sf(k, lam) == pytest.approx(1.0, abs=0.001)


class TestBivariatePoissonGrid:
    def test_grid_shape(self) -> None:
        g = bivariate_poisson_grid(1.5, 1.2, rho=0.0)
        assert g.shape == (11, 11)

    def test_grid_sums_to_near_one(self) -> None:
        g = bivariate_poisson_grid(1.5, 1.2, rho=0.0)
        assert g.sum() == pytest.approx(1.0, abs=0.001)

    def test_grid_independent_matches_outer(self) -> None:
        # rho=0 should equal outer(home_pmf, away_pmf)
        lam_h, lam_a = 1.5, 1.2
        g = bivariate_poisson_grid(lam_h, lam_a, rho=0.0)
        h_probs = np.array([poisson_pmf(i, lam_h) for i in range(11)])
        a_probs = np.array([poisson_pmf(j, lam_a) for j in range(11)])
        expected = np.outer(h_probs, a_probs)
        np.testing.assert_allclose(g, expected, atol=1e-10)

    def test_grid_correlated_differs_from_independent(self) -> None:
        g0 = bivariate_poisson_grid(1.5, 1.2, rho=0.0)
        g3 = bivariate_poisson_grid(1.5, 1.2, rho=0.3)
        # With rho>0, P(0,0) should be lower (positive corr increases joint high scores)
        # and diagonal entries higher.
        assert abs(g3.sum() - 1.0) < 0.01
        # Difference exists
        assert not np.allclose(g0, g3)


class TestMarketProbsFromGrid:
    def test_p_over_basic(self) -> None:
        g = bivariate_poisson_grid(1.5, 1.0, rho=0.0)
        p_over25 = p_over_total(g, 2.5)
        # With combined λ=2.5, P(total>2.5) ~ 0.46 from Poisson tables
        assert 0.40 < p_over25 < 0.55

    def test_p_btts_independent(self) -> None:
        g = bivariate_poisson_grid(1.5, 1.5, rho=0.0)
        p = p_btts(g)
        # P(both>=1) when independent Poisson(1.5) each:
        # (1 - exp(-1.5))^2 = (1 - 0.2231)^2 = 0.604
        expected = (1 - np.exp(-1.5)) ** 2
        assert p == pytest.approx(expected, abs=0.01)

    def test_p_home_covers_ah_outright_win(self) -> None:
        # PRIMITIVE semantics: p_home_covers_ah(grid, line) returns
        # P(h - a > line). For "home wins outright", caller passes line=0.5
        # (i.e. h - a must exceed 0.5). With λ=1.5 each, ~37% home outright win.
        g = bivariate_poisson_grid(1.5, 1.5, rho=0.0)
        p = p_home_covers_ah(g, 0.5)
        assert 0.30 < p < 0.40

    def test_p_home_covers_ah_minus_0_5_no_loss(self) -> None:
        # PRIMITIVE called with line=-0.5 means P(h - a > -0.5) = P(h >= a),
        # i.e. home doesn't lose. With balanced λ, this is ~62%.
        g = bivariate_poisson_grid(1.5, 1.5, rho=0.0)
        p = p_home_covers_ah(g, -0.5)
        assert 0.55 < p < 0.70

    def test_p_home_pushes_only_whole_line(self) -> None:
        g = bivariate_poisson_grid(1.5, 1.5, rho=0.0)
        # Half-line never pushes
        assert p_home_pushes_ah(g, -0.5) == 0.0
        # Whole-line might push
        assert p_home_pushes_ah(g, 0.0) > 0.0  # exact draw count
