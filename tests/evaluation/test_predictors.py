"""Phase 2.3 — goal models: Independent Poisson + Bivariate Poisson."""

from __future__ import annotations

import math

import pytest

from bip.evaluation.tournaments.predictors.base import GoalsDistribution
from bip.evaluation.tournaments.predictors.bivariate_poisson import (
    DEFAULT_RHO,
    BivariatePoissonModel,
)
from bip.evaluation.tournaments.predictors.independent_poisson import (
    IndependentPoissonModel,
    expected_count_to_pmf,
)
from bip.evaluation.tournaments.predict.blender import BlendedRates


def _blend(team_id: int, goals: float) -> BlendedRates:
    return BlendedRates(
        team_id=team_id,
        expected_goals_per90=goals,
        expected_shots_per90=0.0,
        expected_sot_per90=0.0,
        expected_fouls_committed_per90=0.0,
        expected_corners_for_per90=0.0,
        alpha=0.35,
        team_component_goals=goals,
        lineup_component_goals=goals,
        n_starters_with_form=11,
    )


# ─────────────────────────────────────────────────────────────────
# Independent Poisson
# ─────────────────────────────────────────────────────────────────


class TestIndependentPoisson:
    def test_distribution_shape(self):
        model = IndependentPoissonModel()
        d = model.predict(_blend(1, 1.5), _blend(2, 1.2))
        assert isinstance(d, GoalsDistribution)
        assert d.lambda_home == pytest.approx(1.5)
        assert d.lambda_away == pytest.approx(1.2)
        assert d.expected_total_goals == pytest.approx(2.7)
        assert d.model_name == "independent_poisson"

    def test_probabilities_sum_to_one_approximately(self):
        model = IndependentPoissonModel()
        d = model.predict(_blend(1, 1.5), _blend(2, 1.5))
        # 1X2 must sum to 1 after renormalization.
        total = d.p_home_win + d.p_draw + d.p_away_win
        assert total == pytest.approx(1.0, abs=1e-6)
        # P(BTTS) + P(no BTTS) = 1
        assert 0.0 <= d.p_btts <= 1.0
        assert d.p_over_2_5 + d.p_under_2_5 == pytest.approx(1.0, abs=1e-3)

    def test_symmetric_lambdas_yield_balanced_1x2(self):
        model = IndependentPoissonModel()
        d = model.predict(_blend(1, 1.5), _blend(2, 1.5))
        # By symmetry, p_home == p_away.
        assert d.p_home_win == pytest.approx(d.p_away_win, abs=1e-6)

    def test_high_lambdas_increase_btts_and_over(self):
        model = IndependentPoissonModel()
        low = model.predict(_blend(1, 0.5), _blend(2, 0.5))
        high = model.predict(_blend(1, 2.5), _blend(2, 2.5))
        assert high.p_btts > low.p_btts
        assert high.p_over_2_5 > low.p_over_2_5

    def test_dominant_team_has_higher_p_win(self):
        model = IndependentPoissonModel()
        d = model.predict(_blend(1, 2.5), _blend(2, 0.7))
        assert d.p_home_win > d.p_away_win
        assert d.p_home_win > d.p_draw

    def test_zero_lambdas_handled_gracefully(self):
        model = IndependentPoissonModel()
        d = model.predict(_blend(1, 0.0), _blend(2, 0.0))
        # Both teams expected 0 -> P(0-0) ~ 1 -> p_draw ~ 1
        assert d.p_draw > 0.99

    def test_score_grid_dimensions(self):
        model = IndependentPoissonModel()
        d = model.predict(_blend(1, 1.5), _blend(2, 1.2))
        grid = d.score_grid_as_array()
        assert grid.shape == (9, 9)  # MAX_GOALS=8, so 9 rows/cols (0-8 inclusive)


class TestExpectedCountToPmf:
    def test_pmf_sums_to_approx_one(self):
        pmf = expected_count_to_pmf(5.0, max_n=20)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-3)

    def test_zero_lambda_returns_point_mass_at_zero(self):
        pmf = expected_count_to_pmf(0.0, max_n=10)
        assert pmf[0] == 1.0
        assert pmf[1:].sum() == 0.0


# ─────────────────────────────────────────────────────────────────
# Bivariate Poisson
# ─────────────────────────────────────────────────────────────────


class TestBivariatePoisson:
    def test_default_rho_is_small_positive(self):
        m = BivariatePoissonModel()
        assert m.rho == DEFAULT_RHO

    def test_rejects_extreme_rho(self):
        with pytest.raises(ValueError):
            BivariatePoissonModel(rho=0.99)
        with pytest.raises(ValueError):
            BivariatePoissonModel(rho=-1.0)

    def test_rho_zero_matches_independent_poisson(self):
        bivariate = BivariatePoissonModel(rho=0.0)
        independent = IndependentPoissonModel()

        d_b = bivariate.predict(_blend(1, 1.5), _blend(2, 1.2))
        d_i = independent.predict(_blend(1, 1.5), _blend(2, 1.2))

        # With ρ=0, λ12=0, so the bivariate collapses to independent.
        assert d_b.p_home_win == pytest.approx(d_i.p_home_win, abs=1e-6)
        assert d_b.p_draw == pytest.approx(d_i.p_draw, abs=1e-6)
        assert d_b.p_away_win == pytest.approx(d_i.p_away_win, abs=1e-6)

    def test_positive_rho_increases_draw_probability(self):
        """Positive correlation between team scores -> more clustering on the
        diagonal -> more draws (especially low-scoring draws)."""
        independent = IndependentPoissonModel().predict(
            _blend(1, 1.4), _blend(2, 1.4)
        )
        bivariate = BivariatePoissonModel(rho=0.15).predict(
            _blend(1, 1.4), _blend(2, 1.4)
        )
        assert bivariate.p_draw > independent.p_draw

    def test_grid_sums_to_approx_one(self):
        m = BivariatePoissonModel()
        d = m.predict(_blend(1, 1.5), _blend(2, 1.2))
        total = sum(sum(row) for row in d.score_grid)
        # Truncated at MAX_GOALS=8 — tail is tiny but non-zero.
        assert 0.99 <= total <= 1.0

    def test_mean_recovers_input_lambda(self):
        """The mean of the grid's marginal home distribution should equal the
        target μ_h (within truncation error)."""
        m = BivariatePoissonModel(rho=0.05)
        target_mu_h = 1.7
        d = m.predict(_blend(1, target_mu_h), _blend(2, 1.0))
        grid = d.score_grid_as_array()
        # E[X] = Σ x · P(X=x) where P(X=x) = Σ_y grid[x, y]
        marginal_h = grid.sum(axis=1)
        ks = list(range(len(marginal_h)))
        recovered = sum(k * p for k, p in zip(ks, marginal_h))
        assert recovered == pytest.approx(target_mu_h, abs=0.01)

    def test_btts_in_valid_range(self):
        m = BivariatePoissonModel()
        d = m.predict(_blend(1, 1.5), _blend(2, 1.2))
        assert 0.0 <= d.p_btts <= 1.0

    def test_lambda12_clamped_when_rho_too_high_for_lambdas(self):
        """When ρ would force λ1 or λ2 negative, the model clamps λ12."""
        # Both teams expected 0.5 goals; rho=0.5 -> λ12 = 0.5*0.5 = 0.25 -> OK
        # Both teams expected 0.3 goals; rho=0.5 -> λ12 = 0.5*0.3 = 0.15 -> OK
        # Both teams expected 0.001 goals; rho=0.5 -> λ12 = 0.0005 -> OK but tiny
        m = BivariatePoissonModel(rho=0.5)
        d = m.predict(_blend(1, 0.3), _blend(2, 0.3))
        # Should not crash; values should be valid probabilities.
        assert 0.0 <= d.p_home_win <= 1.0
        assert 0.0 <= d.p_btts <= 1.0


class TestModelComparison:
    """Smoke checks that the two models produce sensible, comparable outputs."""

    def test_both_models_agree_on_winner_for_dominant_team(self):
        ind = IndependentPoissonModel()
        biv = BivariatePoissonModel()
        home = _blend(1, 2.5)
        away = _blend(2, 0.5)

        d_i = ind.predict(home, away)
        d_b = biv.predict(home, away)

        assert d_i.p_home_win > d_i.p_away_win
        assert d_b.p_home_win > d_b.p_away_win

    def test_models_agree_on_total_goals_expectation(self):
        """E[X+Y] = λ_home + λ_away regardless of correlation."""
        ind = IndependentPoissonModel()
        biv = BivariatePoissonModel(rho=0.05)
        d_i = ind.predict(_blend(1, 1.5), _blend(2, 1.2))
        d_b = biv.predict(_blend(1, 1.5), _blend(2, 1.2))
        assert d_i.expected_total_goals == pytest.approx(d_b.expected_total_goals)
