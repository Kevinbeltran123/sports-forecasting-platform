"""Tests del ajuste por rival + shrinkage (item ① roadmap next-gen)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.team_style_profiler.opponent_adjust import (
    james_stein_shrink,
    opponent_adjust_rate,
)


class TestOpponentAdjustRate:
    def test_weak_schedule_deflates_for_market(self):
        # rivales más flojos que ref (team_mean<ref), sensitivity<0 → tasa baja (corrige inflado)
        adj = opponent_adjust_rate(6.5, 1500.0, 1700.0, -0.004)
        assert adj < 6.5

    def test_hard_schedule_inflates_for_market(self):
        # rivales más duros que ref (team_mean>ref), sensitivity<0 → tasa sube
        adj = opponent_adjust_rate(5.0, 1900.0, 1700.0, -0.004)
        assert adj > 5.0

    def test_against_market_sign(self):
        # mercado 'en contra' tiene sensitivity>0: rivales flojos → menos córners en contra de los que parece
        adj = opponent_adjust_rate(3.0, 1500.0, 1700.0, 0.004)
        assert adj > 3.0

    def test_zero_sensitivity_is_identity(self):
        assert opponent_adjust_rate(6.5, 1500.0, 1700.0, 0.0) == 6.5

    def test_team_at_reference_unchanged(self):
        assert opponent_adjust_rate(6.5, 1700.0, 1700.0, -0.004) == 6.5


class TestJamesSteinShrink:
    def test_more_n_less_shrink(self):
        _, w_small = james_stein_shrink(6.5, 10, 5.0, 4.0, 1.0)
        _, w_big = james_stein_shrink(6.5, 100, 5.0, 4.0, 1.0)
        assert w_big < w_small

    def test_shrunk_between_value_and_cohort(self):
        v, w = james_stein_shrink(6.5, 10, 5.0, 4.0, 1.0)
        assert 5.0 <= v <= 6.5
        assert 0.0 <= w <= 1.0

    @pytest.mark.parametrize("n,between_var", [(0, 1.0), (-1, 1.0), (10, 0.0), (10, -1.0)])
    def test_degenerate_returns_all_cohort(self, n, between_var):
        v, w = james_stein_shrink(6.5, n, 5.0, 4.0, between_var)
        assert v == 5.0 and w == 1.0

    def test_zero_within_var_is_all_data(self):
        # sin ruido por-partido → confía 100% en el dato (peso 0)
        v, w = james_stein_shrink(6.5, 10, 5.0, 0.0, 1.0)
        assert w == 0.0 and v == 6.5

    def test_large_within_var_shrinks_hard(self):
        # mucho ruido por-partido → tira fuerte al cohorte
        _, w = james_stein_shrink(6.5, 10, 5.0, 1000.0, 1.0)
        assert w > 0.9
