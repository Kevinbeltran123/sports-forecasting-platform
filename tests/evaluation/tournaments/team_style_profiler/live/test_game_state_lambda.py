"""Tests for game_state_lambda — adjusted λ for live betting.

Anchors validation around the empirical case that motivated this module:
Ireland 1-0 Qatar (2026-05-28). Naive Poisson predicted P(Ire ≥ 1 more) = 78%
over 85 remaining minutes; adjusted prediction with Hallgrímsson defensive +
leading +1 should be < 55% (closer to realized outcome of 0 second-half
Ireland goals).
"""
from __future__ import annotations

import math

import pytest

from bip.evaluation.tournaments.team_style_profiler.live.game_state_lambda import (
    adjust_lambda_for_game_state,
    p_at_least_one_more_goal,
)


class TestTimeScaling:
    def test_full_match_no_state_returns_base(self):
        # 90 min remaining, level, balanced → no adjustment
        result = adjust_lambda_for_game_state(1.80, 90, 0, "balanced")
        assert result == pytest.approx(1.80, rel=1e-3)

    def test_half_remaining_halves_lambda(self):
        result = adjust_lambda_for_game_state(1.80, 45, 0, "balanced")
        assert result == pytest.approx(0.90, rel=1e-3)

    def test_zero_minutes_returns_floor(self):
        assert adjust_lambda_for_game_state(1.80, 0, 0, "balanced") == pytest.approx(0.05)

    def test_negative_minutes_returns_floor(self):
        assert adjust_lambda_for_game_state(1.80, -5, 0, "balanced") == pytest.approx(0.05)


class TestCoachStyleLeading:
    def test_defensive_leading_strong_lockdown(self):
        # Hallgrímsson archetype: leading +1, full 90 → 1.80 × 0.40 = 0.72
        result = adjust_lambda_for_game_state(1.80, 90, 1, "defensive")
        assert result == pytest.approx(0.72, rel=1e-3)

    def test_balanced_leading_moderate_reduction(self):
        # 1.80 × 0.70 = 1.26
        result = adjust_lambda_for_game_state(1.80, 90, 1, "balanced")
        assert result == pytest.approx(1.26, rel=1e-3)

    def test_attacking_leading_minor_reduction(self):
        # Spain/Brazil archetype: 1.80 × 0.90 = 1.62
        result = adjust_lambda_for_game_state(1.80, 90, 1, "attacking")
        assert result == pytest.approx(1.62, rel=1e-3)

    def test_defensive_locks_more_than_attacking_when_leading(self):
        defensive = adjust_lambda_for_game_state(1.80, 60, 1, "defensive")
        attacking = adjust_lambda_for_game_state(1.80, 60, 1, "attacking")
        assert defensive < attacking


class TestCoachStyleTrailing:
    def test_attacking_trailing_significant_increase(self):
        # 1.5 × 1.20 = 1.80
        result = adjust_lambda_for_game_state(1.50, 90, -1, "attacking")
        assert result == pytest.approx(1.80, rel=1e-3)

    def test_defensive_trailing_minimal_increase(self):
        # Defensive coach rigid even when chasing — 1.5 × 0.95 = 1.425
        result = adjust_lambda_for_game_state(1.50, 90, -1, "defensive")
        assert result == pytest.approx(1.425, rel=1e-3)

    def test_attacking_trailing_outscores_defensive_trailing(self):
        attacking = adjust_lambda_for_game_state(1.50, 60, -1, "attacking")
        defensive = adjust_lambda_for_game_state(1.50, 60, -1, "defensive")
        assert attacking > defensive


class TestBigMargin:
    def test_leading_by_2_extra_reduction(self):
        # 1.80 × 0.70 (balanced leading) × 0.85 (big margin) = 1.071
        result = adjust_lambda_for_game_state(1.80, 90, 2, "balanced")
        assert result == pytest.approx(1.071, rel=1e-3)

    def test_trailing_by_3_panic_inefficiency(self):
        # 1.50 × 1.20 (attacking trailing) × 0.85 (big margin) = 1.530
        result = adjust_lambda_for_game_state(1.50, 90, -3, "attacking")
        assert result == pytest.approx(1.530, rel=1e-3)

    def test_no_big_margin_when_diff_one(self):
        # diff=+1 should NOT apply the 0.85 compound
        with_one = adjust_lambda_for_game_state(2.00, 90, 1, "balanced")
        # 2.00 × 0.70 = 1.40 (no 0.85 factor)
        assert with_one == pytest.approx(1.40, rel=1e-3)


class TestRedCard:
    def test_red_card_reduces_45_percent(self):
        # Level, 45 mins remaining, balanced, own red card
        # 1.80 × (45/90) × 1.00 × 0.55 = 0.495
        result = adjust_lambda_for_game_state(1.80, 45, 0, "balanced", has_red_card=True)
        assert result == pytest.approx(0.495, rel=1e-3)

    def test_red_card_compounds_with_lead(self):
        # Defensive leading + own red card = max lockdown
        # 1.80 × (45/90) × 0.40 × 0.55 = 0.198
        result = adjust_lambda_for_game_state(1.80, 45, 1, "defensive", has_red_card=True)
        assert result == pytest.approx(0.198, rel=1e-3)


class TestIrelandQatarAnchor:
    """The empirical case that motivated this module.

    Ireland 1-0 Qatar, 2026-05-28. λ_base for Ireland ≈ 1.80
    (TSV GF=1.26 × home boost 1.2 × Qatar weak opponent +0.2).

    Real outcome: 0 second-half Ireland goals (Hallgrímsson locked down,
    own red card to Moylan reinforced bunker).
    """

    def test_min_5_lead_p_more_goal_below_naive(self):
        # Adjusted P should be < 0.55 (naive Poisson said 0.78)
        p = p_at_least_one_more_goal(1.80, 85, 1, "defensive")
        assert p < 0.55, f"Expected lockdown but got P={p:.3f}"
        # And should be ≥ 0.30 (not zeroed out completely)
        assert p > 0.30, f"Expected some scoring chance but got P={p:.3f}"

    def test_post_red_card_near_bunker(self):
        # ~45 min remaining, still 1-0 lead, Moylan red card
        p = p_at_least_one_more_goal(1.80, 45, 1, "defensive", has_red_card=True)
        # Should be < 0.25 — very low chance of another Ireland goal
        assert p < 0.25, f"Expected bunker mode but got P={p:.3f}"

    def test_naive_vs_adjusted_diverges_meaningfully(self):
        # Naive: just time-scale base λ
        naive_lambda = 1.80 * (85 / 90)
        naive_p = 1.0 - math.exp(-naive_lambda)

        # Adjusted with defensive + leading +1
        adjusted_p = p_at_least_one_more_goal(1.80, 85, 1, "defensive")

        # Adjusted should be substantially lower (lesson of the day)
        assert (naive_p - adjusted_p) > 0.20, (
            f"Naive {naive_p:.3f} vs adjusted {adjusted_p:.3f} should diverge >20pp"
        )


class TestProbabilityHelper:
    def test_p_returns_in_unit_interval(self):
        p = p_at_least_one_more_goal(1.50, 30, 0, "balanced")
        assert 0.0 <= p <= 1.0

    def test_higher_lambda_higher_probability(self):
        p_low = p_at_least_one_more_goal(0.5, 60, 0, "balanced")
        p_high = p_at_least_one_more_goal(2.5, 60, 0, "balanced")
        assert p_high > p_low

    def test_zero_minutes_near_zero_prob(self):
        # Floor 0.05 → P ≈ 1 - e^-0.05 ≈ 0.049
        p = p_at_least_one_more_goal(1.80, 0, 0, "balanced")
        assert p < 0.06
