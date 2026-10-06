"""Layer-1 tests for LiveMatchPredictor — Dixon-Robinson scoring."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.live.predictor import (
    MARKET_BTTS,
    MARKET_DOUBLE_CHANCE,
    MARKET_FULLTIME_RESULT,
    MARKET_OU_25,
    LiveMatchPredictor,
    _bivariate_poisson_grid,
    _normalise,
    _poisson_cdf,
)
from bip.sports.football.sportmonks.types import PredictionType


def _make_state(
    *,
    minute: int = 0,
    home_goals: int = 0,
    away_goals: int = 0,
    is_live: bool = True,
    home_pressure: list[float] | None = None,
    away_pressure: list[float] | None = None,
    sportmonks: dict[int, dict] | None = None,
) -> LiveMatchState:
    """Build a minimal LiveMatchState for tests, bypassing the API."""
    return LiveMatchState(
        fixture_id=1,
        home_team_id=10,
        away_team_id=20,
        home_team_name="Home",
        away_team_name="Away",
        home_goals=home_goals,
        away_goals=away_goals,
        minute=minute,
        period_id=2 if minute >= 45 else 1,
        is_live=is_live,
        is_half_time=False,
        is_finished=False,
        home_stats={},
        away_stats={},
        home_pressure_recent=home_pressure or [],
        away_pressure_recent=away_pressure or [],
        sportmonks_predictions=sportmonks or {},
        goal_events=[],
        red_card_events=[],
    )


# Default Sportmonks predictions for a balanced match (1.4 goals/team avg)
_BALANCED_PRE_MATCH = {
    PredictionType.FULLTIME_RESULT_PROBABILITY: {
        "home": 40, "draw": 30, "away": 30,
    },
    PredictionType.OVER_UNDER_2_5_PROBABILITY: {"yes": 55, "no": 45},
    PredictionType.OVER_UNDER_1_5_PROBABILITY: {"yes": 78, "no": 22},
    PredictionType.OVER_UNDER_3_5_PROBABILITY: {"yes": 25, "no": 75},
    PredictionType.BTTS_PROBABILITY: {"yes": 60, "no": 40},
    PredictionType.HOME_OVER_UNDER_0_5_PROBABILITY: {"yes": 75, "no": 25},
    PredictionType.AWAY_OVER_UNDER_0_5_PROBABILITY: {"yes": 70, "no": 30},
    PredictionType.HOME_OVER_UNDER_1_5_PROBABILITY: {"yes": 40, "no": 60},
    PredictionType.AWAY_OVER_UNDER_1_5_PROBABILITY: {"yes": 35, "no": 65},
}


# ── Bivariate Poisson grid ───────────────────────────────────────────────────


class TestBivariateGrid:
    def test_grid_normalised(self):
        g = _bivariate_poisson_grid(1.5, 1.2)
        assert abs(g.sum() - 1.0) < 0.01  # truncated grid → ~1

    def test_zero_lambdas_concentrate_at_origin(self):
        g = _bivariate_poisson_grid(0.0, 0.0)
        assert g[0, 0] == pytest.approx(1.0, abs=1e-9)
        assert g[1:, :].sum() == pytest.approx(0.0, abs=1e-9)

    def test_high_lambda_spread(self):
        g = _bivariate_poisson_grid(3.0, 3.0)
        # Mass shouldn't all be in the corner
        assert g[2, 2] > g[0, 0]


# ── Predictor scoring ────────────────────────────────────────────────────────


class TestFulltimeResultPreMatch:
    def test_pre_match_returns_sportmonks(self):
        state = _make_state(minute=0, is_live=False, sportmonks=_BALANCED_PRE_MATCH)
        probs = LiveMatchPredictor().predict(state)
        ft = probs.by_market[MARKET_FULLTIME_RESULT]
        assert ft["home"] == pytest.approx(0.40, abs=1e-3)
        assert ft["draw"] == pytest.approx(0.30, abs=1e-3)
        assert ft["away"] == pytest.approx(0.30, abs=1e-3)
        assert probs.sources[MARKET_FULLTIME_RESULT] == "sportmonks"


class TestFulltimeResultLive:
    def test_late_minute_with_lead_favors_winner(self):
        # Home leads 1-0 at minute 85 → home win heavily favoured
        state = _make_state(
            minute=85, home_goals=1, away_goals=0,
            sportmonks=_BALANCED_PRE_MATCH,
        )
        probs = LiveMatchPredictor().predict(state)
        ft = probs.by_market[MARKET_FULLTIME_RESULT]
        assert ft["home"] > 0.70
        assert ft["away"] < 0.10

    def test_late_minute_trailing_almost_certain_loss(self):
        # Home loses 1-2 at minute 95 → away win ~certain
        state = _make_state(
            minute=95, home_goals=1, away_goals=2,
            sportmonks=_BALANCED_PRE_MATCH,
        )
        probs = LiveMatchPredictor().predict(state)
        ft = probs.by_market[MARKET_FULLTIME_RESULT]
        assert ft["away"] > 0.90
        assert ft["home"] < 0.05

    def test_early_minute_balanced_stays_balanced(self):
        # Score 0-0 minute 5 → close to pre-match Sportmonks
        state = _make_state(minute=5, sportmonks=_BALANCED_PRE_MATCH)
        probs = LiveMatchPredictor().predict(state)
        ft = probs.by_market[MARKET_FULLTIME_RESULT]
        # Within ~5pp of pre-match
        assert abs(ft["home"] - 0.40) < 0.07
        assert abs(ft["away"] - 0.30) < 0.07

    def test_pressure_diff_shifts_probabilities(self):
        baseline = _make_state(minute=30, sportmonks=_BALANCED_PRE_MATCH)
        pressured = _make_state(
            minute=30, sportmonks=_BALANCED_PRE_MATCH,
            home_pressure=[80] * 5, away_pressure=[20] * 5,
        )
        ft_baseline = LiveMatchPredictor().predict(baseline).by_market[MARKET_FULLTIME_RESULT]
        ft_pressured = LiveMatchPredictor().predict(pressured).by_market[MARKET_FULLTIME_RESULT]
        # Home pressure should boost home probability
        assert ft_pressured["home"] > ft_baseline["home"]

    def test_probabilities_sum_to_one(self):
        state = _make_state(minute=60, home_goals=1, away_goals=1,
                             sportmonks=_BALANCED_PRE_MATCH)
        ft = LiveMatchPredictor().predict(state).by_market[MARKET_FULLTIME_RESULT]
        total = ft["home"] + ft["draw"] + ft["away"]
        assert total == pytest.approx(1.0, abs=1e-6)


class TestBTTSConditioning:
    def test_btts_yes_certain_when_both_scored(self):
        state = _make_state(home_goals=1, away_goals=1, minute=30,
                             sportmonks=_BALANCED_PRE_MATCH)
        btts = LiveMatchPredictor().predict(state).by_market[MARKET_BTTS]
        assert btts["yes"] == 1.0
        assert btts["no"] == 0.0

    def test_btts_conditioned_on_one_team_scored(self):
        state = _make_state(home_goals=1, away_goals=0, minute=85,
                             sportmonks=_BALANCED_PRE_MATCH)
        btts = LiveMatchPredictor().predict(state).by_market[MARKET_BTTS]
        # Away has very little time to score → BTTS-yes much lower
        assert btts["yes"] < 0.30


class TestOUDixonRobinson:
    def test_ou_25_goes_to_one_when_already_three(self):
        state = _make_state(home_goals=2, away_goals=1, minute=70,
                             sportmonks=_BALANCED_PRE_MATCH)
        ou = LiveMatchPredictor().predict(state).by_market[MARKET_OU_25]
        assert ou["over"] == 1.0
        assert ou["under"] == 0.0

    def test_ou_25_goes_to_zero_late_with_low_score(self):
        state = _make_state(home_goals=0, away_goals=0, minute=89,
                             sportmonks=_BALANCED_PRE_MATCH)
        ou = LiveMatchPredictor().predict(state).by_market[MARKET_OU_25]
        # Need 3 goals in 1 minute → essentially impossible
        assert ou["over"] < 0.01

    def test_ou_25_pre_match_uses_sportmonks(self):
        state = _make_state(minute=0, is_live=False,
                             sportmonks=_BALANCED_PRE_MATCH)
        ou = LiveMatchPredictor().predict(state).by_market[MARKET_OU_25]
        assert ou["over"] == pytest.approx(0.55, abs=1e-3)


class TestDoubleChance:
    def test_dc_derived_correctly(self):
        state = _make_state(minute=10, sportmonks=_BALANCED_PRE_MATCH)
        probs = LiveMatchPredictor().predict(state)
        ft = probs.by_market[MARKET_FULLTIME_RESULT]
        dc = probs.by_market[MARKET_DOUBLE_CHANCE]
        assert dc["1x"] == pytest.approx(ft["home"] + ft["draw"], abs=0.01)
        assert dc["x2"] == pytest.approx(ft["draw"] + ft["away"], abs=0.01)
        assert dc["12"] == pytest.approx(ft["home"] + ft["away"], abs=0.01)


class TestPoissonCDF:
    def test_lambda_zero(self):
        assert _poisson_cdf(0, 0.0) == 1.0

    def test_lambda_one_k_zero(self):
        # P(X=0 | λ=1) = e^-1 ≈ 0.368
        assert _poisson_cdf(0, 1.0) == pytest.approx(0.3679, abs=1e-3)

    def test_cdf_monotonic(self):
        a = _poisson_cdf(2, 2.5)
        b = _poisson_cdf(5, 2.5)
        assert b > a
        assert b <= 1.0
