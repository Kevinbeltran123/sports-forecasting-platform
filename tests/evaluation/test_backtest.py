"""Phase 4 — backtest infrastructure: metrics, ELO+logistic model, walk_forward harness."""

from __future__ import annotations

import math

import polars as pl
import pytest

from bip.evaluation.tournaments.backtest.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    mae,
    multiclass_brier_score,
    multiclass_log_loss,
    poisson_deviance,
    reliability_curve,
)
from bip.evaluation.tournaments.backtest.walk_forward import (
    HistoricalFixture,
    compare_reports,
    run_backtest,
)
from bip.evaluation.tournaments.models import Player, Position
from bip.evaluation.tournaments.players.lineup_predictor import (
    LineupConfidence,
    LineupPrediction,
)
from bip.evaluation.tournaments.predict.match_runner import (
    MatchPredictionRunner,
    TeamMatchInputs,
)
from bip.evaluation.tournaments.predictors.bivariate_poisson import (
    BivariatePoissonModel,
)
from bip.evaluation.tournaments.predictors.elo_logistic import (
    DEFAULT_TOTAL_GOALS_BASELINE,
    ELO_PER_GOAL,
    EloLogisticModel,
    elo_to_goal_supremacy,
    elo_win_probability,
)
from bip.evaluation.tournaments.predictors.independent_poisson import (
    IndependentPoissonModel,
)


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────


class TestLogLoss:
    def test_perfect_prediction_low_loss(self):
        # P(1)=0.999 when y=1 -> very low loss.
        loss = log_loss([0.999, 0.001], [1, 0])
        assert loss < 0.01

    def test_worst_prediction_high_loss(self):
        # P(1)=0.001 when y=1 -> huge loss.
        loss = log_loss([0.001], [1])
        assert loss > 6.5  # ~6.9

    def test_uniform_50_50(self):
        # P=0.5 always -> log(2) ≈ 0.693
        loss = log_loss([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert loss == pytest.approx(math.log(2), abs=1e-6)

    def test_clipped_at_eps(self):
        # P=0 with y=1 should clip and return finite (large) loss.
        loss = log_loss([0.0], [1])
        assert math.isfinite(loss)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            log_loss([0.5, 0.5], [1])

    def test_empty_returns_zero(self):
        assert log_loss([], []) == 0.0


class TestMulticlassLogLoss:
    def test_perfect_prediction(self):
        # 1X2 with [1, 0, 0] when home wins -> log(1) = 0
        loss = multiclass_log_loss([(0.99, 0.005, 0.005)], [0])
        assert loss < 0.02

    def test_uniform_three_class(self):
        # P=1/3 for each class -> log(3) ≈ 1.0986
        loss = multiclass_log_loss(
            [(1 / 3, 1 / 3, 1 / 3)] * 3, [0, 1, 2]
        )
        assert loss == pytest.approx(math.log(3), abs=1e-6)

    def test_class_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            multiclass_log_loss([(0.4, 0.3, 0.3)], [3])


class TestBrier:
    def test_perfect_prediction(self):
        assert brier_score([1.0, 0.0], [1, 0]) == 0.0

    def test_worst_prediction(self):
        assert brier_score([0.0, 1.0], [1, 0]) == 1.0

    def test_uniform_random(self):
        # P=0.5 always, half outcomes 1 -> Brier = 0.25
        score = brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert score == pytest.approx(0.25)

    def test_multiclass_perfect(self):
        # 3-class one-hot match -> Brier = 0
        assert multiclass_brier_score([(1.0, 0.0, 0.0)], [0]) == 0.0


class TestMaeAndPoissonDeviance:
    def test_mae_basic(self):
        assert mae([1.0, 2.0, 3.0], [1.5, 2.5, 2.5]) == pytest.approx(0.5)

    def test_poisson_deviance_perfect(self):
        # k = λ everywhere -> deviance = 0.
        assert poisson_deviance([2.0, 3.0], [2, 3]) == pytest.approx(0.0, abs=1e-6)

    def test_poisson_deviance_zero_count(self):
        """k=0 case: -2(0 - λ) = 2λ."""
        d = poisson_deviance([1.5], [0])
        # Per-element: 2 · (0 · log(0/1.5) − (0 − 1.5)) = 2 · 1.5 = 3.0
        assert d == pytest.approx(3.0)

    def test_poisson_deviance_increases_with_error(self):
        small_err = poisson_deviance([2.0], [3])
        large_err = poisson_deviance([2.0], [5])
        assert large_err > small_err


class TestECE:
    def test_perfect_calibration_zero_ece(self):
        # All P=0.5 with exactly 50% outcomes -> bucket-mean=bucket-freq -> ECE=0.
        ece = expected_calibration_error(
            [0.5] * 100,
            [1] * 50 + [0] * 50,
            n_buckets=10,
        )
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_systematic_overconfidence_high_ece(self):
        # P=0.9 always but 50% outcomes -> ECE = |0.9 - 0.5| = 0.4
        ece = expected_calibration_error(
            [0.9] * 100,
            [1] * 50 + [0] * 50,
            n_buckets=10,
        )
        assert ece == pytest.approx(0.4, abs=1e-6)

    def test_empty_returns_zero(self):
        assert expected_calibration_error([], [], n_buckets=10) == 0.0


class TestReliabilityCurve:
    def test_returns_per_bucket_data(self):
        curve = reliability_curve(
            [0.1, 0.2, 0.5, 0.8, 0.9],
            [0, 0, 1, 1, 1],
            n_buckets=5,
        )
        assert sum(curve.bucket_count) == 5

    def test_perfect_calibration_diagonal(self):
        # Build a synthetic perfectly-calibrated dataset.
        probs = [0.1] * 100 + [0.5] * 100 + [0.9] * 100
        outcomes = [1] * 10 + [0] * 90 + [1] * 50 + [0] * 50 + [1] * 90 + [0] * 10
        curve = reliability_curve(probs, outcomes, n_buckets=10)
        # Each non-empty bucket should have bucket_freq close to bucket_mean.
        for pred, freq in zip(curve.bucket_mean_pred, curve.bucket_freq_actual):
            assert abs(pred - freq) < 0.05


# ─────────────────────────────────────────────────────────────────
# ELO + logistic model
# ─────────────────────────────────────────────────────────────────


def _blend(team_id: int, goals: float = 1.5):
    from bip.evaluation.tournaments.predict.blender import BlendedRates

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


class TestEloLogistic:
    def test_equal_elo_yields_equal_lambdas(self):
        model = EloLogisticModel(elo_ratings={1: 1900.0, 2: 1900.0})
        d = model.predict(_blend(1), _blend(2))
        assert d.lambda_home == pytest.approx(d.lambda_away)
        # Total ~= total_goals_baseline
        assert d.lambda_home + d.lambda_away == pytest.approx(
            DEFAULT_TOTAL_GOALS_BASELINE, abs=0.01
        )

    def test_higher_elo_team_higher_lambda(self):
        # Elo diff 100 -> supremacy 1.0 -> stays well above the clamp floor.
        model = EloLogisticModel(elo_ratings={1: 2000.0, 2: 1900.0})
        d = model.predict(_blend(1), _blend(2))
        assert d.lambda_home > d.lambda_away
        # supremacy = 100/100 = 1, total preserved.
        assert d.lambda_home - d.lambda_away == pytest.approx(1.0, abs=0.01)
        assert d.lambda_home + d.lambda_away == pytest.approx(
            DEFAULT_TOTAL_GOALS_BASELINE, abs=0.01
        )

    def test_missing_elo_raises(self):
        model = EloLogisticModel(elo_ratings={1: 1900.0})
        with pytest.raises(KeyError, match="Missing Elo"):
            model.predict(_blend(1), _blend(2))

    def test_lambda_floor_prevents_negative(self):
        # Extreme Elo gap that would push λ_a negative.
        model = EloLogisticModel(elo_ratings={1: 3000.0, 2: 1000.0})
        d = model.predict(_blend(1), _blend(2))
        assert d.lambda_away >= 0.05

    def test_invalid_constructor_args(self):
        with pytest.raises(ValueError):
            EloLogisticModel(elo_ratings={1: 1900.0}, total_goals_baseline=0)
        with pytest.raises(ValueError):
            EloLogisticModel(elo_ratings={1: 1900.0}, elo_per_goal=-1)


class TestEloHelpers:
    def test_elo_win_probability_symmetric(self):
        assert elo_win_probability(1900.0, 1900.0) == pytest.approx(0.5)

    def test_elo_win_probability_dominant(self):
        # +400 Elo -> ~91% win probability.
        p = elo_win_probability(2300.0, 1900.0)
        assert p == pytest.approx(0.909, abs=0.005)

    def test_elo_to_goal_supremacy(self):
        s = elo_to_goal_supremacy(1900.0, 1700.0)
        assert s == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────
# Walk-forward harness
# ─────────────────────────────────────────────────────────────────


def _build_team_inputs(team_id: int, name: str, fifa: str, gf: float = 1.7) -> TeamMatchInputs:
    pid_base = team_id * 1000
    players = [
        Player.from_api_football(pid_base + i, f"{fifa}{i}", Position.MIDFIELDER, team_id)
        for i in range(11)
    ]
    lineup = LineupPrediction(
        players=tuple(players),
        confidence=LineupConfidence.HEURISTIC_QUALIFIER,
    )
    canonical_ids = [f"af-{p.api_football_id}" for p in players]
    form = pl.DataFrame(
        {
            "canonical_id": canonical_ids,
            "position": ["M"] * 11,
            "minutes_total": [1500] * 11,
            "reliable": [True] * 11,
            "normalized_shots_per90": [1.5] * 11,
            "normalized_sot_per90": [0.6] * 11,
            "normalized_goals_per90": [gf / 11] * 11,
            "fouls_committed_per90": [1.0] * 11,
            "shots_per90": [1.5] * 11,
            "sot_per90": [0.6] * 11,
            "goals_per90": [gf / 11] * 11,
        }
    )
    baseline = pl.DataFrame(
        {
            "team_id": [team_id],
            "adjusted_gf_per90": [gf],
            "adjusted_shots_for_per90": [13.0],
            "adjusted_sot_for_per90": [4.5],
            "adjusted_fouls_for_per90": [11.0],
            "adjusted_corners_for_per90": [5.5],
        }
    )
    return TeamMatchInputs(
        team_id=team_id,
        team_name=name,
        fifa_code=fifa,
        baseline_row=baseline,
        lineup=lineup,
        lineup_form=form,
    )


def _historical_fixture(
    fid: int, home_id: int, away_id: int, hg: int, ag: int
) -> HistoricalFixture:
    return HistoricalFixture(
        match_id=f"hist_{fid}",
        tournament_slug="test_tournament",
        home_inputs=_build_team_inputs(home_id, f"H{home_id}", f"H{home_id:02d}"),
        away_inputs=_build_team_inputs(away_id, f"A{away_id}", f"A{away_id:02d}"),
        actual_home_goals=hg,
        actual_away_goals=ag,
    )


class TestHistoricalFixtureProperties:
    def test_outcome_1x2_home_win(self):
        fx = _historical_fixture(1, 6, 26, 2, 1)
        assert fx.actual_outcome_1x2 == 0

    def test_outcome_1x2_draw(self):
        fx = _historical_fixture(1, 6, 26, 1, 1)
        assert fx.actual_outcome_1x2 == 1

    def test_outcome_1x2_away_win(self):
        fx = _historical_fixture(1, 6, 26, 0, 2)
        assert fx.actual_outcome_1x2 == 2

    def test_btts_yes(self):
        fx = _historical_fixture(1, 6, 26, 1, 1)
        assert fx.actual_btts == 1

    def test_btts_no_when_home_zero(self):
        fx = _historical_fixture(1, 6, 26, 0, 2)
        assert fx.actual_btts == 0

    def test_over_2_5(self):
        assert _historical_fixture(1, 6, 26, 2, 1).actual_over_2_5 == 1
        assert _historical_fixture(1, 6, 26, 1, 1).actual_over_2_5 == 0


class TestRunBacktest:
    def test_empty_fixtures_raises(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        with pytest.raises(ValueError, match="at least 1"):
            run_backtest([], runner)

    def test_basic_backtest_runs(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        fixtures = [
            _historical_fixture(1, 6, 26, 2, 1),
            _historical_fixture(2, 6, 26, 0, 0),
            _historical_fixture(3, 6, 26, 1, 2),
        ]
        report = run_backtest(fixtures, runner)
        assert report.n_matches == 3
        assert report.model_name == "independent_poisson"
        # Metric values are floats and finite.
        assert math.isfinite(report.multiclass_log_loss)
        assert math.isfinite(report.btts_brier)
        assert math.isfinite(report.over25_ece)

    def test_keep_per_match_records_each_fixture(self):
        runner = MatchPredictionRunner(BivariatePoissonModel(rho=0.04))
        fixtures = [_historical_fixture(i, 6, 26, 1, 1) for i in range(5)]
        report = run_backtest(fixtures, runner, keep_per_match=True)
        assert len(report.per_match) == 5
        for entry in report.per_match:
            for k in (
                "match_id",
                "lambda_home",
                "lambda_away",
                "actual_home_goals",
                "p_btts",
            ):
                assert k in entry

    def test_per_match_off_by_default(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        fixtures = [_historical_fixture(1, 6, 26, 2, 1)]
        report = run_backtest(fixtures, runner)
        assert report.per_match == ()


class TestCompareReports:
    def test_compare_two_models(self):
        runner_ind = MatchPredictionRunner(IndependentPoissonModel())
        runner_biv = MatchPredictionRunner(BivariatePoissonModel(rho=0.04))
        fixtures = [
            _historical_fixture(1, 6, 26, 2, 1),
            _historical_fixture(2, 6, 26, 1, 1),
            _historical_fixture(3, 6, 26, 0, 2),
        ]
        report_ind = run_backtest(fixtures, runner_ind)
        report_biv = run_backtest(fixtures, runner_biv)

        md = compare_reports([report_ind, report_biv])
        assert "independent_poisson" in md
        assert "bivariate_poisson" in md
        assert "1X2 log-loss" in md
        assert "BTTS Brier" in md
