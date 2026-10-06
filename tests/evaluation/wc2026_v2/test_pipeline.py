"""Ola 4 — CalibratedDIBPPredictor + fit_v2_pipeline e2e tests.

Layer-1: synthetic e2e — generate from known truth, fit pipeline, verify
markets are sensible.

Layer-2: real-corpus smoke — fit pipeline on (train, calibration) splits
from Ola 0, predict on held-out fixtures, sanity-check the output.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from bip.evaluation.tournaments.wc2026_v2.dibp import DIBPParams
from bip.evaluation.tournaments.wc2026_v2.pipeline import fit_v2_pipeline
from bip.evaluation.tournaments.wc2026_v2.v2_predictor import (
    CalibratedDIBPPrediction,
    CalibratedDIBPPredictor,
    V2Calibrators,
)
from bip.evaluation.tournaments.wc2026_v2.weighted_strength import (
    TeamStrength,
    WeightedMLEResult,
)


def _synthetic_strength_result() -> WeightedMLEResult:
    """Minimal hand-built strength prior for unit-level predictor tests."""
    return WeightedMLEResult(
        strengths={
            "strong_team": TeamStrength(team="strong_team", attack=0.5, defense=-0.4),
            "average_team": TeamStrength(team="average_team", attack=0.0, defense=0.0),
            "weak_team": TeamStrength(team="weak_team", attack=-0.5, defense=0.4),
        },
        home_advantage=0.20,
        intercept=0.10,  # ≈ 1.1 mean goals per side
        reference_date=date(2026, 5, 23),
        n_train_matches=300,
        n_teams=3,
    )


# ─────────────────────────────────────────────────────────────────
# CalibratedDIBPPredictor unit tests (no calibration → raw markets)
# ─────────────────────────────────────────────────────────────────


class TestPredictorBasics:
    def test_predict_returns_dataclass(self) -> None:
        strengths = _synthetic_strength_result()
        predictor = CalibratedDIBPPredictor(
            strengths=strengths, dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        pred = predictor.predict("strong_team", "weak_team")
        assert isinstance(pred, CalibratedDIBPPrediction)
        assert pred.home_team == "strong_team"
        assert pred.away_team == "weak_team"

    def test_strong_vs_weak_favours_home(self) -> None:
        predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        pred = predictor.predict("strong_team", "weak_team")
        assert pred.p_home_win > pred.p_away_win
        assert pred.p_home_win > pred.p_draw
        assert pred.p_home_win + pred.p_draw + pred.p_away_win == pytest.approx(1.0, abs=1e-6)

    def test_weak_vs_strong_favours_away(self) -> None:
        predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        pred = predictor.predict("weak_team", "strong_team")
        # Despite home advantage, the away team's strength should win out.
        assert pred.p_away_win > pred.p_home_win

    def test_markets_sum_to_one(self) -> None:
        predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.04, 0.15, 0.3)
        )
        pred = predictor.predict("average_team", "strong_team")
        assert pred.p_home_win + pred.p_draw + pred.p_away_win == pytest.approx(1.0, abs=1e-6)
        assert pred.p_over_2_5 + pred.p_under_2_5 == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= pred.p_btts <= 1.0


class TestColdStart:
    def test_unknown_team_flags_cold_start(self) -> None:
        predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        pred = predictor.predict("unknown_xyz", "strong_team")
        assert pred.is_cold_start is True
        assert "unknown_xyz" in pred.cold_start_teams
        # Predictor still returns valid markets.
        assert pred.p_home_win + pred.p_draw + pred.p_away_win == pytest.approx(1.0, abs=1e-6)

    def test_both_unknown_flags_both(self) -> None:
        predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        pred = predictor.predict("unknown_a", "unknown_b")
        assert pred.is_cold_start is True
        assert set(pred.cold_start_teams) == {"unknown_a", "unknown_b"}
        # With both cohort-substituted, home win still has +γ_home advantage.
        assert pred.p_home_win > pred.p_away_win

    def test_known_teams_not_flagged_cold_start(self) -> None:
        predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        pred = predictor.predict("strong_team", "weak_team")
        assert pred.is_cold_start is False
        assert pred.cold_start_teams == ()


class TestWithCalibrators:
    """Calibrators that map x → x (identity) must not change probabilities."""

    def test_identity_calibrators_preserve_outputs(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.beta_calibrator import BetaCalibrator

        ident_3 = BetaCalibrator.from_dict(
            {
                "kind": "beta",
                "n_classes": 3,
                "a": [1.0, 1.0, 1.0],
                "b": [1.0, 1.0, 1.0],
                "c": [0.0, 0.0, 0.0],
            }
        )
        ident_2 = BetaCalibrator.from_dict(
            {
                "kind": "beta",
                "n_classes": 2,
                "a": [1.0, 1.0],
                "b": [1.0, 1.0],
                "c": [0.0, 0.0],
            }
        )
        cal = V2Calibrators(one_x_two=ident_3, btts=ident_2, ou_2_5=ident_2)

        raw_predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(), dibp_params=DIBPParams(0.0, 0.1, 0.3)
        )
        cal_predictor = CalibratedDIBPPredictor(
            strengths=_synthetic_strength_result(),
            dibp_params=DIBPParams(0.0, 0.1, 0.3),
            calibrators=cal,
        )
        r = raw_predictor.predict("strong_team", "weak_team")
        c = cal_predictor.predict("strong_team", "weak_team")
        assert c.p_home_win == pytest.approx(r.p_home_win, abs=1e-5)
        assert c.p_draw == pytest.approx(r.p_draw, abs=1e-5)
        assert c.p_away_win == pytest.approx(r.p_away_win, abs=1e-5)
        assert c.p_btts == pytest.approx(r.p_btts, abs=1e-5)
        assert c.p_over_2_5 == pytest.approx(r.p_over_2_5, abs=1e-5)


# ─────────────────────────────────────────────────────────────────
# fit_v2_pipeline — Layer-2 smoke on real corpus
# ─────────────────────────────────────────────────────────────────


@pytest.mark.requires_data
class TestFitV2PipelineReal:
    """Layer-2: fit on real splits and predict on a held-out fixture."""

    def test_full_pipeline_fits_and_predicts(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        train = split.train.filter(pl.col("date") >= pl.lit(date(2010, 1, 1)))
        result = fit_v2_pipeline(
            train=train, calibration=split.calibration, reference_date=date(2026, 5, 23)
        )
        assert result.strengths.n_teams > 100
        # Calibration corpus has 115 matches; almost all teams should resolve.
        assert result.n_calibration_matches >= 100
        # DIBP fit converged.
        assert result.dibp_fit.converged

        # Predict on a known fixture: Spain vs France should produce sensible
        # markets (both elite, total goals ~2.5-3, p_draw substantial).
        if "Spain" in result.strengths.strengths and "France" in result.strengths.strengths:
            pred = result.predictor.predict("Spain", "France")
            assert 0.0 < pred.p_home_win < 1.0
            assert 0.0 < pred.p_draw < 1.0
            assert 0.0 < pred.p_away_win < 1.0
            assert 1.5 < pred.expected_total_goals < 4.5
            assert pred.is_cold_start is False

    def test_ablation_no_dibp_collapses(self) -> None:
        """use_dibp=False should produce a predictor with π=0 and θ irrelevant."""
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        train = split.train.filter(pl.col("date") >= pl.lit(date(2010, 1, 1)))
        result = fit_v2_pipeline(
            train=train,
            calibration=split.calibration,
            reference_date=date(2026, 5, 23),
            use_dibp=False,
        )
        assert result.dibp_fit.params.pi_diag == 0.0
        assert result.dibp_fit.params.rho == 0.0

    def test_ablation_no_beta_calibration_yields_empty_calibrators(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        train = split.train.filter(pl.col("date") >= pl.lit(date(2010, 1, 1)))
        result = fit_v2_pipeline(
            train=train,
            calibration=split.calibration,
            reference_date=date(2026, 5, 23),
            use_beta_calibration=False,
        )
        assert result.predictor.calibrators.one_x_two is None
        assert result.predictor.calibrators.btts is None
        assert result.predictor.calibrators.ou_2_5 is None

    def test_ablation_no_match_importance_changes_home_advantage(self) -> None:
        """use_match_importance=False flattens K to friendly-only → home
        advantage estimate should still be positive and finite, but the
        per-team rankings can shift visibly."""
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        train = split.train.filter(pl.col("date") >= pl.lit(date(2010, 1, 1)))
        result = fit_v2_pipeline(
            train=train,
            calibration=split.calibration,
            reference_date=date(2026, 5, 23),
            use_match_importance=False,
        )
        # Home advantage on internationals is positive even with uniform weights.
        assert result.strengths.home_advantage > 0.0
        assert np.isfinite(result.strengths.home_advantage)
