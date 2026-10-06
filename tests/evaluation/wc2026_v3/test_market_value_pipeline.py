"""Tests for the Wave 1.A pipeline extension: market_value toggle + offsets.

Verifies that:

1. CalibratedDIBPPredictor with ``market_offsets=None`` produces IDENTICAL
   probabilities to the un-extended v2 predictor (backward-compat).
2. A positive offset for the home team shifts probability mass toward home
   win; a negative offset shifts away from it (mechanism sanity check).
3. Missing-team coverage in the offsets dict falls back to 0.0 silently —
   no KeyError, no NaN — matching the documented graceful-degradation
   behavior in _resolve_lambdas.
4. fit_v2_pipeline rejects use_market_value=True without market_values_path.
5. fit_v2_pipeline with use_market_value=True wires offsets through to the
   predictor and exposes them on the V2PipelineResult for ablation logging.

The v2 pipeline backtest harness consumes V2PipelineResult.predictor; these
tests guard the contract that wave 1.A extends without breaking.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bip.evaluation.tournaments.wc2026_v2.dibp import DIBPParams
from bip.evaluation.tournaments.wc2026_v2.pipeline import (
    V2PipelineResult,
    fit_v2_pipeline,
)
from bip.evaluation.tournaments.wc2026_v2.v2_predictor import (
    CalibratedDIBPPredictor,
)
from bip.evaluation.tournaments.wc2026_v2.weighted_strength import (
    TeamStrength,
    WeightedMLEResult,
)


def _synthetic_strengths() -> WeightedMLEResult:
    """Three teams, deliberately balanced so any prediction shift is
    attributable to the market_offset injection."""
    return WeightedMLEResult(
        strengths={
            "alpha": TeamStrength(team="alpha", attack=0.0, defense=0.0),
            "beta": TeamStrength(team="beta", attack=0.0, defense=0.0),
            "gamma": TeamStrength(team="gamma", attack=0.0, defense=0.0),
        },
        home_advantage=0.20,
        intercept=0.10,
        reference_date=date(2026, 5, 23),
        n_train_matches=300,
        n_teams=3,
    )


# ─────────────────────────────────────────────────────────────────
# CalibratedDIBPPredictor.market_offsets — unit
# ─────────────────────────────────────────────────────────────────


class TestMarketOffsetsBackwardCompat:
    def test_none_offsets_match_v2_baseline(self) -> None:
        """A predictor with market_offsets=None must produce IDENTICAL
        markets to the un-extended v2 default. Locks in backward-compat."""
        strengths = _synthetic_strengths()
        baseline = CalibratedDIBPPredictor(
            strengths=strengths, dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        extended = CalibratedDIBPPredictor(
            strengths=strengths,
            dibp_params=DIBPParams(0.0, 0.0, 0.5),
            market_offsets=None,
        )
        for h, a in [("alpha", "beta"), ("beta", "alpha"), ("gamma", "alpha")]:
            b = baseline.predict(h, a)
            x = extended.predict(h, a)
            assert b.p_home_win == pytest.approx(x.p_home_win, abs=1e-12)
            assert b.p_draw == pytest.approx(x.p_draw, abs=1e-12)
            assert b.p_away_win == pytest.approx(x.p_away_win, abs=1e-12)
            assert b.p_btts == pytest.approx(x.p_btts, abs=1e-12)
            assert b.p_over_2_5 == pytest.approx(x.p_over_2_5, abs=1e-12)

    def test_empty_offsets_dict_equivalent_to_none(self) -> None:
        """Empty dict is functionally identical to None (every team gets 0.0)."""
        strengths = _synthetic_strengths()
        none_p = CalibratedDIBPPredictor(
            strengths=strengths, dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        empty_p = CalibratedDIBPPredictor(
            strengths=strengths,
            dibp_params=DIBPParams(0.0, 0.0, 0.5),
            market_offsets={},
        )
        n = none_p.predict("alpha", "beta")
        e = empty_p.predict("alpha", "beta")
        assert e.p_home_win == pytest.approx(n.p_home_win, abs=1e-12)
        assert e.p_draw == pytest.approx(n.p_draw, abs=1e-12)
        assert e.p_away_win == pytest.approx(n.p_away_win, abs=1e-12)


class TestMarketOffsetsMechanism:
    @pytest.mark.parametrize("offset_home", [0.05, 0.10, 0.20])
    def test_positive_home_offset_increases_home_win(self, offset_home: float) -> None:
        """A larger positive home offset must monotonically raise P(home)."""
        strengths = _synthetic_strengths()
        baseline = CalibratedDIBPPredictor(
            strengths=strengths, dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        shifted = CalibratedDIBPPredictor(
            strengths=strengths,
            dibp_params=DIBPParams(0.0, 0.0, 0.5),
            market_offsets={"alpha": offset_home},
        )
        b = baseline.predict("alpha", "beta")
        s = shifted.predict("alpha", "beta")
        assert s.p_home_win > b.p_home_win
        assert s.p_away_win < b.p_away_win
        # Markets still sum to 1
        assert s.p_home_win + s.p_draw + s.p_away_win == pytest.approx(1.0, abs=1e-9)

    def test_negative_home_offset_decreases_home_win(self) -> None:
        strengths = _synthetic_strengths()
        baseline = CalibratedDIBPPredictor(
            strengths=strengths, dibp_params=DIBPParams(0.0, 0.0, 0.5)
        )
        shifted = CalibratedDIBPPredictor(
            strengths=strengths,
            dibp_params=DIBPParams(0.0, 0.0, 0.5),
            market_offsets={"alpha": -0.20},
        )
        b = baseline.predict("alpha", "beta")
        s = shifted.predict("alpha", "beta")
        assert s.p_home_win < b.p_home_win
        assert s.p_away_win > b.p_away_win

    def test_missing_team_falls_back_to_zero_offset(self) -> None:
        """Partial offset coverage degrades gracefully — no KeyError, no NaN."""
        strengths = _synthetic_strengths()
        predictor = CalibratedDIBPPredictor(
            strengths=strengths,
            dibp_params=DIBPParams(0.0, 0.0, 0.5),
            market_offsets={"alpha": 0.15},  # beta missing — gets 0.0
        )
        pred = predictor.predict("alpha", "beta")
        assert np.isfinite(pred.p_home_win)
        assert pred.p_home_win + pred.p_draw + pred.p_away_win == pytest.approx(
            1.0, abs=1e-9
        )

    def test_log_rate_offset_matches_analytical_formula(self) -> None:
        """Verify the injected offset shifts log-lambda_h by exactly off_h.

        With (attack=0, defense=0, intercept=0.1, gamma=0.2) the baseline
        log_lambda_h = 0.3 → mu_h ≈ exp(0.3). Injecting off_h=0.5 must
        produce mu_h ≈ exp(0.3 + 0.5) = exp(0.8).
        """
        strengths = _synthetic_strengths()
        predictor = CalibratedDIBPPredictor(
            strengths=strengths,
            dibp_params=DIBPParams(0.0, 0.0, 0.5),
            market_offsets={"alpha": 0.5},
        )
        # _resolve_lambdas is a private API but covered here because the
        # log-rate math is what makes this whole feature falsifiable.
        mu_h, mu_a, _ = predictor._resolve_lambdas("alpha", "beta")
        assert mu_h == pytest.approx(float(np.exp(0.1 + 0.0 + 0.0 + 0.2 + 0.5)), rel=1e-9)
        # Away rate unchanged — offset applies only to attack of the
        # respective team, beta has no offset entry.
        assert mu_a == pytest.approx(float(np.exp(0.1 + 0.0 + 0.0)), rel=1e-9)


# ─────────────────────────────────────────────────────────────────
# fit_v2_pipeline use_market_value toggle
# ─────────────────────────────────────────────────────────────────


def _tiny_training_frame() -> pl.DataFrame:
    """6-team round-robin × 4 repeats so min_appearances=4 catches everyone.

    Goals are intentionally noisy but well-typed — the MLE just needs
    enough rows to converge for pipeline e2e."""
    rng = np.random.default_rng(seed=42)
    teams = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    rows = []
    for repeat in range(4):
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                rows.append(
                    {
                        "home_team": h,
                        "away_team": a,
                        "home_score": int(rng.poisson(1.4)),
                        "away_score": int(rng.poisson(1.0)),
                        "date": date(2024, 1 + repeat, 15),
                        "tournament": "Friendly",
                    }
                )
                rows.append(
                    {
                        "home_team": a,
                        "away_team": h,
                        "home_score": int(rng.poisson(1.4)),
                        "away_score": int(rng.poisson(1.0)),
                        "date": date(2024, 1 + repeat, 16),
                        "tournament": "Friendly",
                    }
                )
    return pl.DataFrame(rows)


def _tiny_calibration_frame() -> pl.DataFrame:
    """Just enough rows for the calibration loop to fit the beta calibrator."""
    rng = np.random.default_rng(seed=99)
    teams = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    rows = []
    for i, h in enumerate(teams):
        for a in teams[i + 1 :]:
            rows.append(
                {
                    "home_team": h,
                    "away_team": a,
                    "home_goals": int(rng.poisson(1.4)),
                    "away_goals": int(rng.poisson(1.0)),
                    "date": date(2024, 6, 1),
                    "tournament": "Friendly",
                }
            )
    return pl.DataFrame(rows)


def test_pipeline_rejects_use_market_value_without_path() -> None:
    train = _tiny_training_frame()
    cal = _tiny_calibration_frame()
    with pytest.raises(ValueError, match="market_values_path"):
        fit_v2_pipeline(
            train,
            cal,
            reference_date=date(2026, 5, 23),
            min_appearances=4,
            use_market_value=True,
            market_values_path=None,
            use_beta_calibration=False,  # skip beta — n is tiny
            use_dibp=False,
        )


def test_pipeline_wires_offsets_through_to_predictor(tmp_path: Path) -> None:
    """End-to-end: writing a TM file → pipeline → predictor → predict()
    must surface a non-zero attack offset on the rich team's home rate."""
    train = _tiny_training_frame()
    cal = _tiny_calibration_frame()

    # Three squads, deliberately ordered by value so median is 'beta'.
    values_path = tmp_path / "squad_values.parquet"
    pl.DataFrame(
        [
            {"team_name": "alpha", "market_value_eur": 100_000_000.0},
            {"team_name": "beta", "market_value_eur": 200_000_000.0},
            {"team_name": "gamma", "market_value_eur": 400_000_000.0},
            {"team_name": "delta", "market_value_eur": 200_000_000.0},
            {"team_name": "epsilon", "market_value_eur": 200_000_000.0},
            {"team_name": "zeta", "market_value_eur": 200_000_000.0},
        ]
    ).write_parquet(values_path)

    result = fit_v2_pipeline(
        train,
        cal,
        reference_date=date(2026, 5, 23),
        min_appearances=4,
        use_market_value=True,
        market_values_path=str(values_path),
        market_value_beta=0.10,
        use_beta_calibration=False,
        use_dibp=False,
    )

    assert isinstance(result, V2PipelineResult)
    assert result.market_offsets is not None
    # gamma > beta = median → positive offset; alpha < median → negative.
    assert result.market_offsets["gamma"] > 0
    assert result.market_offsets["alpha"] < 0
    assert result.market_offsets["beta"] == pytest.approx(0.0, abs=1e-9)

    # Predictor inherits the offsets and uses them at predict time.
    assert result.predictor.market_offsets is result.market_offsets
    pred_with = result.predictor.predict("gamma", "alpha")
    # Markets are valid (sum to 1, no NaN). Absolute direction can't be
    # asserted at this synthetic n — the MLE on ~120 Poisson-noisy rows
    # produces unstable strength estimates that may dominate the offset
    # contribution. The mechanism test (test_log_rate_offset_matches_
    # analytical_formula) covers the directional shift cleanly.
    assert pred_with.p_home_win + pred_with.p_draw + pred_with.p_away_win == pytest.approx(
        1.0, abs=1e-9
    )
    assert np.isfinite(pred_with.p_btts)

    # Mechanism at the pipeline boundary: re-fit with use_market_value=False
    # using the SAME synthetic data → compare. The only difference is the
    # offsets injection; gamma's home P(win) must be strictly higher with
    # the positive offset than without.
    baseline = fit_v2_pipeline(
        train,
        cal,
        reference_date=date(2026, 5, 23),
        min_appearances=4,
        use_market_value=False,
        use_beta_calibration=False,
        use_dibp=False,
    )
    pred_without = baseline.predictor.predict("gamma", "alpha")
    assert pred_with.p_home_win > pred_without.p_home_win


def test_pipeline_use_market_value_false_leaves_offsets_none() -> None:
    train = _tiny_training_frame()
    cal = _tiny_calibration_frame()
    result = fit_v2_pipeline(
        train,
        cal,
        reference_date=date(2026, 5, 23),
        min_appearances=4,
        use_market_value=False,
        use_beta_calibration=False,
        use_dibp=False,
    )
    assert result.market_offsets is None
    assert result.predictor.market_offsets is None
