"""End-to-end fit pipeline for the v2 predictor (Olas 1-4 composed).

``fit_v2_pipeline`` is the canonical "build the model" entry point used by
both Ola 5 (backtest) and Ola 7 (lock_v2 emission). It runs:

    1. WeightedMLEFitter.fit(train, reference_date) → strength prior
    2. fit_dibp(calibration, strengths) → DIBP params
    3. Apply (strengths, dibp_params) over the calibration corpus to get
       raw market probabilities + actual outcomes
    4. Fit one BetaCalibrator per market (1X2, BTTS, OU2.5)
    5. Return CalibratedDIBPPredictor

Each stage's intermediate result is exposed on ``V2PipelineResult`` so the
backtest + ablation can inspect what changed at each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from ._market_data import DEFAULT_BETA_MV, compute_offsets, load_squad_values
from .beta_calibrator import BetaCalibrator
from .dibp import DIBPParams, compute_dibp_grid, markets_from_grid
from .dibp_fitter import DIBPFitResult, fit_dibp
from .v2_predictor import CalibratedDIBPPredictor, V2Calibrators
from .weighted_strength import WeightedMLEFitter, WeightedMLEResult


@dataclass(frozen=True)
class V2PipelineResult:
    """Composed predictor + per-stage intermediate state."""

    predictor: CalibratedDIBPPredictor
    strengths: WeightedMLEResult
    dibp_fit: DIBPFitResult
    n_calibration_matches: int
    calibration_skipped_cold_start: int
    # Wave 1.A: populated when use_market_value=True so the ablation can log
    # per-team offsets + summary stats. ``None`` when the toggle is off, so
    # backward-compat with v2 callers that destructure V2PipelineResult is
    # preserved as long as they don't access this field positionally.
    market_offsets: dict[str, float] | None = None


def fit_v2_pipeline(
    train: pl.DataFrame,
    calibration: pl.DataFrame,
    *,
    reference_date: date,
    min_appearances: int = 10,
    half_life_days: float | None = None,
    init_dibp: DIBPParams = DIBPParams(rho=0.0, pi_diag=0.05, theta_diag=0.5),
    max_goals: int = 8,
    use_dibp: bool = True,
    use_beta_calibration: bool = True,
    use_match_importance: bool = True,
    use_market_value: bool = False,
    market_values_path: str | None = None,
    market_value_beta: float = DEFAULT_BETA_MV,
    market_offset_mode: str = "attack",
) -> V2PipelineResult:
    """Fit the full v2 pipeline.

    Ablation toggles:
    - ``use_dibp=False``  → forces DIBPParams(0, 0, 0); plain independent Poisson
    - ``use_beta_calibration=False`` → returns predictor with no calibrators
    - ``use_match_importance=False`` → weights collapse to time-decay only
      (set all tournament K-weights to 1.0 via half_life-only weighting). This
      is achieved here by routing through the standard fitter but with a
      pre-flattened tournament column.
    - ``use_market_value=True`` → load Transfermarkt squad values from
      ``market_values_path`` (parquet/CSV), compute log-rate offsets with
      coefficient ``market_value_beta``, inject into the predictor via
      ``CalibratedDIBPPredictor(market_offsets=...)``. Requires
      ``market_values_path`` to be set; raises ValueError otherwise.
      Default ``use_market_value=False`` preserves exact v2 behavior.
    """

    # Wave 1.A: validate market-value config BEFORE running any fit, so a
    # missing-path misconfiguration fails fast (not after a 30s MLE).
    if use_market_value and not market_values_path:
        raise ValueError(
            "use_market_value=True requires market_values_path to be set"
        )

    # Stage 1 — strength prior
    if use_match_importance:
        df_train = train
    else:
        # Ablation: pretend every match is "Friendly" so all K-weights collapse
        # to K_FRIENDLY = 20.0 (uniform). Time-decay still applies.
        df_train = train.with_columns(pl.lit("Friendly").alias("tournament"))

    fitter_kwargs: dict[str, object] = {"min_appearances": min_appearances}
    if half_life_days is not None:
        fitter_kwargs["half_life_days"] = half_life_days
    strengths = WeightedMLEFitter(**fitter_kwargs).fit(  # type: ignore[arg-type]
        df_train, reference_date=reference_date
    )

    # Stage 2 — DIBP fit (or skip)
    if use_dibp:
        dibp_fit = fit_dibp(
            calibration,
            strengths,
            home_col="home_team",
            away_col="away_team",
            max_goals=max_goals,
            init_params=init_dibp,
        )
    else:
        dibp_fit = DIBPFitResult(
            params=DIBPParams(rho=0.0, pi_diag=0.0, theta_diag=0.5),
            n_matches_used=0,
            log_likelihood=float("nan"),
            converged=True,
        )

    # Stages 3+4 — produce raw markets on calibration corpus, then fit
    # beta calibrators on them.
    n_cal = 0
    n_skip = 0
    raw_1x2: list[tuple[float, float, float]] = []
    raw_btts: list[float] = []
    raw_ou: list[float] = []
    outcomes_1x2: list[int] = []  # 0=home, 1=draw, 2=away
    outcomes_btts: list[int] = []
    outcomes_ou: list[int] = []

    for row in calibration.iter_rows(named=True):
        h, a = row["home_team"], row["away_team"]
        if h not in strengths.strengths or a not in strengths.strengths:
            n_skip += 1
            continue
        mu_h, mu_a = strengths.predict_lambdas(h, a)
        grid = compute_dibp_grid(mu_h, mu_a, dibp_fit.params, max_goals=max_goals)
        m = markets_from_grid(grid)
        raw_1x2.append((m.p_home_win, m.p_draw, m.p_away_win))
        raw_btts.append(m.p_btts)
        raw_ou.append(m.p_over_2_5)

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        if hg > ag:
            outcomes_1x2.append(0)
        elif hg == ag:
            outcomes_1x2.append(1)
        else:
            outcomes_1x2.append(2)
        outcomes_btts.append(1 if (hg >= 1 and ag >= 1) else 0)
        outcomes_ou.append(1 if (hg + ag) >= 3 else 0)
        n_cal += 1

    calibrators = V2Calibrators()
    if use_beta_calibration and n_cal > 0:
        cal_1x2 = BetaCalibrator().fit(np.array(raw_1x2), np.array(outcomes_1x2))
        cal_btts = BetaCalibrator().fit(np.array(raw_btts), np.array(outcomes_btts))
        cal_ou = BetaCalibrator().fit(np.array(raw_ou), np.array(outcomes_ou))
        calibrators = V2Calibrators(one_x_two=cal_1x2, btts=cal_btts, ou_2_5=cal_ou)

    # Wave 1.A: load + compute market-value offsets once, post-MLE, pre-predictor.
    market_offsets: dict[str, float] | None = None
    if use_market_value:
        assert market_values_path is not None  # guarded above
        squad_values = load_squad_values(market_values_path)
        market_offsets = compute_offsets(squad_values, beta_mv=market_value_beta)

    predictor = CalibratedDIBPPredictor(
        strengths=strengths,
        dibp_params=dibp_fit.params,
        calibrators=calibrators,
        max_goals=max_goals,
        market_offsets=market_offsets,
        market_offset_mode=market_offset_mode,
    )
    return V2PipelineResult(
        predictor=predictor,
        strengths=strengths,
        dibp_fit=dibp_fit,
        n_calibration_matches=n_cal,
        calibration_skipped_cold_start=n_skip,
        market_offsets=market_offsets,
    )
