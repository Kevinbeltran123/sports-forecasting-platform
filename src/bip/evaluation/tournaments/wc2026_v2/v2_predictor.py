"""Ensemble predictor v2 — Ley 2019 prior → DIBP → beta calibration.

Composes the three building blocks from Olas 1-3 into a single predictor
the Ola 5 backtest and Ola 7 lock emitter can consume:

    (home_team, away_team)
        → WeightedMLEResult.predict_lambdas(...)         # Ola 1 prior
        → compute_dibp_grid(μ_h, μ_a, dibp_params)        # Ola 3 grid
        → markets_from_grid(grid)                         # raw markets
        → BetaCalibrator.transform(...)                   # Ola 2 calibration
        → CalibratedDIBPPrediction

Cold-start handling: teams missing from the strength prior fall back to a
cohort-median (attack = defense = 0 since the prior is mean-centred) and
the prediction is flagged via ``is_cold_start=True``. This mirrors the
existing lock's transparent cold-start handling for the 11 confederation
play-off slot teams (per internal research notes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beta_calibrator import BetaCalibrator
from .dibp import DIBPParams, compute_dibp_grid, markets_from_grid
from .weighted_strength import TeamStrength, WeightedMLEResult


@dataclass(frozen=True)
class V2Calibrators:
    """Per-market calibrators trained on validation-corpus raw probabilities."""

    one_x_two: BetaCalibrator | None = None
    btts: BetaCalibrator | None = None
    ou_2_5: BetaCalibrator | None = None


@dataclass(frozen=True)
class CalibratedDIBPPrediction:
    """Single-fixture prediction payload.

    - ``mu_h_raw``, ``mu_a_raw``: log-rate-derived goal expectancies before
      DIBP / calibration. Useful for diagnostics + market-line overlay
      (e.g., comparing to a Pinnacle AH-implied goal margin).
    - ``is_cold_start``: True if either team was missing from the prior;
      cohort fallback (attack=defense=0) was used.
    - ``cold_start_teams``: tuple of team names that triggered fallback.
    """

    home_team: str
    away_team: str
    mu_h_raw: float
    mu_a_raw: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_btts: float
    p_over_2_5: float
    p_under_2_5: float
    expected_total_goals: float
    is_cold_start: bool
    cold_start_teams: tuple[str, ...]


class CalibratedDIBPPredictor:
    """v2 ensemble predictor — weighted-MLE prior → DIBP → beta calibration."""

    def __init__(
        self,
        strengths: WeightedMLEResult,
        dibp_params: DIBPParams,
        calibrators: V2Calibrators | None = None,
        max_goals: int = 8,
        market_offsets: dict[str, float] | None = None,
        market_offset_mode: str = "attack",
    ) -> None:
        self.strengths = strengths
        self.dibp_params = dibp_params
        self.calibrators = calibrators or V2Calibrators()
        self.max_goals = max_goals
        # Wave 1.A (Peeters 2018): optional additive log-rate offsets
        # injected at _resolve_lambdas time. ``None`` (default) preserves
        # exact v2 behavior — the v2 test contract relies on this.
        self.market_offsets = market_offsets
        if market_offset_mode not in {"attack", "symmetric"}:
            raise ValueError(
                f"market_offset_mode must be 'attack' or 'symmetric', got {market_offset_mode!r}"
            )
        # 'attack': richer team scores more (offset added to attack only).
        # 'symmetric': richer team also defends better (offset subtracted
        # from opponent's defense term). Doubles the effect magnitude per
        # offset unit; β needs to be ~halved to compare with 'attack' mode.
        self.market_offset_mode = market_offset_mode

    # ─────────────────────────────────────────────────────────────
    # Raw rate lookup with cold-start fallback
    # ─────────────────────────────────────────────────────────────

    def _resolve_lambdas(
        self, home_team: str, away_team: str
    ) -> tuple[float, float, tuple[str, ...]]:
        """Look up (μ_h, μ_a), substituting cohort priors for cold-start teams.

        When ``self.market_offsets`` is non-None, the offset for each team
        is added to its attack term (Peeters 2018 mechanism). Teams missing
        from the offsets dict get offset 0.0 — so partial coverage degrades
        gracefully (richer-than-median teams without TM data are treated
        as median; bias toward underestimating strength of teams without
        market data is acknowledged but preferred to silently skipping the
        fixture).
        """

        cold: list[str] = []
        # The prior is mean-centred (mean(attack) = mean(defense) = 0), so
        # the cohort-median substitution is a TeamStrength(team=name, 0, 0).
        zero = TeamStrength(team="cold_start", attack=0.0, defense=0.0)
        h = self.strengths.strengths.get(home_team)
        if h is None:
            cold.append(home_team)
            h = zero
        a = self.strengths.strengths.get(away_team)
        if a is None:
            cold.append(away_team)
            a = zero

        if self.market_offsets is not None:
            off_h = float(self.market_offsets.get(home_team, 0.0))
            off_a = float(self.market_offsets.get(away_team, 0.0))
        else:
            off_h = 0.0
            off_a = 0.0

        if self.market_offset_mode == "symmetric":
            # Richer team attacks more AND defends better. Subtracting the
            # OPPONENT's offset from one's own defense raises one's own
            # log_lambda for the opponent's rate (i.e., opponent scores less
            # when they face a high-value defense).
            log_lambda_h = (
                self.strengths.intercept
                + (h.attack + off_h)
                + (a.defense - off_a)
                + self.strengths.home_advantage
            )
            log_lambda_a = (
                self.strengths.intercept + (a.attack + off_a) + (h.defense - off_h)
            )
        else:
            # 'attack' (default). Backward-compatible with Wave 1.A.2.
            log_lambda_h = (
                self.strengths.intercept
                + (h.attack + off_h)
                + a.defense
                + self.strengths.home_advantage
            )
            log_lambda_a = self.strengths.intercept + (a.attack + off_a) + h.defense
        return float(np.exp(log_lambda_h)), float(np.exp(log_lambda_a)), tuple(cold)

    # ─────────────────────────────────────────────────────────────
    # predict()
    # ─────────────────────────────────────────────────────────────

    def predict(self, home_team: str, away_team: str) -> CalibratedDIBPPrediction:
        mu_h, mu_a, cold = self._resolve_lambdas(home_team, away_team)
        grid = compute_dibp_grid(mu_h, mu_a, self.dibp_params, max_goals=self.max_goals)
        raw = markets_from_grid(grid)

        # Apply calibrators where provided. 1X2 is 3-class; BTTS and OU2.5
        # are binary (P(positive class)).
        if self.calibrators.one_x_two is not None:
            cal_1x2 = self.calibrators.one_x_two.transform(
                np.array([[raw.p_home_win, raw.p_draw, raw.p_away_win]])
            )[0]
            p_h, p_d, p_a = float(cal_1x2[0]), float(cal_1x2[1]), float(cal_1x2[2])
        else:
            p_h, p_d, p_a = raw.p_home_win, raw.p_draw, raw.p_away_win

        if self.calibrators.btts is not None:
            p_btts = float(self.calibrators.btts.transform(np.array([raw.p_btts]))[0])
        else:
            p_btts = raw.p_btts

        if self.calibrators.ou_2_5 is not None:
            p_over = float(self.calibrators.ou_2_5.transform(np.array([raw.p_over_2_5]))[0])
        else:
            p_over = raw.p_over_2_5
        p_under = 1.0 - p_over

        return CalibratedDIBPPrediction(
            home_team=home_team,
            away_team=away_team,
            mu_h_raw=mu_h,
            mu_a_raw=mu_a,
            p_home_win=p_h,
            p_draw=p_d,
            p_away_win=p_a,
            p_btts=p_btts,
            p_over_2_5=p_over,
            p_under_2_5=p_under,
            expected_total_goals=raw.expected_total_goals,
            is_cold_start=bool(cold),
            cold_start_teams=cold,
        )
