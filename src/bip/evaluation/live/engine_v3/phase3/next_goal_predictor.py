"""NextGoal-family predictor — hazard point model with state covariates.

Section 3 of the v3 engine design notes:

> Next-X: hazard puntual con state covariates

For team-specific "next goal" markets we model the two competing
hazards (home λ_h, away λ_a) and combine via the standard race
formula::

    P(no goal in window)    = exp(-(λ_h + λ_a) · w)
    P(home next | goal in w) = λ_h / (λ_h + λ_a)
    P(home scores next)      = (1 − P(no goal)) × λ_h / (λ_h + λ_a)

The per-team hazard rates blend:
- The minute-prorated pre-match λ (priors.lambda_*)
- The observed live-xG rate over the last 15 minutes (decays the prior
  by ``minute / (minute + 18)``)
- A game-phase multiplier (desperate/chasing speeds it up, cruise
  slows it down).

Direction support: ``home`` / ``away`` / ``draw`` (= "no goal in window
remains").
"""
from __future__ import annotations

import math

from bip.evaluation.live.engine_v3.conditional_predictor import PredictionPoint
from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis


_FAST_PHASES: frozenset[str] = frozenset({"desperate", "open_attacking"})
_SLOW_PHASES: frozenset[str] = frozenset({"cruise", "cagey_closed", "cagey_open"})


def _phase_factor(gsv: GameStateVector) -> float:
    if gsv.tactical.game_phase in _FAST_PHASES:
        return 1.30
    if gsv.tactical.game_phase in _SLOW_PHASES:
        return 0.65
    return 1.0


def _blended_lambda(
    minute: int,
    minute_prior_blend: float,
    prior_lambda: float,
    live_per_min: float,
) -> float:
    """Blend pre-match prior with live rate, weighted by elapsed minutes.

    ``minute / (minute + minute_prior_blend)`` = the live weight.
    Match-pace ``prior_lambda / 90`` is the per-minute pre-match anchor.
    """
    minute = max(1, minute)
    w_live = minute / (minute + minute_prior_blend)
    return (1.0 - w_live) * (prior_lambda / 90.0) + w_live * live_per_min


class NextGoalPredictor:
    """Hazard-point model for ``next_goal`` markets."""

    family = MarketFamily.NEXT_GOAL

    def __init__(self, default_window_minutes: int = 15,
                 minute_prior_blend: float = 18.0) -> None:
        self.default_window_minutes = default_window_minutes
        self.minute_prior_blend = minute_prior_blend

    def _per_side_lambda(self, gsv: GameStateVector, side: str) -> float:
        prior = (
            gsv.priors.lambda_home_prematch if side == "home"
            else gsv.priors.lambda_away_prematch
        )
        live = (
            gsv.xg.xg_per_min_home_last_15 if side == "home"
            else gsv.xg.xg_per_min_away_last_15
        )
        lam_per_min = _blended_lambda(
            gsv.time.minute, self.minute_prior_blend, prior, live,
        )
        return max(1e-4, lam_per_min * _phase_factor(gsv))

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        if thesis.prediction.family != MarketFamily.NEXT_GOAL:
            return None
        # window: thesis horizon (in minutes) capped at remaining match minutes
        window = min(
            thesis.prediction.horizon.horizon_minutes,
            max(1, gsv.time.time_remaining_match),
        )
        lam_h = self._per_side_lambda(gsv, "home")
        lam_a = self._per_side_lambda(gsv, "away")

        # Thesis magnitude nudges the favoured side's hazard
        if thesis.prediction.direction == "home":
            lam_h *= 1.0 + thesis.prediction.magnitude_pp
        elif thesis.prediction.direction == "away":
            lam_a *= 1.0 + thesis.prediction.magnitude_pp

        total = lam_h + lam_a
        if total <= 0:
            return None
        p_any_goal = 1.0 - math.exp(-total * window)
        share_home = lam_h / total
        share_away = lam_a / total

        if thesis.prediction.direction == "home":
            p = p_any_goal * share_home
        elif thesis.prediction.direction == "away":
            p = p_any_goal * share_away
        elif thesis.prediction.direction == "draw":
            p = 1.0 - p_any_goal
        else:
            return None

        # Crude CI: ± 0.5 / sqrt(window × total) — wider when shorter window
        if total * window > 0:
            sigma = 0.5 / math.sqrt(total * window)
        else:
            sigma = 0.5
        return PredictionPoint(
            p=max(0.0, min(1.0, p)),
            ci_low=max(0.0, p - sigma),
            ci_high=min(1.0, p + sigma),
            family=MarketFamily.NEXT_GOAL,
        )


__all__ = ["NextGoalPredictor"]
