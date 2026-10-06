"""NextGoalPredictor — Phase-2 hazard model for the NEXT_GOAL family.

Replaces the Phase-1 fallback to Goals2HPredictor for NEXT_GOAL theses.
The difference matters: Goals2H predicts P(over X.5 total goals), which
is the wrong quantity when the thesis is "the next goal will be home".

For markets that explicitly express "next goal":
    - team_to_score_first
    - first_goal_home / first_goal_away
    - next_goal_home / next_goal_away (synthetic in Sportmonks)

we want P(home scores before away in the remaining match time).

Hazard model
============

Treat each team's goal arrival as an independent inhomogeneous Poisson
process with state-conditional rate λ_h(t), λ_a(t):

    λ_h(t) = λ_h_base × momentum_h(t) × phase_mult(t)
    λ_a(t) = λ_a_base × momentum_a(t) × phase_mult(t)

with:
- ``λ_base``: pre-match per-minute Poisson rate (priors / 90)
- ``momentum``: ``1 + α × (recent_xg_per_min − typical_rate) / typical_rate``,
  capped on both sides. α defaults to 0.6 (60% of the rate signal weight).
- ``phase_mult``: same multipliers as Goals2HPredictor (cagey_closed→0.55,
  open_attacking→1.00, desperate→1.30 etc.)

The thesis ``magnitude_pp`` further nudges the rate in the thesis's
predicted direction (home/away). For Napoli (dominant_losing + 25-45'),
the dominant team is the predicted scorer; we boost its rate by
``magnitude_pp`` and lower the opponent's by half that.

Closed-form: under independent Poisson with constant rates over the
remaining window τ (we assume momentum stable over the rest of half):

    P(home scores next within τ)
      = λ_h / (λ_h + λ_a) × (1 − exp(−(λ_h + λ_a) × τ))

Two pieces:
  - ``λ_h / (λ_h + λ_a)``: home's share of total intensity (= P(home
    scores first conditional on at least one goal).
  - ``1 − exp(−(λ_h + λ_a) × τ)``: P(at least one goal in remaining τ).

For markets that ask "first goal in the FIRST HALF only" or similar,
``τ`` is ``time_remaining_half`` instead of ``time_remaining_match``.

Routing
=======

The predictor accepts only NEXT_GOAL theses. The market_id must
- Map to MarketFamily.NEXT_GOAL via ``family_for_market_id``
- Contain a "home"/"away"/"draw"/"no_goal"/"neither" token so we know
  which side_a the market expresses.

Mismatch (e.g. thesis predicts home, market is the away side) → None.
"""
from __future__ import annotations

import math

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis
from bip.evaluation.live.engine_v3.conditional_predictor import PredictionPoint


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


_PHASE_MULT: dict[str, float] = {
    "cagey_closed": 0.55,
    "cruise": 0.55,
    "cagey_open": 0.80,
    "open_attacking": 1.00,
    "desperate": 1.30,
}


def _typical_per_min_rate(prior_lam: float) -> float:
    """A "typical" per-minute rate from the pre-match Poisson λ."""
    return max(0.001, prior_lam / 90.0)


def _momentum_multiplier(
    recent_xg_per_min: float,
    typical_rate: float,
    alpha: float = 0.6,
    floor: float = 0.4,
    cap: float = 2.5,
) -> float:
    """Bound momentum into [floor, cap]. Default α = 0.6 means a team
    converting at 2× typical rate gets ``1 + 0.6 × (2 − 1) = 1.6``."""
    if typical_rate <= 0:
        return 1.0
    raw = 1.0 + alpha * (recent_xg_per_min - typical_rate) / typical_rate
    return max(floor, min(cap, raw))


def _parse_side_for_next_goal(market_id: str, gsv: GameStateVector) -> str | None:
    """Identify which side_a a next-goal market expresses.

    Returns "home", "away", "neither", or None if undecidable.

    Examples we want to handle:
        - team_to_score_first_home  → home
        - team_to_score_first_away  → away
        - team_to_score_first_no_goal → neither
        - first_goal_home → home
        - next_goal_real_betis (templated by team name) → check ids
    """
    mid = market_id.lower()
    if "no_goal" in mid or "_neither" in mid or "no_more_goals" in mid:
        return "neither"
    if "home" in mid or "_1" in mid:
        return "home"
    if "away" in mid or "_2" in mid:
        return "away"
    # Templated by team name — try to match home_team_name or away_team_name
    home_slug = (gsv.home_team_name or "").lower().replace(" ", "_")
    away_slug = (gsv.away_team_name or "").lower().replace(" ", "_")
    if home_slug and home_slug in mid:
        return "home"
    if away_slug and away_slug in mid:
        return "away"
    return None


# ──────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────


class NextGoalPredictor:
    """Hazard-based next-goal-team predictor."""

    family = MarketFamily.NEXT_GOAL

    def __init__(
        self,
        *,
        momentum_alpha: float = 0.6,
        first_half_minute_cutoff: int = 45,
    ) -> None:
        self._alpha = momentum_alpha
        self._fh_cutoff = first_half_minute_cutoff

    def _rates(self, gsv: GameStateVector) -> tuple[float, float]:
        """Returns (λ_home_per_min, λ_away_per_min) conditioned on state."""
        lam_h_base = _typical_per_min_rate(gsv.priors.lambda_home_prematch)
        lam_a_base = _typical_per_min_rate(gsv.priors.lambda_away_prematch)
        typical = (lam_h_base + lam_a_base) / 2.0
        m_home = _momentum_multiplier(
            gsv.xg.xg_per_min_home_last_15, typical, self._alpha
        )
        m_away = _momentum_multiplier(
            gsv.xg.xg_per_min_away_last_15, typical, self._alpha
        )
        phase_mult = _PHASE_MULT.get(gsv.tactical.game_phase, 1.0)
        return (
            lam_h_base * m_home * phase_mult,
            lam_a_base * m_away * phase_mult,
        )

    def _remaining_minutes(self, gsv: GameStateVector, market_id: str) -> float:
        """Pick the right horizon: 'first half' markets use
        time_remaining_half, otherwise time_remaining_match."""
        if "first_half" in market_id.lower() or "1h" in market_id.lower():
            return max(0.0, float(gsv.time.time_remaining_half))
        return max(0.0, float(gsv.time.time_remaining_match))

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        if thesis.prediction.family != MarketFamily.NEXT_GOAL:
            return None
        from bip.evaluation.live.engine_v3.market_selector import (
            family_for_market_id,
        )
        if family_for_market_id(market_id) != MarketFamily.NEXT_GOAL:
            return None
        side_a = _parse_side_for_next_goal(market_id, gsv)
        if side_a is None:
            return None
        # The thesis predicts home or away (NEXT_GOAL theses always pick
        # a team-side direction). Map and require alignment with side_a.
        d = thesis.prediction.direction
        if d not in ("home", "away"):
            return None
        if d != side_a and side_a != "neither":
            return None

        lam_h, lam_a = self._rates(gsv)

        # Thesis magnitude_pp boosts the predicted side's rate.
        # We do NOT double-count if d == side_a; same boost either way.
        boost = thesis.prediction.magnitude_pp
        if d == "home":
            lam_h *= (1.0 + boost)
            lam_a *= max(0.1, 1.0 - 0.5 * boost)
        elif d == "away":
            lam_a *= (1.0 + boost)
            lam_h *= max(0.1, 1.0 - 0.5 * boost)

        tau = self._remaining_minutes(gsv, market_id)
        total = lam_h + lam_a
        if total <= 0 or tau <= 0:
            return None
        p_any_goal = 1.0 - math.exp(-total * tau)
        p_home_first = lam_h / total
        p_away_first = lam_a / total
        if side_a == "home":
            p = p_home_first * p_any_goal
        elif side_a == "away":
            p = p_away_first * p_any_goal
        else:  # neither / no_goal
            p = 1.0 - p_any_goal

        sigma = math.sqrt(p * max(1e-6, 1.0 - p))
        return PredictionPoint(
            p=max(0.0, min(1.0, p)),
            ci_low=max(0.0, p - 0.5 * sigma),
            ci_high=min(1.0, p + 0.5 * sigma),
            family=MarketFamily.NEXT_GOAL,
        )


__all__ = ["NextGoalPredictor"]
