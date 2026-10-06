"""Player Props predictor — shrinkage to pre-match + live adjusters.

Section 3 of the v3 engine design notes:

> Props: shrinkage al pre-match + adjusters live

This is a **shadow-mode** component for Phase 3 (sec 10): the system
generates props predictions but every props pick remains shadow-only
until 100-200 resolved picks accumulate (sec 5.5). The promoter in
``tier_promoter.py`` enforces that — props candidates short-circuit
to ``stage="shadow"`` regardless of MES.

Two markets covered initially:

- ``player_X_to_be_carded``:
    base = career_cards_per_min × remaining_minutes
    adjuster = +250% if player on yellow
              + 40% if game_phase ∈ {chasing, desperate}
              + 25% if player on inferior side (red card)

- ``player_X_to_score``:
    base = career_xg_per_min × remaining_minutes
    adjuster = +30% if player on trailing dominant side past 75'
              + 20% if game_phase = desperate

We don't model the exact-anytime distinction here — Phase 3 keeps
"player X to score" as a single hazard event. Later phases can split
into multiple windows.

PlayerPrior carries the per-player career rates. In production, fed
from the Sportmonks player history endpoint cached weekly. For tests,
synthesised by the fixture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from bip.evaluation.live.engine_v3.conditional_predictor import PredictionPoint
from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis


@dataclass(frozen=True)
class PlayerPrior:
    """Per-player rate priors. Fed externally (cache or DB)."""

    player_id: int
    team_id: int
    cards_per_90: float = 0.10        # 1 in 10 games = baseline
    xg_per_90: float = 0.15           # baseline non-striker xG/90
    role: str = "unknown"             # "striker"/"midfielder"/"defender"


def _parse_player_id(market_id: str) -> int | None:
    """Extract player_id from a market like ``player_3421_to_be_carded``."""
    parts = market_id.split("_")
    if len(parts) < 2 or parts[0] != "player":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _market_kind(market_id: str) -> str | None:
    if "to_be_carded" in market_id:
        return "to_be_carded"
    if "to_score" in market_id or "anytime_scorer" in market_id:
        return "to_score"
    return None


class PlayerPropsPredictor:
    """Returns shadow-mode predictions for player_props markets.

    Phase 3 contract: ALWAYS return a prediction (so the selector can
    score it), but the Tier-D promoter must check the prediction's
    ``family`` and refuse to route props to anything beyond shadow
    until calibration accumulates.
    """

    family = MarketFamily.PROPS

    def __init__(self, player_priors: dict[int, PlayerPrior] | None = None) -> None:
        self._priors = player_priors or {}

    def add_prior(self, prior: PlayerPrior) -> None:
        self._priors[prior.player_id] = prior

    # ── kind-specific hazards ───────────────────────────────────────────

    def _carded_hazard(
        self, prior: PlayerPrior, gsv: GameStateVector,
    ) -> float:
        remaining = max(0, gsv.time.time_remaining_match)
        base = prior.cards_per_90 * (remaining / 90.0)
        booked = (
            prior.player_id in {b.player_id for b in gsv.cards.players_on_yellow}
        )
        if booked:
            base *= 3.5  # second-yellow risk
        if gsv.tactical.game_phase in {"chasing", "desperate"}:
            base *= 1.40
        # If team is a man down, this player fouls more
        if prior.team_id == gsv.home_team_id and gsv.numerical.red_cards_home > 0:
            base *= 1.25
        if prior.team_id == gsv.away_team_id and gsv.numerical.red_cards_away > 0:
            base *= 1.25
        return base

    def _score_hazard(
        self, prior: PlayerPrior, gsv: GameStateVector,
    ) -> float:
        remaining = max(0, gsv.time.time_remaining_match)
        base = prior.xg_per_90 * (remaining / 90.0)
        # Trailing dominant-side push, post 75'
        is_dominant_side = (
            prior.team_id == gsv.score.dominant_team_id
        )
        if is_dominant_side and gsv.score.dominant_losing and gsv.time.minute >= 75:
            base *= 1.30
        if gsv.tactical.game_phase == "desperate":
            base *= 1.20
        return base

    # ── public ──────────────────────────────────────────────────────────

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        if thesis.prediction.family != MarketFamily.PROPS:
            return None
        player_id = _parse_player_id(market_id)
        if player_id is None:
            return None
        prior = self._priors.get(player_id)
        if prior is None:
            return None
        kind = _market_kind(market_id)
        if kind == "to_be_carded":
            lam = self._carded_hazard(prior, gsv)
        elif kind == "to_score":
            lam = self._score_hazard(prior, gsv)
        else:
            return None
        p_event = 1.0 - math.exp(-max(0.0, lam))
        if thesis.prediction.direction == "no":
            p = 1.0 - p_event
        else:
            p = p_event
        # Wide CI band for props in v1 — calibration is shadow-only.
        sigma = max(0.10, math.sqrt(lam) / 3.0)
        return PredictionPoint(
            p=max(0.0, min(1.0, p)),
            ci_low=max(0.0, p - sigma),
            ci_high=min(1.0, p + sigma),
            family=MarketFamily.PROPS,
        )


__all__ = ["PlayerPrior", "PlayerPropsPredictor"]
