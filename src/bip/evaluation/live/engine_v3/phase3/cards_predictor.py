"""Cards-family conditional predictor — Phase 3 expansion (sec 10).

Section 3 of the v3 engine design notes:

> Cards: Poisson condicional a ref + estado

The model is intentionally compact:

    λ_remaining = expected_cards_total × (time_remaining / 90)
                × ref_multiplier
                × phase_multiplier
                × momentum_multiplier
                × numerical_disadvantage_multiplier

- ref_multiplier:    bands the referee's career cards/game vs the league
                     baseline (~4.0). Caps at [0.7, 1.5].
- phase_multiplier:  desperate/chasing → +20% (tactical fouling),
                     cruise/cagey_closed → −15% (tempo dies).
- momentum_multiplier: if the last-15-min card rate annualised already
                     exceeds the priors total → +20%.
- numerical_disadvantage_multiplier: the side that's a man down fouls
                     more — boosts λ another +10% per red.

Direction support: ``over``/``under`` on a line (e.g. total cards 5.5
over). The model returns P(over) or 1 − P(over) accordingly.

What's *not* in v1: per-player cards markets (player props live in
``props_predictor.py``), team-specific cards markets (a single-side
slice of the family — easily added later by scaling λ to that side's
share of cumulative cards).
"""
from __future__ import annotations

import math

from bip.evaluation.live.engine_v3.conditional_predictor import (
    PredictionPoint,
    _parse_line,
    _poisson_ge,
)
from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis


_LEAGUE_CARD_BASELINE = 4.0  # avg cards/game in top-5 European leagues
_DESPERATE_PHASES: frozenset[str] = frozenset(
    {"desperate", "chasing", "collapsing"}
)
_QUIET_PHASES: frozenset[str] = frozenset({"cruise", "cagey_closed"})


def _ref_multiplier(gsv: GameStateVector) -> float:
    prior = gsv.cards.ref_card_rate_prior
    if prior <= 0:
        return 1.0
    ratio = prior / _LEAGUE_CARD_BASELINE
    return max(0.7, min(1.5, ratio))


def _phase_multiplier(gsv: GameStateVector) -> float:
    phase = gsv.tactical.game_phase
    if phase in _DESPERATE_PHASES:
        return 1.20
    if phase in _QUIET_PHASES:
        return 0.85
    return 1.0


def _momentum_multiplier(gsv: GameStateVector) -> float:
    """Card-rate annualised over the last 15 min vs expected."""
    if gsv.cards.card_rate_last_15min <= 0:
        return 1.0
    annualised = gsv.cards.card_rate_last_15min * 90.0
    if annualised > gsv.priors.expected_cards_total * 1.30:
        return 1.20
    if annualised < gsv.priors.expected_cards_total * 0.50:
        return 0.90
    return 1.0


def _numerical_multiplier(gsv: GameStateVector) -> float:
    """+10% per red — the inferior side fouls more for the rest of the game."""
    return 1.0 + 0.10 * (gsv.numerical.red_cards_home + gsv.numerical.red_cards_away)


class CardsPredictor:
    """Conditional Poisson predictor for total-cards markets."""

    family = MarketFamily.CARDS

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        if thesis.prediction.family != MarketFamily.CARDS:
            return None
        line = _parse_line(market_id)
        if line is None:
            return None
        already = gsv.cards.yellows[0] + gsv.cards.yellows[1] + 2 * (
            gsv.cards.reds[0] + gsv.cards.reds[1]
        )
        remaining_minutes = max(0, gsv.time.time_remaining_match)
        expected_total = gsv.priors.expected_cards_total
        lam = expected_total * (remaining_minutes / 90.0)
        lam *= _ref_multiplier(gsv)
        lam *= _phase_multiplier(gsv)
        lam *= _momentum_multiplier(gsv)
        lam *= _numerical_multiplier(gsv)

        sign = 1.0 if thesis.prediction.direction == "over" else -1.0
        lam *= (1.0 + sign * thesis.prediction.magnitude_pp)

        target = int(math.ceil(line)) if line != int(line) else int(line) + 1
        need = max(0, target - already)
        p_over = _poisson_ge(max(0.0, lam), need)
        p = p_over if thesis.prediction.direction == "over" else max(0.0, 1.0 - p_over)

        sigma = math.sqrt(max(0.01, lam)) / max(1.0, target)
        return PredictionPoint(
            p=p,
            ci_low=max(0.0, p - sigma),
            ci_high=min(1.0, p + sigma),
            family=MarketFamily.CARDS,
        )


__all__ = ["CardsPredictor"]
