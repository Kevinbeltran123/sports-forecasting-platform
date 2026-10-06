"""Causal Thesis primitive — section 2.2 of the v3 engine design notes.

A thesis is the **minimum formal structure** for a causal bet rationale:

    Thesis {
        id, archetype, premise, mechanism, prediction,
        horizon, invalidation_triggers, confidence_prior, source
    }

Why this structure is non-negotiable: a thesis WITHOUT
``invalidation_triggers`` is unfalsifiable — the system can't say "this
prediction should die if X happens", only "this prediction has EV +5%".
That's exactly the failure mode of the current EV-filter system. Every
v3 pick MUST carry a falsifiable thesis.

Two collateral primitives:

- ``CausalChain`` — a chain of ``cause → effect (because of M)`` steps.
  Three-step depth is the operating norm; deeper chains are a smell.
- ``ConditionalShift`` — the structured prediction object (market family,
  direction, magnitude expectation, time horizon).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.live.engine_v3.gsv import Direction

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")


# ── Archetype catalog ───────────────────────────────────────────────────


class ThesisArchetype(str, Enum):
    """The 12 archetypes from sec 4-bis. Each rule-layer detector emits
    a thesis tagged with one of these IDs. The taxonomy is closed — a
    new archetype requires a new enum entry AND ≥30 backtest examples
    (per risk #1 mitigation in sec 8)."""

    RED_CARD_AWAY_EARLY = "red_card_away_early"               # #1
    DOMINANT_LOSING_NAPOLI = "dominant_losing_napoli"         # #2
    LATE_CAGEY_ZERO_ZERO = "late_cagey_zero_zero"             # #3
    LEAD_TWO_DEFENSIVE_SUB = "lead_two_defensive_sub"         # #4
    REGRESSION_TO_XG = "regression_to_xg"                     # #5
    CARDS_MOMENTUM_STRICT_REF = "cards_momentum_strict_ref"   # #6
    UNDERDOG_LEADS_SIEGE = "underdog_leads_siege"             # #7
    OPEN_GAME_FORMATIONS = "open_game_formations"             # #8
    KEY_PLAYMAKER_OFF = "key_playmaker_off"                   # #9
    SECOND_HALF_RESET = "second_half_reset"                   # #10
    NUMERICAL_SUSTAINED = "numerical_sustained"               # #11
    CRUISE_MODE = "cruise_mode"                               # #12


# Allow-listed archetypes that MAY emit Under-direction picks even when
# ``score.dominant_losing=True``. Used by no-bet rule #2 (anti-Napoli).
UNDER_DIRECTION_ALLOWED_ARCHETYPES: frozenset[ThesisArchetype] = frozenset(
    {ThesisArchetype.CRUISE_MODE}
)


# ── Causal mechanism primitives ─────────────────────────────────────────


class CausalStep(BaseModel):
    """One step in a ``cause → effect`` chain."""

    model_config = _MODEL_CONFIG
    cause: str
    effect: str
    mechanism: str  # the "because of M" clause


class CausalChain(BaseModel):
    """Ordered chain of causal steps. Length ≤ 3 is the operating norm.

    Longer chains compound prior uncertainty — each step costs ≈ 0.1
    confidence on average. A chain of 4+ steps is a hypothesis that
    needs decomposition into multiple smaller theses, not a single
    fragile mega-thesis."""

    model_config = _MODEL_CONFIG
    steps: list[CausalStep] = Field(min_length=1, max_length=3)


# ── Premise predicates ──────────────────────────────────────────────────


class GSVPredicate(BaseModel):
    """A single predicate over the GSV.

    The ``path`` is a dotted accessor (``score.dominant_losing``,
    ``time.minute``, ``tactical.home_phase``). The ``op`` is a small
    closed set so we can evaluate uniformly. This is *deliberately*
    restricted — anything that needs Python lambdas instead lives in
    the archetype's match function (``archetypes.py``).
    """

    model_config = _MODEL_CONFIG
    path: str
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "between", "contains"]
    value: Any


# ── Prediction primitives ───────────────────────────────────────────────


class TimeWindow(BaseModel):
    """How long the thesis is expected to be valid."""

    model_config = _MODEL_CONFIG
    horizon_minutes: int
    label: Literal["next_15", "next_30", "rest_of_half", "rest_of_match", "EoM"]


class MarketFamily(str, Enum):
    GOALS = "goals"
    BTTS = "btts"
    CORNERS = "corners"
    CARDS = "cards"
    NEXT_GOAL = "next_goal"
    NEXT_CORNER = "next_corner"
    PROPS = "props"
    RESULT_1X2 = "result_1x2"
    DOUBLE_CHANCE = "double_chance"
    ASIAN_HANDICAP = "asian_handicap"
    RACE_TO_X = "race_to_x"


class ConditionalShift(BaseModel):
    """The structured prediction: which outcome shifts, in which direction,
    with what magnitude band, over what time horizon."""

    model_config = _MODEL_CONFIG
    family: MarketFamily
    direction: Direction  # over/under/yes/no/home/away/draw
    magnitude_pp: float = 0.0  # expected probability shift vs pre-match
    horizon: TimeWindow


# ── Invalidation events ─────────────────────────────────────────────────


class InvalidationTrigger(BaseModel):
    """An event that kills the thesis. AT LEAST ONE must be supplied;
    the validator in ``Thesis`` enforces non-emptiness."""

    model_config = _MODEL_CONFIG
    kind: Literal[
        "goal_for_dominant",
        "goal_for_underdog",
        "any_goal",
        "red_card_dominant",
        "red_card_underdog",
        "key_sub_dominant_offensive",
        "key_sub_underdog_defensive",
        "halftime_reached",
        "formation_change",
        "ref_card_strict_regime",
        "minute_threshold_passed",
        "xg_normalised",
        "possession_flipped",
    ]
    description: str  # human-readable, surfaces in the Telegram audit footer
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Source provenance ──────────────────────────────────────────────────


class ThesisSource(BaseModel):
    """Where the thesis came from — rule layer, pattern layer, or LLM."""

    model_config = _MODEL_CONFIG
    layer: Literal["rule", "pattern", "commentary"]
    identifier: str  # rule_id, pattern_id, or commentary extract


# ── Top-level thesis ────────────────────────────────────────────────────


class Thesis(BaseModel):
    """A falsifiable, auditable causal bet rationale.

    Two structural invariants enforced by Pydantic:
    - ``invalidation_triggers`` MUST be non-empty (no unfalsifiable theses)
    - ``confidence_prior`` ∈ [0, 1] (no out-of-range priors)

    Construction is by archetypes (``archetypes.py``) or pattern
    layer — never by hand in the value detector. This keeps thesis
    provenance auditable end-to-end.
    """

    model_config = _MODEL_CONFIG

    id: str
    archetype: ThesisArchetype
    premise: list[GSVPredicate]
    mechanism: CausalChain
    prediction: ConditionalShift
    invalidation_triggers: list[InvalidationTrigger] = Field(min_length=1)
    confidence_prior: float = Field(ge=0.0, le=1.0)
    source: ThesisSource
    activated_at_minute: int


HorizonLabel = Literal["next_15", "next_30", "rest_of_half", "rest_of_match", "EoM"]


def build_horizon(horizon_label: HorizonLabel, minute: int) -> TimeWindow:
    """Translate a label and current minute into a concrete window."""
    mapping: dict[HorizonLabel, int] = {
        "next_15": 15,
        "next_30": 30,
        "rest_of_half": max(1, (45 if minute < 45 else 90) - minute),
        "rest_of_match": max(1, 90 - minute),
        "EoM": max(1, 90 - minute),
    }
    return TimeWindow(horizon_minutes=mapping[horizon_label], label=horizon_label)


__all__ = [
    "CausalChain",
    "CausalStep",
    "ConditionalShift",
    "GSVPredicate",
    "InvalidationTrigger",
    "MarketFamily",
    "Thesis",
    "ThesisArchetype",
    "ThesisSource",
    "TimeWindow",
    "UNDER_DIRECTION_ALLOWED_ARCHETYPES",
    "build_horizon",
]
