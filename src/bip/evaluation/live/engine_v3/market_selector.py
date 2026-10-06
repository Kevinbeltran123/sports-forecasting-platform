"""Market Selector — section 3 of the v3 engine design notes.

For each generated thesis, enumerate available markets in the
``MarketSnapshot`` and compute MES per (thesis, market). Pick the top-K
per thesis. **May return empty** — that is the whole point: if no
market expresses the thesis well, we abort instead of defaulting to
the 3 markets the system calibrates well.

Sec 3 architectural note: this component is **separated** from the
Conditional Predictor on purpose. Routing is a decision, prediction is
an estimation. Mixing them is the bug the v3 fixes.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.mes import MESResult, compute_mes
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis


# ──────────────────────────────────────────────────────────────────────
# Family routing — map sportmonks-ish market IDs to v3 families
# ──────────────────────────────────────────────────────────────────────


_FAMILY_BY_MARKET_ID_PREFIX: tuple[tuple[str, MarketFamily], ...] = (
    ("match_corners", MarketFamily.CORNERS),
    ("first_half_corners", MarketFamily.CORNERS),
    ("second_half_corners", MarketFamily.CORNERS),
    ("asian_corners", MarketFamily.CORNERS),
    ("corners_race", MarketFamily.NEXT_CORNER),
    ("next_corner", MarketFamily.NEXT_CORNER),
    ("first_goal", MarketFamily.NEXT_GOAL),
    ("next_goal", MarketFamily.NEXT_GOAL),
    ("team_to_score_first", MarketFamily.NEXT_GOAL),
    ("number_of_cards", MarketFamily.CARDS),
    ("cards", MarketFamily.CARDS),
    ("player_", MarketFamily.PROPS),
    ("prop_", MarketFamily.PROPS),
    ("btts", MarketFamily.BTTS),
    ("both_teams_to_score", MarketFamily.BTTS),
    ("match_goals", MarketFamily.GOALS),
    ("first_half_goals", MarketFamily.GOALS),
    ("alternative_match_goals", MarketFamily.GOALS),
    ("goal_line", MarketFamily.GOALS),
    ("asian_handicap", MarketFamily.ASIAN_HANDICAP),
    ("double_chance", MarketFamily.DOUBLE_CHANCE),
    ("fulltime_result", MarketFamily.RESULT_1X2),
    ("race_to", MarketFamily.RACE_TO_X),
)


def family_for_market_id(market_id: str) -> MarketFamily | None:
    """Best-effort mapping. Returns ``None`` for unknown IDs (which
    won't get routed)."""
    mid = market_id.lower()
    for prefix, family in _FAMILY_BY_MARKET_ID_PREFIX:
        if prefix in mid:
            return family
    return None


# ──────────────────────────────────────────────────────────────────────
# Thesis-vs-Market family compatibility
#
# A thesis predicts within ONE market family. Routing it to a market of
# a different family is a routing bug (e.g. a CORNERS thesis emerging in
# a GOALS market because the goals-market line parses as a corner total).
# The compatibility matrix encodes which thesis families can legitimately
# emerge in which market families.
# ──────────────────────────────────────────────────────────────────────


_COMPATIBLE_MARKET_FAMILIES: dict[MarketFamily, frozenset[MarketFamily]] = {
    MarketFamily.CORNERS: frozenset({MarketFamily.CORNERS, MarketFamily.NEXT_CORNER}),
    MarketFamily.NEXT_CORNER: frozenset({MarketFamily.NEXT_CORNER, MarketFamily.CORNERS}),
    MarketFamily.GOALS: frozenset({MarketFamily.GOALS}),
    MarketFamily.BTTS: frozenset({MarketFamily.BTTS}),
    # NEXT_GOAL theses can degrade to GOALS markets (over/under) — the
    # Goals2H predictor handles the line-based mapping. They CANNOT
    # emerge in BTTS markets (semantically different bet).
    MarketFamily.NEXT_GOAL: frozenset({MarketFamily.NEXT_GOAL, MarketFamily.GOALS}),
    MarketFamily.CARDS: frozenset({MarketFamily.CARDS}),
    MarketFamily.PROPS: frozenset({MarketFamily.PROPS}),
    MarketFamily.ASIAN_HANDICAP: frozenset({MarketFamily.ASIAN_HANDICAP}),
    MarketFamily.DOUBLE_CHANCE: frozenset({MarketFamily.DOUBLE_CHANCE}),
    MarketFamily.RESULT_1X2: frozenset({MarketFamily.RESULT_1X2}),
    MarketFamily.RACE_TO_X: frozenset({MarketFamily.RACE_TO_X}),
}


def is_compatible(thesis_family: MarketFamily, market_family: MarketFamily) -> bool:
    """Return True iff a thesis of ``thesis_family`` can legitimately
    route to a market of ``market_family``."""
    return market_family in _COMPATIBLE_MARKET_FAMILIES.get(thesis_family, frozenset())


# ──────────────────────────────────────────────────────────────────────
# Fair-prob provider abstraction
# ──────────────────────────────────────────────────────────────────────


# A fair-prob provider takes a (thesis, market_id, gsv) tuple and
# returns the probability the conditional predictor assigns to the
# outcome the thesis predicts. Implemented out-of-module so the selector
# stays free of model imports — testable with a synthetic provider.
FairProbProvider = Callable[[Thesis, str, GameStateVector], float | None]


@dataclass(frozen=True)
class MarketCandidate:
    thesis: Thesis
    market_id: str
    family: MarketFamily
    fair_prob: float
    mes: MESResult


# ──────────────────────────────────────────────────────────────────────
# Selector
# ──────────────────────────────────────────────────────────────────────


def select_markets(
    theses: list[Thesis],
    gsv: GameStateVector,
    fair_prob_provider: FairProbProvider,
    *,
    top_k: int = 3,
    target_stake: float = 100.0,
    mes_threshold: float = 0.6,
) -> list[MarketCandidate]:
    """For each thesis, enumerate available markets and rank by MES.

    Drops markets where:
    - ``family_for_market_id`` returns None (unknown family)
    - the fair-prob provider returns None (predictor refuses to score
      this thesis-market pair — e.g., out of training distribution)
    - the MES falls below ``mes_threshold`` (sec 6 rule #5)

    Returns: a flat list of ``(thesis, market, MES)`` candidates,
    sorted by MES descending. Empty list = no thesis × market routing
    passed the bar; the No-Bet gate will abort downstream.
    """
    candidates: list[MarketCandidate] = []
    for thesis in theses:
        per_thesis: list[MarketCandidate] = []
        for market_id, line in gsv.markets.lines.items():
            family = family_for_market_id(market_id)
            if family is None:
                continue
            if not is_compatible(thesis.prediction.family, family):
                # Routing safety: prevent thesis-market family mismatches
                # (e.g. CORNERS thesis on a GOALS market parsing the line
                # as a corner total → inflated fair_prob = 1.0). See the
                # compatibility matrix above.
                continue
            fair_prob = fair_prob_provider(thesis, market_id, gsv)
            if fair_prob is None:
                continue
            mes = compute_mes(
                thesis, market_id, family, line, gsv,
                fair_prob=fair_prob, target_stake=target_stake,
            )
            if not mes.passes_threshold(mes_threshold):
                continue
            per_thesis.append(
                MarketCandidate(
                    thesis=thesis,
                    market_id=market_id,
                    family=family,
                    fair_prob=fair_prob,
                    mes=mes,
                )
            )
        per_thesis.sort(key=lambda c: c.mes.score, reverse=True)
        candidates.extend(per_thesis[:top_k])
    candidates.sort(key=lambda c: c.mes.score, reverse=True)
    return candidates


__all__ = [
    "FairProbProvider",
    "MarketCandidate",
    "family_for_market_id",
    "is_compatible",
    "select_markets",
]
