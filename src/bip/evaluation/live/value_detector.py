"""ValueDetector — finds picks where our probability beats the bookie's.

For each market the predictor emits, we look at the matching live odds
on the API. If our probability times the decimal odd exceeds 1 + min_edge,
we have value. We then size the stake via fractional Kelly (¼ default).

CRITICAL DESIGN CHOICES:
- Suspended / stopped odds are SKIPPED (no live trade possible)
- Only one bookmaker is currently exposed by Sportmonks at our tier
  (bookmaker_id=2). Multi-bookmaker comparison comes when the operator
  adds The Odds API or a second feed.
- Fair odds = 1 / our_probability; commission ignored (Betano-style).
- Kelly fraction is bounded at MAX_STAKE_PCT (1.5%) to protect bankroll.

HARDENING (2026-05-09 night audit):
- VALUEBET gate (B-15): Sportmonks's own value detector is a confirmation;
  if it flags the OPPOSITE selection on the same market, we drop our pick.
- CORRECT_SCORE cross-check (B-16): when the Sportmonks score grid yields
  a marginal that disagrees with us by >5pp on the same selection, flag.
- Bookmaker coherence (B-G1): when 1X2 vs DC or OU monotonicity is
  internally inconsistent (cluster A), skip those markets entirely.
- Default drop_extreme=True (B-G2): EV > 50% picks DROP rather than flag.
- Information-density floor (B-G3): below 0.20 we emit nothing live.
- Per-pick credibility-interval gate (B-G4): require edge ≥ 2 × CI.
- Stale-odd gate (C-1): drop odds whose last-update predates the last
  material event by >30s.
- Material-event blackout (C-3): suppress all picks for 90s after a
  goal/red card and from minute 89 onward.
- Bundle deduplication (B-G5): correlated cross-market picks on the same
  fixture collapse to the highest-information one.
- Logical-pick score: composite [0, 1] gate spanning consilience, odds
  credibility, liquidity, informational content, size consistency.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from bip.evaluation.live.calibration import (
    IsotonicProbabilityCalibrator,
    PerMarketCalibrator,
)
from bip.evaluation.live.commentary import (
    DEFAULT_COOLOFF_MINUTES,
    CommentaryEventType,
    recent_event as _recent_commentary_event,
)
from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.live.predictor import (
    MARKET_AWAY_CLEAN_SHEET,
    MARKET_AWAY_OU_15,
    MARKET_BTTS,
    MARKET_BTTS_FIRST_HALF,
    MARKET_BTTS_SECOND_HALF,
    MARKET_CARDS_TOTAL_3_5,
    MARKET_CARDS_TOTAL_4_5,
    MARKET_CARDS_TOTAL_5_5,
    MARKET_CORNERS_TOTAL_8_5,
    MARKET_CORNERS_TOTAL_9_5,
    MARKET_CORNERS_TOTAL_10_5,
    MARKET_CORNERS_TOTAL_11_5,
    MARKET_DOUBLE_CHANCE,
    MARKET_DRAW_NO_BET,
    MARKET_FIRST_HALF_OU_05,
    MARKET_FIRST_HALF_OU_15,
    MARKET_FIRST_HALF_RESULT,
    MARKET_FULLTIME_RESULT,
    MARKET_HOME_CLEAN_SHEET,
    MARKET_HOME_OU_15,
    MARKET_HTFT,
    MARKET_OU_05,
    MARKET_OU_15,
    MARKET_OU_25,
    MARKET_OU_35,
    MARKET_TEAM_TO_SCORE_FIRST,
    MarketProbabilities,
)
from bip.sports.football.sportmonks.schemas import Odd
from bip.sports.football.sportmonks.types import MarketID, PredictionType

logger = logging.getLogger(__name__)


# All matchers take an ``Odd`` and return either our internal selection
# key (e.g. "home"/"away"/"draw" for 1X2) or None if the odd doesn't
# correspond to any of our markets.
SelectionMatcher = Callable[[Odd], str | None]


# ── label-based matchers ────────────────────────────────────────────────────


def _ftr_matcher(odd: Odd) -> str | None:
    return {
        "1": "home", "X": "draw", "2": "away",
        "Home": "home", "Draw": "draw", "Away": "away",
    }.get(odd.label)


def _double_chance_matcher(odd: Odd) -> str | None:
    return {
        "1X": "1x", "12": "12", "X2": "x2",
        "Home or Draw": "1x", "Home or Away": "12", "Draw or Away": "x2",
    }.get(odd.label)


def _yes_no_matcher(odd: Odd) -> str | None:
    return {"Yes": "yes", "No": "no"}.get(odd.label)


def _htft_matcher(odd: Odd) -> str | None:
    """Match Sportmonks HT/FT 9-cell labels to our internal keys.

    Bug #30 (post-audit): the predictor has emitted HTFT probabilities
    since the original implementation, but the value detector was bound
    with ``_ftr_matcher`` — which only matches 1X2 single-outcome labels
    ("1", "X", "2", "Home", "Draw", "Away"). HT/FT labels are 9 cells
    formatted "first/second" (e.g., "Home/Home", "1/X", "Draw/Away").
    Result: the system burned compute on HTFT probabilities for months
    while NEVER actually picking against HTFT odds.

    This matcher accepts both numeric ("1/1", "X/2") and word ("Home/Home",
    "Draw/Away") variants and produces our internal keys
    ``{home,draw,away}_{home,draw,away}`` (9 combinations).
    """
    label = odd.label or ""
    if "/" not in label:
        return None
    parts = label.split("/", 1)
    ht_raw = parts[0].strip()
    ft_raw = parts[1].strip()
    norm = {
        "1": "home", "Home": "home", "home": "home",
        "X": "draw", "Draw": "draw", "draw": "draw",
        "2": "away", "Away": "away", "away": "away",
    }
    ht = norm.get(ht_raw)
    ft = norm.get(ft_raw)
    if ht is None or ft is None:
        return None
    return f"{ht}_{ft}"


def _draw_no_bet_matcher(odd: Odd) -> str | None:
    return {"1": "home", "2": "away", "Home": "home", "Away": "away"}.get(odd.label)


def _team_to_score_first_matcher(odd: Odd) -> str | None:
    return {
        "Home": "home", "Away": "away", "No Goal": "none", "None": "none",
        "1": "home", "2": "away",
    }.get(odd.label)


def _line_matcher_for(target_line: float) -> SelectionMatcher:
    """Returns a matcher for a specific line on a Match Goals / Half Goals odd."""
    def _match(odd: Odd) -> str | None:
        try:
            line = float(odd.total) if odd.total else None
        except (TypeError, ValueError):
            return None
        if line != target_line:
            return None
        return {"Over": "over", "Under": "under"}.get(odd.label)
    return _match


# Map our predictor market keys → (Sportmonks market_id, selection matcher).
_MARKET_BINDINGS: dict[str, tuple[int, SelectionMatcher]] = {
    MARKET_FULLTIME_RESULT: (MarketID.FULLTIME_RESULT, _ftr_matcher),
    MARKET_DOUBLE_CHANCE: (MarketID.DOUBLE_CHANCE, _double_chance_matcher),
    MARKET_DRAW_NO_BET: (MarketID.DRAW_NO_BET, _draw_no_bet_matcher),
    MARKET_FIRST_HALF_RESULT: (MarketID.HALF_TIME_RESULT, _ftr_matcher),
    MARKET_HTFT: (MarketID.HALFTIME_FULLTIME, _htft_matcher),
    MARKET_BTTS: (MarketID.BOTH_TEAMS_TO_SCORE, _yes_no_matcher),
    MARKET_BTTS_FIRST_HALF: (MarketID.BTTS_FIRST_HALF, _yes_no_matcher),
    MARKET_BTTS_SECOND_HALF: (MarketID.BTTS_SECOND_HALF, _yes_no_matcher),
    MARKET_OU_05: (MarketID.MATCH_GOALS, _line_matcher_for(0.5)),
    MARKET_OU_15: (MarketID.MATCH_GOALS, _line_matcher_for(1.5)),
    MARKET_OU_25: (MarketID.MATCH_GOALS, _line_matcher_for(2.5)),
    MARKET_OU_35: (MarketID.MATCH_GOALS, _line_matcher_for(3.5)),
    MARKET_FIRST_HALF_OU_05: (MarketID.FIRST_HALF_GOALS, _line_matcher_for(0.5)),
    MARKET_FIRST_HALF_OU_15: (MarketID.FIRST_HALF_GOALS, _line_matcher_for(1.5)),
    MARKET_HOME_OU_15: (MarketID.HOME_GOALS, _line_matcher_for(1.5)),
    MARKET_AWAY_OU_15: (MarketID.AWAY_GOALS, _line_matcher_for(1.5)),
    MARKET_HOME_CLEAN_SHEET: (MarketID.TEAM_CLEAN_SHEET, _yes_no_matcher),
    MARKET_AWAY_CLEAN_SHEET: (MarketID.TEAM_CLEAN_SHEET, _yes_no_matcher),
    MARKET_TEAM_TO_SCORE_FIRST: (MarketID.FIRST_GOAL, _team_to_score_first_matcher),
    MARKET_CORNERS_TOTAL_8_5: (MarketID.MATCH_CORNERS, _line_matcher_for(8.5)),
    MARKET_CORNERS_TOTAL_9_5: (MarketID.MATCH_CORNERS, _line_matcher_for(9.5)),
    MARKET_CORNERS_TOTAL_10_5: (MarketID.MATCH_CORNERS, _line_matcher_for(10.5)),
    MARKET_CORNERS_TOTAL_11_5: (MarketID.MATCH_CORNERS, _line_matcher_for(11.5)),
    MARKET_CARDS_TOTAL_3_5: (MarketID.NUMBER_OF_CARDS, _line_matcher_for(3.5)),
    MARKET_CARDS_TOTAL_4_5: (MarketID.NUMBER_OF_CARDS, _line_matcher_for(4.5)),
    MARKET_CARDS_TOTAL_5_5: (MarketID.NUMBER_OF_CARDS, _line_matcher_for(5.5)),
}


# Defaults — operator can override per-CLI run
DEFAULT_MIN_EDGE_PCT = 3.0     # Reject anything < +3% EV
DEFAULT_MAX_STAKE_PCT = 1.5    # Kelly cap at 1.5% of bankroll
DEFAULT_KELLY_FRACTION = 0.25  # ¼ Kelly (conservative sizing convention)
DEFAULT_MIN_ODD = 1.20         # avoid micro-odds (low absolute return)
DEFAULT_MAX_ODD = 8.0          # avoid lottery tickets (variance kills CLV)
# Edge values above SANITY_EDGE_CAP are usually stale-odd artifacts, NOT
# real value. With drop_extreme=True (default) they are HARD-DROPPED.
DEFAULT_SANITY_EDGE_CAP = 50.0
# Hard floor for live picks — below this density, no live emission.
DEFAULT_INFO_DENSITY_FLOOR = 0.20
# Logical-score thresholds for the EMIT/FLAG/DROP cascade.
DEFAULT_MIN_LOGICAL_SCORE_EMIT = 0.70
DEFAULT_MIN_LOGICAL_SCORE_FLAG = 0.40

# ── Day-1 wrong-side audit gates (2026-05-10) ───────────────────────────────
# Justification: reports/sportmonks_live/exploratory/05_day5_action_plan.md
#   • Tier 1.1: market `btts` was −89.6% ROI on 38 picks (98% confidence in
#     negative); flipping to "no" recovers +78.5% — for now, blacklist.
#   • Tier 1.3: `away_ou_1_5` both sides negative; `cards_total_3_5` −78%.
#   • Tier 1.2: positive-side (over/yes) on binary markets is −91% ROI on
#     105 picks while negative-side is +42% on 410. Defensive upstream ban.
#   • Tier 1.4: D9-D10 (predicted p≥0.87) overconfident by 17pp; apply
#     linear Kelly haircut until isotonic/Platt calibrator ships.
#   • Tier 1.5: 57% of Over picks emit at 0-0; gate Over@0-0 after min 30.
DEFAULT_MARKET_BLACKLIST: frozenset[str] = frozenset({
    MARKET_BTTS,             # Tier 1.1
    MARKET_AWAY_OU_15,       # Tier 1.3 — both sides lose
    MARKET_CARDS_TOTAL_3_5,  # Tier 1.3 — −78% ROI
})
DEFAULT_BAN_POSITIVE_SIDE_BINARIES = True
_BINARY_MARKETS_FOR_SIDE_BAN: frozenset[str] = frozenset({
    MARKET_OU_05, MARKET_OU_15, MARKET_OU_25, MARKET_OU_35,
    MARKET_FIRST_HALF_OU_05, MARKET_FIRST_HALF_OU_15,
    MARKET_HOME_OU_15, MARKET_AWAY_OU_15,
    MARKET_BTTS, MARKET_BTTS_FIRST_HALF, MARKET_BTTS_SECOND_HALF,
    MARKET_CARDS_TOTAL_3_5, MARKET_CARDS_TOTAL_4_5, MARKET_CARDS_TOTAL_5_5,
    MARKET_CORNERS_TOTAL_8_5, MARKET_CORNERS_TOTAL_9_5,
    MARKET_CORNERS_TOTAL_10_5, MARKET_CORNERS_TOTAL_11_5,
})
_POSITIVE_SIDE_SELECTIONS: frozenset[str] = frozenset({"over", "yes"})
DEFAULT_DROP_OVER_ZERO_ZERO = True
DEFAULT_ZERO_ZERO_MIN_MINUTE = 30
DEFAULT_HIGH_PROB_HAIRCUT_THRESHOLD = 0.85
DEFAULT_HIGH_PROB_HAIRCUT_SLOPE = 0.65
# Tier 1.6: commentary cool-off gate (Day-1 post-Tier-2 ship).
# Suppresses picks during VAR / disallowed-goal / injury / penalty /
# red-card disruption windows. See bip.evaluation.live.commentary.
DEFAULT_ENFORCE_COMMENTARY_COOLOFF = True
# Material-event blackout (seconds after last goal / red card).
MATERIAL_EVENT_BLACKOUT_SECONDS = 90
# Stale-odd tolerance after a material event: when an event landed in the
# last ~3 match-minutes, an odd must have refreshed within this window of
# the snapshot to be considered fresh.
STALE_ODD_TOLERANCE_SECONDS = 30
# Hard-freshness floor: an odd not refreshed in this many seconds is
# considered abandoned regardless of recent events. 5 minutes covers the
# common case of bookmakers sleeping on inactive markets.
STALE_ODD_FRESHNESS_SECONDS = 300
# Late-minute hard suspension threshold.
LATE_MINUTE_BLACKOUT = 89

# Critical sanity flags — these always HARD-DROP under drop_extreme.
_CRITICAL_FLAGS: tuple[str, ...] = (
    "edge_above_sanity_cap",
    "extreme_edge_no_sm_confirmation",
    "bookmaker_incoherent",
    "correct_score_disagrees",
    "sportmonks_valuebet_disagrees",
    "stale_odd",
    "post_event_blackout",
    "late_minute_blackout",
    "low_info_density",
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _market_is_for_team(market: str, selection: str, side: str) -> bool:
    """Return True iff this market+selection pick implies that ``side`` wins
    or doesn't lose.

    Used by sanity filters to drop picks that depend on a team in a
    structural disadvantage (e.g. red card with 30+ min left).
    """
    if market == MARKET_FULLTIME_RESULT:
        return selection == side
    if market == MARKET_FIRST_HALF_RESULT:
        return selection == side
    if market == MARKET_DRAW_NO_BET:
        return selection == side
    if market == MARKET_DOUBLE_CHANCE:
        if side == "home":
            return selection in ("1x", "12")
        return selection in ("x2", "12")
    if market == MARKET_HOME_OU_15:
        return side == "home" and selection == "over"
    if market == MARKET_AWAY_OU_15:
        return side == "away" and selection == "over"
    return False


# Correlation clusters for bundle-dedup (B-G5).
# Picks within the same cluster on the same fixture collapse to one.
_CLUSTER_1X2 = frozenset({
    MARKET_FULLTIME_RESULT, MARKET_DOUBLE_CHANCE, MARKET_DRAW_NO_BET,
    MARKET_FIRST_HALF_RESULT, MARKET_HTFT,
})
_CLUSTER_TOTALS = frozenset({
    MARKET_OU_05, MARKET_OU_15, MARKET_OU_25, MARKET_OU_35,
    MARKET_FIRST_HALF_OU_05, MARKET_FIRST_HALF_OU_15,
    MARKET_HOME_OU_15, MARKET_AWAY_OU_15,
})
_CLUSTER_BTTS = frozenset({
    MARKET_BTTS, MARKET_BTTS_FIRST_HALF, MARKET_BTTS_SECOND_HALF,
    MARKET_HOME_CLEAN_SHEET, MARKET_AWAY_CLEAN_SHEET,
    MARKET_TEAM_TO_SCORE_FIRST,
})
_CLUSTER_CORNERS = frozenset({
    MARKET_CORNERS_TOTAL_8_5, MARKET_CORNERS_TOTAL_9_5,
    MARKET_CORNERS_TOTAL_10_5, MARKET_CORNERS_TOTAL_11_5,
})
_CLUSTER_CARDS = frozenset({
    MARKET_CARDS_TOTAL_3_5, MARKET_CARDS_TOTAL_4_5, MARKET_CARDS_TOTAL_5_5,
})


def _cluster_of(market: str) -> str:
    if market in _CLUSTER_1X2:
        return "1x2"
    if market in _CLUSTER_TOTALS:
        return "totals"
    if market in _CLUSTER_BTTS:
        return "btts"
    if market in _CLUSTER_CORNERS:
        return "corners"
    if market in _CLUSTER_CARDS:
        return "cards"
    return market  # singleton cluster


# Bidirectional valuebet alignment: which Sportmonks-bet labels conflict
# with which (market, selection)?  A pick's selection ALIGNS with a
# Sportmonks valuebet if they refer to the same outcome; CONFLICTS if they
# refer to mutually-exclusive outcomes within the same market family.
_VB_ALIGN: dict[str, dict[str, set[str]]] = {
    # market_key → selection → set of normalised SM bet labels that ALIGN
    MARKET_FULLTIME_RESULT: {
        "home": {"home", "1"}, "draw": {"draw", "x"}, "away": {"away", "2"},
    },
    MARKET_DOUBLE_CHANCE: {
        "1x": {"1x", "home_or_draw", "draw_home"},
        "x2": {"x2", "draw_or_away", "draw_away"},
        "12": {"12", "home_or_away", "home_away"},
    },
    MARKET_BTTS: {"yes": {"yes", "btts_yes"}, "no": {"no", "btts_no"}},
}

_VB_CONFLICT: dict[str, dict[str, set[str]]] = {
    MARKET_FULLTIME_RESULT: {
        "home": {"draw", "x", "away", "2"},
        "draw": {"home", "1", "away", "2"},
        "away": {"home", "1", "draw", "x"},
    },
    MARKET_DOUBLE_CHANCE: {
        "1x": {"away", "2"}, "x2": {"home", "1"}, "12": {"draw", "x"},
    },
    MARKET_BTTS: {"yes": {"no", "btts_no"}, "no": {"yes", "btts_yes"}},
}


def _norm_vb_label(s: Any) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def _selections_align(market: str, selection: str, sm_bet_label: str) -> bool:
    aligned = _VB_ALIGN.get(market, {}).get(selection, set())
    return _norm_vb_label(sm_bet_label) in aligned


def _selections_conflict(market: str, selection: str, sm_bet_label: str) -> bool:
    conflict = _VB_CONFLICT.get(market, {}).get(selection, set())
    return _norm_vb_label(sm_bet_label) in conflict


def _sportmonks_score_grid_marginal(
    grid, market: str, selection: str, home_goals: int, away_goals: int,
) -> float | None:
    """Compute Sportmonks's implied marginal P(market.selection) from the
    correct-score grid, conditioning on (home_goals, away_goals) already
    scored.

    Returns None for markets without a grid-derivable marginal.
    """
    if grid is None:
        return None
    n = grid.shape[0]
    h_min, a_min = max(0, home_goals), max(0, away_goals)
    if h_min >= n or a_min >= n:
        return None
    # Mass over (h, a) ≥ (h_min, a_min) — i.e. final scorelines compatible
    # with the current state. Renormalise.
    sub = grid[h_min:, a_min:]
    total = float(sub.sum())
    if total <= 0:
        return None
    if market == MARKET_FULLTIME_RESULT:
        p = 0.0
        for h in range(sub.shape[0]):
            for a in range(sub.shape[1]):
                if selection == "home" and h > a:
                    p += sub[h, a]
                elif selection == "draw" and h == a:
                    p += sub[h, a]
                elif selection == "away" and h < a:
                    p += sub[h, a]
        return p / total
    if market == MARKET_DOUBLE_CHANCE:
        ftr_home = sum(sub[h, a] for h in range(sub.shape[0])
                       for a in range(sub.shape[1]) if h > a) / total
        ftr_draw = sum(sub[h, a] for h in range(sub.shape[0])
                       for a in range(sub.shape[1]) if h == a) / total
        ftr_away = 1.0 - ftr_home - ftr_draw
        return {
            "1x": ftr_home + ftr_draw,
            "x2": ftr_draw + ftr_away,
            "12": ftr_home + ftr_away,
        }.get(selection)
    if market == MARKET_BTTS:
        # Need BOTH final h>0 AND a>0 (already-scored counts toward this)
        yes_mass = 0.0
        for h_off in range(sub.shape[0]):
            for a_off in range(sub.shape[1]):
                final_h = h_min + h_off
                final_a = a_min + a_off
                if final_h > 0 and final_a > 0:
                    yes_mass += sub[h_off, a_off]
        p_yes = yes_mass / total
        return p_yes if selection == "yes" else 1.0 - p_yes
    if market in (MARKET_OU_05, MARKET_OU_15, MARKET_OU_25, MARKET_OU_35):
        line = {
            MARKET_OU_05: 0.5, MARKET_OU_15: 1.5,
            MARKET_OU_25: 2.5, MARKET_OU_35: 3.5,
        }[market]
        over_mass = 0.0
        for h_off in range(sub.shape[0]):
            for a_off in range(sub.shape[1]):
                if (h_min + h_off) + (a_min + a_off) > line:
                    over_mass += sub[h_off, a_off]
        p_over = over_mass / total
        return p_over if selection == "over" else 1.0 - p_over
    return None


# ── core dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LivePick:
    """One actionable pick — model probability strongly beats bookmaker odds."""

    fixture_id: int
    minute: int
    home_team: str
    away_team: str
    market: str           # our market key (e.g. 'first_half_result')
    selection: str        # our selection key (e.g. 'home')
    bookmaker_id: int
    bookmaker_odd: float
    our_probability: float
    fair_odd: float
    edge_pct: float       # ((odd × prob) - 1) × 100
    kelly_fraction_full: float
    suggested_stake_pct: float  # bounded fraction-Kelly of bankroll
    market_description: str | None = None
    snapshot_kind: str = "live"
    # Sanity flag — emitted but NOT auto-confirmed; operator should verify
    flagged_reason: str | None = None
    # Logical-pick composite score in [0, 1]; full breakdown in components.
    logical_score: float = 1.0
    logical_components: dict[str, float] = field(default_factory=dict)
    # Confidence half-width on our_probability (decimal, e.g. 0.04 = ±4pp)
    confidence_half_width: float = 0.0
    # Raw model probability before calibration. Equals `our_probability`
    # when no calibrator is loaded. Preserved for diagnostics and audit.
    model_probability_raw: float = 0.0


class ValueDetector:
    """Compare predictor probabilities against in-play odds; emit LivePicks."""

    def __init__(
        self,
        *,
        min_edge_pct: float = DEFAULT_MIN_EDGE_PCT,
        max_stake_pct: float = DEFAULT_MAX_STAKE_PCT,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        min_odd: float = DEFAULT_MIN_ODD,
        max_odd: float = DEFAULT_MAX_ODD,
        sanity_edge_cap: float = DEFAULT_SANITY_EDGE_CAP,
        drop_flagged: bool = False,
        drop_extreme: bool = True,
        enforce_coherence: bool = True,
        enforce_blackout: bool = True,
        enforce_stale_odd_check: bool = True,
        enforce_ci_gate: bool = True,
        info_density_floor: float = DEFAULT_INFO_DENSITY_FLOOR,
        min_logical_score_emit: float = DEFAULT_MIN_LOGICAL_SCORE_EMIT,
        min_logical_score_flag: float = DEFAULT_MIN_LOGICAL_SCORE_FLAG,
        bundle_dedup: bool = True,
        market_blacklist: frozenset[str] = DEFAULT_MARKET_BLACKLIST,
        ban_positive_side_binaries: bool = DEFAULT_BAN_POSITIVE_SIDE_BINARIES,
        drop_over_zero_zero: bool = DEFAULT_DROP_OVER_ZERO_ZERO,
        zero_zero_min_minute: int = DEFAULT_ZERO_ZERO_MIN_MINUTE,
        high_prob_haircut_threshold: float = DEFAULT_HIGH_PROB_HAIRCUT_THRESHOLD,
        high_prob_haircut_slope: float = DEFAULT_HIGH_PROB_HAIRCUT_SLOPE,
        calibrator: IsotonicProbabilityCalibrator | PerMarketCalibrator | None = None,
        enforce_commentary_cooloff: bool = DEFAULT_ENFORCE_COMMENTARY_COOLOFF,
        commentary_cooloff_minutes: dict[CommentaryEventType, int] | None = None,
    ) -> None:
        self.min_edge_pct = min_edge_pct
        self.max_stake_pct = max_stake_pct
        self.kelly_fraction = kelly_fraction
        self.min_odd = min_odd
        self.max_odd = max_odd
        self.sanity_edge_cap = sanity_edge_cap
        self.drop_flagged = drop_flagged
        self.drop_extreme = drop_extreme
        self.enforce_coherence = enforce_coherence
        self.enforce_blackout = enforce_blackout
        self.enforce_stale_odd_check = enforce_stale_odd_check
        self.enforce_ci_gate = enforce_ci_gate
        self.info_density_floor = info_density_floor
        self.min_logical_score_emit = min_logical_score_emit
        self.min_logical_score_flag = min_logical_score_flag
        self.bundle_dedup = bundle_dedup
        self.market_blacklist = market_blacklist
        self.ban_positive_side_binaries = ban_positive_side_binaries
        self.drop_over_zero_zero = drop_over_zero_zero
        self.zero_zero_min_minute = zero_zero_min_minute
        self.high_prob_haircut_threshold = high_prob_haircut_threshold
        self.high_prob_haircut_slope = high_prob_haircut_slope
        self.calibrator = calibrator
        self.enforce_commentary_cooloff = enforce_commentary_cooloff
        self.commentary_cooloff_minutes = (
            commentary_cooloff_minutes
            if commentary_cooloff_minutes is not None
            else DEFAULT_COOLOFF_MINUTES
        )

    def evaluate(
        self,
        probs: MarketProbabilities,
        odds: list[Odd],
        *,
        home_team_name: str,
        away_team_name: str,
        state: LiveMatchState | None = None,
        on_decision: Callable[..., None] | None = None,
    ) -> list[LivePick]:
        """Return a sorted list of LivePicks (highest-edge first).

        ``on_decision`` (optional): callback invoked for every gate-drop with
        full candidate context. Powers the §E falsifiability metric — without
        it, ``pick_decisions`` only records emits/flags (set by the watcher),
        and ``gate_rejection_rates()`` returns nothing useful. Signature:

            on_decision(fixture_id, minute, market, selection, decision,
                        drop_reason, bookmaker_id=None, bookmaker_odd=None,
                        our_probability=None, edge_pct=None,
                        logical_score=None, logical_components=None,
                        confidence_half_width=None,
                        informational_density=None)
        """
        # ── Helper: per-snapshot decision emitter ────────────────────────
        density = (
            state.informational_density if state is not None else None
        )

        def _emit_drop(
            *, market: str, selection: str, drop_reason: str,
            bookmaker_id: int | None = None, bookmaker_odd: float | None = None,
            our_probability: float | None = None, edge_pct: float | None = None,
            logical_score: float | None = None,
            logical_components: dict[str, float] | None = None,
            confidence_half_width: float | None = None,
        ) -> None:
            if on_decision is None:
                return
            on_decision(
                fixture_id=probs.fixture_id, minute=probs.minute,
                market=market, selection=selection,
                decision="drop", drop_reason=drop_reason,
                bookmaker_id=bookmaker_id, bookmaker_odd=bookmaker_odd,
                our_probability=our_probability, edge_pct=edge_pct,
                logical_score=logical_score,
                logical_components=logical_components,
                confidence_half_width=confidence_half_width,
                informational_density=density,
            )

        # ── Hard-floor blackouts (apply before any per-market work) ──
        if state is not None and self.enforce_blackout:
            blackout = self._global_blackout(state)
            if blackout is not None:
                logger.debug("blackout fixture=%d reason=%s", state.fixture_id, blackout)
                # Log one drop record per market×selection to attribute the
                # blackout granularly (gate_rejection_rates can then split
                # post_event_blackout vs late_minute_blackout impact).
                for market_key, market_probs in probs.by_market.items():
                    for sel in market_probs:
                        _emit_drop(
                            market=market_key, selection=sel,
                            drop_reason=blackout,
                        )
                return []
        # Information-density hard floor — kills cluster C entirely
        if (
            state is not None
            and state.is_live
            and probs.snapshot_kind == "live"
            and state.informational_density < self.info_density_floor
        ):
            for market_key, market_probs in probs.by_market.items():
                for sel in market_probs:
                    _emit_drop(
                        market=market_key, selection=sel,
                        drop_reason="low_info_density",
                    )
            return []

        odds_by_market: dict[int, list[Odd]] = defaultdict(list)
        for o in odds:
            if o.suspended or o.stopped:
                continue
            odds_by_market[o.market_id].append(o)

        # Bookmaker self-coherence — skip markets whose internal pricing
        # is inconsistent (cluster A: DC X2 == draw, OU non-monotonic).
        coherence_skip: set[str] = set()
        if self.enforce_coherence:
            coherence_skip = self._bookmaker_coherence_flags(odds_by_market)

        # Compute Sportmonks score grid once per snapshot for cross-checks
        score_grid = state.sportmonks_score_grid if state is not None else None
        last_event_minute = self._last_material_event_minute(state)

        picks: list[LivePick] = []
        for market_key, market_probs in probs.by_market.items():
            if market_key in coherence_skip:
                # Log per-selection drop to attribute the cluster-A kill.
                for sel in market_probs:
                    _emit_drop(
                        market=market_key, selection=sel,
                        drop_reason="bookmaker_incoherent",
                    )
                continue
            binding = _MARKET_BINDINGS.get(market_key)
            if not binding:
                continue
            market_id, matcher = binding
            available_odds = odds_by_market.get(market_id, [])
            if not available_odds:
                continue
            for odd in available_odds:
                sel = matcher(odd)
                if sel is None:
                    continue
                our_prob_raw = market_probs.get(sel)
                if our_prob_raw is None or our_prob_raw <= 0.0:
                    continue
                # Apply probability calibrator if loaded. Day-1 audit found
                # D9-D10 (raw p >= 0.87) are 17pp overconfident; isotonic
                # calibration corrects this without retraining the predictor.
                # When no calibrator is loaded, the Tier 1.4 Kelly haircut
                # (below) provides a coarser stopgap correction.
                if self.calibrator is not None:
                    our_prob = self.calibrator.transform(
                        our_prob_raw, market=market_key,
                    )
                    if our_prob <= 0.0:
                        continue
                else:
                    our_prob = our_prob_raw
                d = odd.decimal_odd
                if d is None or d < self.min_odd or d > self.max_odd:
                    if d is not None:
                        _emit_drop(
                            market=market_key, selection=sel,
                            bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                            our_probability=our_prob,
                            drop_reason="odd_outside_range",
                        )
                    continue

                # Stale-odd gate (C-1)
                if (
                    self.enforce_stale_odd_check
                    and state is not None
                    and self._is_stale_odd(odd, state, last_event_minute)
                ):
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob,
                        drop_reason="stale_odd",
                    )
                    continue

                # ── Day-1 wrong-side audit gates (Tier 1.1, 1.2, 1.3, 1.5) ─
                # Tier 1.1 + 1.3: market blacklist (btts, away_ou_1_5, cards_total_3_5)
                if market_key in self.market_blacklist:
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob,
                        drop_reason="market_blacklist",
                    )
                    continue
                # Tier 1.2: positive-side ban on binary markets
                if (
                    self.ban_positive_side_binaries
                    and market_key in _BINARY_MARKETS_FOR_SIDE_BAN
                    and sel in _POSITIVE_SIDE_SELECTIONS
                ):
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob,
                        drop_reason="positive_side_binary_ban",
                    )
                    continue
                # Tier 1.5: score-state 0-0 Over gate
                if (
                    self.drop_over_zero_zero
                    and sel == "over"
                    and state is not None
                    and state.home_goals == 0
                    and state.away_goals == 0
                    and state.minute >= self.zero_zero_min_minute
                ):
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob,
                        drop_reason="zero_zero_over_gate",
                    )
                    continue

                ev = our_prob * d - 1.0
                edge_pct = ev * 100.0
                if edge_pct < self.min_edge_pct:
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob, edge_pct=edge_pct,
                        drop_reason="below_min_edge",
                    )
                    continue

                # Per-pick credibility-interval gate (B-G4) — only when
                # the predictor actually published a confidence for this
                # market. Without that we have no basis to demand more.
                ci_optional = probs.confidences.get(market_key)
                if self.enforce_ci_gate and ci_optional is not None:
                    # Required edge%% to clear 2 × CI half-width
                    edge_required_pct = max(
                        self.min_edge_pct, 200.0 * ci_optional * d
                    )
                    if edge_pct < edge_required_pct:
                        _emit_drop(
                            market=market_key, selection=sel,
                            bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                            our_probability=our_prob, edge_pct=edge_pct,
                            confidence_half_width=ci_optional,
                            drop_reason="ci_gate",
                        )
                        continue
                ci = ci_optional if ci_optional is not None else 0.0

                # Kelly: f* = (bp - q) / b, where b = d-1, p = our_prob, q = 1-p
                b = d - 1.0
                if b <= 0:
                    continue
                # Diagnostic Kelly fraction (from raw our_prob, no haircut).
                f_star = (b * our_prob - (1 - our_prob)) / b
                # Tier 1.4: high-probability haircut on the STAKE only.
                # Day-1 calibration audit: D9-D10 (predicted p ≥ 0.87) are 17pp
                # overconfident; linear taper reduces Kelly size on exactly the
                # picks the model is most wrong about. Removable once isotonic
                # calibration ships.
                if our_prob > self.high_prob_haircut_threshold:
                    kelly_prob = (
                        self.high_prob_haircut_threshold
                        + self.high_prob_haircut_slope
                        * (our_prob - self.high_prob_haircut_threshold)
                    )
                    f_star_hc = (b * kelly_prob - (1 - kelly_prob)) / b
                else:
                    f_star_hc = f_star
                f_kelly = max(0.0, f_star_hc) * self.kelly_fraction
                stake_pct = min(f_kelly * 100.0, self.max_stake_pct)
                fair_odd = 1.0 / our_prob

                # ── Sanity filters (incl. VALUEBET + CORRECT_SCORE gates) ─
                flagged_reason = self._check_sanity(
                    market=market_key, selection=sel, edge_pct=edge_pct,
                    our_prob=our_prob, state=state, score_grid=score_grid,
                )

                # ── Logical-pick score (composite gate) ─────────────
                logical_score, components = self._logical_score(
                    market=market_key, selection=sel,
                    our_prob=our_prob, decimal_odd=d, edge_pct=edge_pct,
                    ci=ci, flagged_reason=flagged_reason, state=state,
                    coherence_skip=coherence_skip,
                )

                # B-G2: hard-drop critical flags under drop_extreme
                if (
                    flagged_reason
                    and self.drop_extreme
                    and any(flagged_reason.startswith(f) for f in _CRITICAL_FLAGS)
                ):
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob, edge_pct=edge_pct,
                        logical_score=logical_score,
                        logical_components=components,
                        confidence_half_width=ci,
                        # Use the critical-flag prefix as the canonical
                        # drop_reason for aggregation; details live on
                        # the full flagged_reason string elsewhere.
                        drop_reason=next(
                            (f for f in _CRITICAL_FLAGS
                             if flagged_reason.startswith(f)),
                            flagged_reason,
                        ),
                    )
                    continue

                # Logical-score cascade
                if logical_score < self.min_logical_score_flag:
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob, edge_pct=edge_pct,
                        logical_score=logical_score,
                        logical_components=components,
                        confidence_half_width=ci,
                        drop_reason="logical_score_below_flag",
                    )
                    continue   # DROP — below flag threshold
                if logical_score < self.min_logical_score_emit and not flagged_reason:
                    # Under emit-threshold but no critical flag → mark as flagged
                    flagged_reason = flagged_reason or "logical_score_below_emit"

                # Size-down for suspect EV (B-G4 / size_consistency)
                size_scale = components.get("size_consistency", 1.0)
                stake_pct = stake_pct * size_scale

                # Stake-by-CI: wider credibility-interval = less confident
                # in our_probability = smaller stake. With CI=0.02 (floor):
                # multiplier=0.92. With CI=0.10 (high disagreement): 0.60.
                # Bounded at 0.40 minimum so we still stake something on
                # qualified picks even with very wide intervals.
                if ci > 0:
                    ci_scale = max(0.40, 1.0 - 4.0 * ci)
                    stake_pct = stake_pct * ci_scale

                pick = LivePick(
                    fixture_id=probs.fixture_id,
                    minute=probs.minute,
                    home_team=home_team_name,
                    away_team=away_team_name,
                    market=market_key,
                    selection=sel,
                    bookmaker_id=odd.bookmaker_id,
                    bookmaker_odd=d,
                    our_probability=our_prob,
                    fair_odd=fair_odd,
                    edge_pct=edge_pct,
                    kelly_fraction_full=f_star,
                    suggested_stake_pct=stake_pct,
                    market_description=odd.market_description,
                    snapshot_kind=probs.snapshot_kind,
                    flagged_reason=flagged_reason,
                    logical_score=logical_score,
                    logical_components=components,
                    confidence_half_width=ci,
                    model_probability_raw=our_prob_raw,
                )

                # Drop flagged picks entirely if configured (CLI flag)
                if flagged_reason and self.drop_flagged:
                    _emit_drop(
                        market=market_key, selection=sel,
                        bookmaker_id=odd.bookmaker_id, bookmaker_odd=d,
                        our_probability=our_prob, edge_pct=edge_pct,
                        logical_score=logical_score,
                        logical_components=components,
                        confidence_half_width=ci,
                        drop_reason=f"drop_flagged:{flagged_reason}",
                    )
                    continue

                picks.append(pick)

        # Bundle dedup: collapse correlated cross-market picks
        if self.bundle_dedup and len(picks) > 1:
            picks = _bundle_dedup_picks(picks)

        # Sort: highest edge first, tie-breaker on lower variance (lower odd)
        picks.sort(key=lambda p: (-p.edge_pct, p.bookmaker_odd))
        return picks

    # ── Blackouts (C-3) ─────────────────────────────────────────────────

    def _global_blackout(self, state: LiveMatchState) -> str | None:
        """Return reason string when ALL picks should be suppressed."""
        if state.minute >= LATE_MINUTE_BLACKOUT and not state.is_half_time:
            return "late_minute_blackout"
        # Material-event blackout (~last 1 minute of in-game time)
        last_min = self._last_material_event_minute(state)
        if last_min is not None and state.minute - last_min <= 1:
            return "post_event_blackout"
        # Commentary cool-off (Tier 1.6): suppress picks within window
        # of a disruptive narrative event (VAR check, disallowed goal,
        # injury delay, penalty awarded, red card from commentary).
        if (
            self.enforce_commentary_cooloff
            and getattr(state, "commentary_events", None)
        ):
            ev = _recent_commentary_event(
                state.commentary_events,
                current_minute=state.minute,
                cooloff=self.commentary_cooloff_minutes,
            )
            if ev is not None:
                return f"commentary_cooloff:{ev.event_type.value}"
        return None

    @staticmethod
    def _last_material_event_minute(state: LiveMatchState | None) -> int | None:
        if state is None:
            return None
        last = -1
        for m, _ in state.goal_events:
            if m > last:
                last = m
        for m, _ in state.red_card_events:
            if m > last:
                last = m
        return last if last >= 0 else None

    # ── Stale-odd check (C-1) ───────────────────────────────────────────

    def _is_stale_odd(
        self,
        odd: Odd,
        state: LiveMatchState,
        last_event_minute: int | None,
    ) -> bool:
        """An odd is stale if it hasn't been refreshed recently enough
        relative to the snapshot.

        Two conditions, ANY of which marks the odd stale:

        1. **Snapshot freshness** — the odd's last update is older than
           ``STALE_ODD_FRESHNESS_SECONDS`` before the snapshot itself. A
           bookmaker that hasn't moved a price in 5+ minutes during a live
           match is signalling abandonment, regardless of events.

        2. **Post-event freshness** — when there was a recent material
           event (goal / red card within the last few match minutes),
           the odd must have been updated *after* the snapshot minus a
           short tolerance. We use ``state.snapshot_taken_at`` as the
           reference; we DO NOT convert match minutes to wall-clock
           because that mapping is broken by halftime, ET, and stoppage
           time (which can add 20+ wall-clock minutes vs match-minutes
           in a single half).

        Requires ``state.snapshot_taken_at`` and ``odd.latest_bookmaker_update``.
        When either is absent we cannot judge — returns False.
        """
        if state.snapshot_taken_at is None or odd.latest_bookmaker_update is None:
            return False
        # Sportmonks frequently emits timestamps without timezone info;
        # Pydantic parses them as tz-naive. Subtraction with our tz-aware
        # snapshot_taken_at would raise. Defensive normalisation: assume
        # UTC when naive (Sportmonks convention).
        snap = state.snapshot_taken_at
        odd_ts = odd.latest_bookmaker_update
        if snap.tzinfo is None:
            from datetime import timezone as _tz
            snap = snap.replace(tzinfo=_tz.utc)
        if odd_ts.tzinfo is None:
            from datetime import timezone as _tz
            odd_ts = odd_ts.replace(tzinfo=_tz.utc)
        odd_age_seconds = (snap - odd_ts).total_seconds()
        # Negative age = odd is from the future relative to the snapshot
        # (clock skew); treat as fresh, not stale.
        if odd_age_seconds <= 0:
            return False
        # 1. Hard freshness floor — odd not refreshed in N seconds.
        if odd_age_seconds > STALE_ODD_FRESHNESS_SECONDS:
            return True
        # 2. Post-event freshness — if a material event landed within the
        # last 3 match-minutes, the odd must have refreshed AFTER snapshot
        # minus a tight window. Match-minutes here are only used to detect
        # "recency of event" which is robust to clock distortions in the
        # short window we care about.
        if last_event_minute is not None and last_event_minute >= 0:
            minutes_since_event = max(0, state.minute - last_event_minute)
            if minutes_since_event <= 3 and odd_age_seconds > STALE_ODD_TOLERANCE_SECONDS:
                return True
        return False

    # ── Bookmaker coherence (B-G1) ──────────────────────────────────────

    def _bookmaker_coherence_flags(
        self, odds_by_market: dict[int, list[Odd]],
    ) -> set[str]:
        """Detect internally inconsistent bookmaker surfaces.

        Returns the set of OUR market keys to skip on this snapshot.
        Only fires when there is enough data to judge: complete 1X2 + DC
        for the cross-check, or ≥2 OU lines for monotonicity.

        Two distinct coherence checks need different reference frames:
        - **Cluster-A pricing pathology** (X2 odd ≈ draw odd): a RAW
          implied-prob comparison. The bookmaker is offering a price
          you'd actually pay; if X2 pays the same as draw, that's a
          structural error regardless of margin.
        - **DC monotonicity** (X2 ≥ draw, 1X ≥ home, 12 ≥ home): a FAIR
          comparison after removing margin. Bookmakers price 1X2 and DC
          with potentially different margins; raw inequalities can flip
          purely from margin asymmetry, falsely tripping the gate.
        """
        skip: set[str] = set()

        # 1X2 vs DC coherence
        ftr_implied = {
            o.label: o.implied_prob
            for o in odds_by_market.get(MarketID.FULLTIME_RESULT, [])
            if o.implied_prob is not None
        }
        dc_implied = {
            o.label: o.implied_prob
            for o in odds_by_market.get(MarketID.DOUBLE_CHANCE, [])
            if o.implied_prob is not None
        }
        # Raw implied probabilities (include bookmaker margin)
        ftr_raw = {
            "home": ftr_implied.get("Home") or ftr_implied.get("1"),
            "draw": ftr_implied.get("Draw") or ftr_implied.get("X"),
            "away": ftr_implied.get("Away") or ftr_implied.get("2"),
        }
        dc_raw = {
            "1x": dc_implied.get("Home or Draw") or dc_implied.get("1X"),
            "x2": dc_implied.get("Draw or Away") or dc_implied.get("X2"),
            "12": dc_implied.get("Home or Away") or dc_implied.get("12"),
        }
        if all(v is not None for v in ftr_raw.values()) and \
                all(v is not None for v in dc_raw.values()):
            # ── Cluster-A: raw X2 ≈ raw draw (1pp absolute) ─────────
            if abs(dc_raw["x2"] - ftr_raw["draw"]) < 0.005:
                skip.add(MARKET_DOUBLE_CHANCE)
                skip.add(MARKET_FULLTIME_RESULT)

            # ── DC monotonicity (margin-normalised) ─────────────────
            # 1X2 fair: divide by sum (target sum=1)
            # DC fair: each DC selection covers 2 outcomes, so fair sum
            # is 2; multiply by 2/sum to normalize.
            ftr_total = sum(ftr_raw.values())
            dc_total = sum(dc_raw.values())
            if ftr_total > 0 and dc_total > 0:
                ftr_fair = {k: v / ftr_total for k, v in ftr_raw.items()}
                dc_fair = {k: v * 2.0 / dc_total for k, v in dc_raw.items()}
                tol = 0.01
                if dc_fair["x2"] + tol < ftr_fair["draw"]:
                    skip.add(MARKET_DOUBLE_CHANCE)
                    skip.add(MARKET_FULLTIME_RESULT)
                if dc_fair["1x"] + tol < ftr_fair["home"]:
                    skip.add(MARKET_DOUBLE_CHANCE)
                    skip.add(MARKET_FULLTIME_RESULT)
                if dc_fair["12"] + tol < ftr_fair["home"]:
                    skip.add(MARKET_DOUBLE_CHANCE)
                    skip.add(MARKET_FULLTIME_RESULT)

        # OU monotonicity: P(over 0.5) ≥ P(over 1.5) ≥ P(over 2.5) ≥ P(over 3.5)
        overs: dict[float, float] = {}
        for o in odds_by_market.get(MarketID.MATCH_GOALS, []):
            if o.label != "Over" or not o.total or o.implied_prob is None:
                continue
            try:
                overs[float(o.total)] = o.implied_prob
            except ValueError:
                continue
        sorted_lines = sorted(overs.keys())
        for i in range(len(sorted_lines) - 1):
            if overs[sorted_lines[i]] + 0.01 < overs[sorted_lines[i + 1]]:
                # Lower line should imply at least as much "over" probability
                skip.update({MARKET_OU_05, MARKET_OU_15, MARKET_OU_25, MARKET_OU_35})
                break
        return skip

    # ── Sanity filters ──────────────────────────────────────────────────

    def _check_sanity(
        self,
        *,
        market: str, selection: str, edge_pct: float, our_prob: float,
        state: LiveMatchState | None,
        score_grid=None,
    ) -> str | None:
        """Return a reason string if the pick is suspect, else None.

        Reasons documented:
        - 'edge_above_sanity_cap': EV > 50% — almost always a stale-odd
          artifact, requires manual verification.
        - 'extreme_edge_no_sm_confirmation': EV > 100% AND Sportmonks
          valuebet did not corroborate.
        - 'sportmonks_valuebet_disagrees': Sportmonks's valuebet flag
          identifies a CONFLICTING selection on the same market family.
        - 'correct_score_disagrees': Sportmonks correct-score-grid
          marginal disagrees with our probability by > 5pp.
        - 'red_card_invalidates_pick:<side>': pick depends on the man-down
          team winning or avoiding loss with substantial time remaining.
        - 'red_card_inflates_draw': pick is on draw or "team-not-loses"
          when a team has a numerical advantage with substantial time
          left → draw probability is over-modeled.
        - 'btts_yes_late_red_card_scoreless_<side>': BTTS-yes pick after a
          red card with that team yet to score and < 30 min remaining.
        """
        if edge_pct > self.sanity_edge_cap:
            return f"edge_above_sanity_cap (>{self.sanity_edge_cap:.0f}%)"

        if state is None:
            return None

        # ── VALUEBET cross-check (B-15) ─────────────────────────────
        sm_vb = state.sportmonks_predictions.get(PredictionType.VALUEBET)
        if isinstance(sm_vb, dict):
            sm_bet_label = sm_vb.get("bet")
            sm_is_value = bool(sm_vb.get("is_value", False))
            if sm_bet_label and sm_is_value:
                if _selections_conflict(market, selection, sm_bet_label):
                    return f"sportmonks_valuebet_disagrees:{sm_bet_label}"
            # Hard gate for >100% EV picks: REQUIRE Sportmonks corroboration —
            # but ONLY for markets where Sportmonks emits valuebet labels
            # (1X2/DC/BTTS). Markets without VB coverage (corners, BTTS-2H,
            # team-clean-sheet, team-to-score-first, DNB, OUs) cannot be
            # corroborated this way; the sanity_edge_cap (50%) still drops
            # them as `edge_above_sanity_cap` before this branch.
            if edge_pct > 100.0 and market in _VB_ALIGN:
                if not (
                    sm_bet_label
                    and sm_is_value
                    and _selections_align(market, selection, sm_bet_label)
                ):
                    return "extreme_edge_no_sm_confirmation"

        # ── CORRECT_SCORE-grid cross-check (B-16) ───────────────────
        # Mixed absolute+relative tolerance. 5pp absolute is too strict for
        # low-prob markets (e.g., OU 3.5 with sm=0.20 vs ours=0.30 = 50%
        # relative error but only 10pp absolute — flag-worthy). It's also
        # too loose for high-prob markets (sm=0.95 vs ours=0.88 = 7pp
        # absolute, only ~7% relative — could be normal noise). Use:
        #   tolerance = max(0.05, 0.30 × min(sm_marginal, our_prob))
        # — at low probs the relative term dominates; at high probs the
        # 5pp absolute floor prevents over-flagging.
        if score_grid is not None:
            sm_marginal = _sportmonks_score_grid_marginal(
                score_grid, market, selection,
                state.home_goals, state.away_goals,
            )
            if sm_marginal is not None:
                tolerance = max(0.05, 0.30 * min(sm_marginal, our_prob))
                if abs(sm_marginal - our_prob) > tolerance:
                    return (
                        f"correct_score_disagrees:"
                        f"sm={sm_marginal:.3f}_vs_ours={our_prob:.3f}"
                    )

        remaining = max(0, 90 - state.minute) if state.is_live else 0

        # ── Red-card filters — apply when ≥20 min remain ──────────────
        if remaining >= 20:
            for side, has_red in (
                ("home", state.has_red_card_home),
                ("away", state.has_red_card_away),
            ):
                if not has_red:
                    continue
                if _market_is_for_team(market, selection, side):
                    return f"red_card_invalidates_pick:{side}"

            if state.has_red_card_home != state.has_red_card_away:
                if market == MARKET_FULLTIME_RESULT and selection == "draw":
                    return "red_card_inflates_draw"
                if market == MARKET_FIRST_HALF_RESULT and selection == "draw":
                    return "red_card_inflates_draw"

        # ── BTTS-yes sanity ──────────────────────────────────────────
        if market == MARKET_BTTS and selection == "yes" and remaining < 30:
            scoreless_home = state.home_goals == 0
            scoreless_away = state.away_goals == 0
            if scoreless_home and state.has_red_card_home:
                return "btts_yes_late_red_card_scoreless_home"
            if scoreless_away and state.has_red_card_away:
                return "btts_yes_late_red_card_scoreless_away"

        # ── Over-goals sanity in late game ───────────────────────────
        if remaining < 10 and market.startswith("ou_") and selection == "over":
            line_str = market[3:]  # e.g., '2_5'
            try:
                line_int = int(line_str.split("_")[0])
                line_dec = int(line_str.split("_")[1])
                line = line_int + line_dec / 10.0
            except (IndexError, ValueError):
                line = None
            if line is not None:
                total = state.home_goals + state.away_goals
                goals_needed = math.ceil(line) - total
                if goals_needed >= 2:
                    return "over_late_game_unrealistic"

        return None

    # ── Logical-pick composite score (§D) ───────────────────────────────

    def _logical_score(
        self,
        *,
        market: str, selection: str,
        our_prob: float, decimal_odd: float, edge_pct: float, ci: float,
        flagged_reason: str | None,
        state: LiveMatchState | None,
        coherence_skip: set[str],
    ) -> tuple[float, dict[str, float]]:
        """Compute the composite logical-pick score in [0, 1].

        cascade-with-mins: ``score = min(consilience, odds_credibility,
        liquidity, informational_content, size_consistency)``. Any single
        zero subscore zeros the score — every property is necessary.

        Returns (score, components) where components hold the individual
        subscores for audit logging.
        """
        # ── Consilience ─────────────────────────────────────────────
        # ONLY independent prediction sources (Sportmonks CORRECT_SCORE
        # grid + VALUEBET) contribute. Density is its own dimension
        # (c_info) and must not be double-counted here — that bug
        # previously dragged consilience down whenever density was low,
        # even with both SM cross-checks agreeing perfectly.
        consilience_terms: list[float] = []
        if state is not None:
            grid = state.sportmonks_score_grid
            if grid is not None:
                sm_marginal = _sportmonks_score_grid_marginal(
                    grid, market, selection,
                    state.home_goals, state.away_goals,
                )
                if sm_marginal is not None:
                    consilience_terms.append(_agree(sm_marginal, our_prob))
            sm_vb = state.sportmonks_predictions.get(PredictionType.VALUEBET)
            if isinstance(sm_vb, dict) and sm_vb.get("is_value"):
                lbl = sm_vb.get("bet")
                if _selections_align(market, selection, lbl):
                    consilience_terms.append(1.0)
                elif _selections_conflict(market, selection, lbl):
                    consilience_terms.append(0.0)

        if state is None:
            # Test convenience — no cross-check infrastructure available.
            # Don't penalize unit tests that fixture state=None.
            c_consilience = 0.85
        elif consilience_terms:
            c_consilience = sum(consilience_terms) / len(consilience_terms)
        else:
            # State present but Sportmonks emitted no usable cross-check
            # sources (no CORRECT_SCORE grid + no value-flagged bet) for
            # this market. Default below emit threshold (0.70) so picks
            # without external corroboration get FLAGGED unless other
            # dimensions are strong — but above flag threshold (0.40) so
            # they are not auto-dropped (cross-checks can be missing for
            # legitimate reasons in lower-tier leagues).
            c_consilience = 0.65

        # ── Odds credibility ────────────────────────────────────────
        ev_ratio = our_prob * decimal_odd
        if ev_ratio < 1.03:
            c_odds = 0.0
        elif ev_ratio <= 1.20:
            c_odds = 1.0
        elif ev_ratio <= 1.50:
            c_odds = 1.0 - (ev_ratio - 1.20) / 0.30
        else:
            c_odds = 0.0   # +50% EV is cluster-B territory

        # ── Liquidity ──────────────────────────────────────────────
        c_liquidity = 1.0
        if market in coherence_skip:
            c_liquidity = 0.0
        elif state is not None:
            if state.minute < 5 or state.minute >= LATE_MINUTE_BLACKOUT:
                c_liquidity = 0.0
            elif flagged_reason and any(
                flagged_reason.startswith(f) for f in (
                    "red_card_invalidates_pick",
                    "red_card_inflates_draw",
                    "btts_yes_late_red_card_scoreless",
                    "over_late_game_unrealistic",
                    "stale_odd",
                )
            ):
                c_liquidity = 0.0

        # ── Informational content ───────────────────────────────────
        if state is None or not state.is_live:
            c_info = 1.0
        else:
            c_info = state.informational_density

        # ── Size consistency ────────────────────────────────────────
        edge_dec = ev_ratio - 1
        if edge_dec <= 0.20:
            c_size = 1.0
        elif edge_dec <= 0.40:
            c_size = 0.5
        elif edge_dec <= 0.80:
            c_size = 0.20
        else:
            c_size = 0.0

        components = {
            "consilience": c_consilience,
            "odds_credibility": c_odds,
            "liquidity": c_liquidity,
            "informational_content": c_info,
            "size_consistency": c_size,
        }
        # size_consistency is a STAKE MODULATOR, not an emit/drop gate.
        # Including it in the cascade min() created a dead-code path:
        # extreme-edge picks with c_size=0.20 fell below min_logical_score_flag
        # and were dropped before the stake-down at line 567 could fire.
        # Filter dimensions are everything except size_consistency.
        filter_score = min(
            v for k, v in components.items() if k != "size_consistency"
        )
        return filter_score, components


def _agree(source_p: float, our_p: float, tol: float = 0.05) -> float:
    """Return [0, 1] agreement between a source probability and ours.

    1.0 when within tol (default 5pp), decays linearly to 0 at 2 × tol,
    floors at 0.
    """
    delta = abs(source_p - our_p)
    if delta <= tol:
        return 1.0
    if delta >= 2 * tol:
        return 0.0
    return 1.0 - (delta - tol) / tol


def _bundle_dedup_picks(picks: list[LivePick]) -> list[LivePick]:
    """Collapse correlated cross-market picks per fixture (B-G5 + #5b).

    Two-pass strategy:

    Pass 1 — INTRA-cluster dedup. Picks in the same correlation cluster
    (1x2 / totals / btts / corners / cards) collapse to one survivor by
    ``(clean_flag, confidence_weighted_edge)`` lexicographic ranking.

    Pass 2 — CROSS-cluster correlation. Pairs of survivors from different
    clusters can still encode the SAME underlying directional bet
    (e.g., DC 1X + OU 2.5 under = "low-scoring home-not-loses"; BTTS no +
    OU 2.5 under = "low-scoring shutout"). When the operator places both
    they're effectively double-staking the same view. We collapse cross-
    cluster correlated pairs to the higher-quality one using the same
    ranking. The correlation matrix is intentionally conservative — only
    pairs with clear directional overlap are flagged.

    Different fixtures are always independent.
    """
    by_fixture: dict[int, list[LivePick]] = defaultdict(list)
    for p in picks:
        by_fixture[p.fixture_id].append(p)

    def _rank(pick: LivePick) -> tuple[int, float]:
        clean_flag = 1 if pick.flagged_reason is None else 0
        confidence_weighted_edge = pick.edge_pct * (
            1.0 - pick.our_probability * (1 - pick.our_probability)
        )
        return (clean_flag, confidence_weighted_edge)

    out: list[LivePick] = []
    for fix_picks in by_fixture.values():
        # ── Pass 1: intra-cluster dedup
        clusters: dict[str, list[LivePick]] = defaultdict(list)
        for p in fix_picks:
            clusters[_cluster_of(p.market)].append(p)
        survivors: list[LivePick] = []
        for cluster_picks in clusters.values():
            if len(cluster_picks) == 1:
                survivors.extend(cluster_picks)
                continue
            survivors.append(max(cluster_picks, key=_rank))

        # ── Pass 2: cross-cluster correlation collapse
        # Iteratively eliminate the WORSE of any correlated pair until
        # no more pairs remain. O(n²) but n ≤ 5 in practice.
        eliminated: set[int] = set()  # indices into ``survivors``
        for i in range(len(survivors)):
            if i in eliminated:
                continue
            for j in range(i + 1, len(survivors)):
                if j in eliminated:
                    continue
                if _picks_strongly_correlated(survivors[i], survivors[j]):
                    # Keep the higher-ranked, eliminate the other
                    if _rank(survivors[i]) >= _rank(survivors[j]):
                        eliminated.add(j)
                    else:
                        eliminated.add(i)
                        break  # i was eliminated, no more j-comparisons
        out.extend(p for k, p in enumerate(survivors) if k not in eliminated)
    return out


# Cross-cluster correlation pairs — intentionally conservative. Each entry
# is a frozenset of (market, selection) pairs; if two picks both match
# any pair in this set, they're treated as the same directional bet.
_CORRELATED_PAIRS: tuple[frozenset[tuple[str, str]], ...] = (
    # "Low-scoring + home not-loses" view
    frozenset({(MARKET_DOUBLE_CHANCE, "1x"), (MARKET_OU_25, "under")}),
    frozenset({(MARKET_DOUBLE_CHANCE, "1x"), (MARKET_OU_15, "under")}),
    # "Low-scoring + away not-loses" view
    frozenset({(MARKET_DOUBLE_CHANCE, "x2"), (MARKET_OU_25, "under")}),
    frozenset({(MARKET_DOUBLE_CHANCE, "x2"), (MARKET_OU_15, "under")}),
    # "Low-scoring shutout" view (BTTS no + low totals all encode same)
    frozenset({(MARKET_BTTS, "no"), (MARKET_OU_25, "under")}),
    frozenset({(MARKET_BTTS, "no"), (MARKET_OU_15, "under")}),
    # "Home wins comfortably" — FT home + DC 1X is double-encoded already
    # (collapsed in pass 1 — same cluster). But FT home + Home OU 1.5 over
    # are both bullish-home views from different angles.
    frozenset({(MARKET_FULLTIME_RESULT, "home"), (MARKET_HOME_OU_15, "over")}),
    frozenset({(MARKET_FULLTIME_RESULT, "away"), (MARKET_AWAY_OU_15, "over")}),
    # Clean sheet + opp side fails-to-score (opposite framing, same bet)
    frozenset({(MARKET_HOME_CLEAN_SHEET, "yes"), (MARKET_BTTS, "no")}),
    frozenset({(MARKET_AWAY_CLEAN_SHEET, "yes"), (MARKET_BTTS, "no")}),
)


def _picks_strongly_correlated(a: LivePick, b: LivePick) -> bool:
    """Return True if the two picks encode the same directional view.

    Used by bundle_dedup to collapse cross-cluster correlated pairs.
    Symmetrical: order of a/b doesn't matter.
    """
    pair = frozenset({(a.market, a.selection), (b.market, b.selection)})
    return any(pair == cp for cp in _CORRELATED_PAIRS)


# ── Best-stack deployment factory ───────────────────────────────────────────


def make_best_stack_detector(
    *,
    calibrator_path: str | None = None,
    use_per_market: bool = True,
    **overrides: Any,
) -> ValueDetector:
    """Build a ValueDetector with the Day-1 best-stack configuration.

    Encapsulates the deployment recommendation from CV honesty pass
    (commit d08ebcd / report 11_cv_validation.md):

    - All Tier 1 gates ON (blacklist, positive-side ban, 0-0 over, etc.)
    - Per-market calibrator loaded when `calibrator_path` provided
    - Tier 1.4 high-prob haircut auto-disabled when calibrator is present
      (the calibrator subsumes it; running both would double-correct)
    - Commentary cool-off gate ON
    - Logical-score / CI / coherence / blackout / stale-odd gates ON

    Arguments:
        calibrator_path: Path to per-market or global calibrator JSON.
            When None, no calibration applied (Tier 1.4 haircut stays on
            as the only overconfidence corrector).
        use_per_market: When True (default), expects a per-market JSON;
            falls back to global if the JSON `type` field says 'isotonic'.
        **overrides: Any ValueDetector kwarg to override the defaults.

    Returns:
        A configured ValueDetector ready for production.

    Example:
        >>> from bip.evaluation.live.value_detector import make_best_stack_detector
        >>> det = make_best_stack_detector(
        ...     calibrator_path="data/calibration/per_market_v1.json",
        ... )
    """
    calibrator: IsotonicProbabilityCalibrator | PerMarketCalibrator | None = None
    if calibrator_path is not None:
        try:
            if use_per_market:
                calibrator = PerMarketCalibrator.load_json(calibrator_path)
            else:
                calibrator = IsotonicProbabilityCalibrator.load_json(
                    calibrator_path
                )
        except ValueError:
            # JSON is of the other type — fall back gracefully.
            try:
                calibrator = IsotonicProbabilityCalibrator.load_json(
                    calibrator_path
                )
            except ValueError:
                calibrator = PerMarketCalibrator.load_json(calibrator_path)

    kwargs: dict[str, Any] = {
        "calibrator": calibrator,
    }
    # When calibrator is loaded, disable Tier 1.4 haircut (subsumed).
    if calibrator is not None:
        kwargs["high_prob_haircut_threshold"] = 1.01
    kwargs.update(overrides)
    return ValueDetector(**kwargs)
