"""WC2026 avoid rules — bookings to block, not place.

C-series functions take a candidate pick + context and return an
``AvoidVerdict`` saying whether the pick should be SUPPRESSED.

Use case: after `value_detector` or the A-series rules emit a
candidate, run it through ``run_avoids(...)``. If any C-rule blocks
it, drop the pick.

Rules:
    C1  LOCK_SOFT_FAVORITE_AH    block AH≤-1.0 on favorites where lock_H ≤ 0.45
    C2  TM_EXTREME_BIG_SPREAD    block AH≤-2.5 on Spain/France/Germany-tier
    C3  CINDERELLA_ML            block ML on debutants/Cinderella teams
    C4  SENTIMENT_LINE_MOVE      hook for "line moved >5% by sentiment news"
                                  (requires line-history feed; currently
                                  manual-flag interface)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_bias_flags import (
    LockPrediction,
    WC_DEBUTANT_OR_WEAK,
)
from bip.evaluation.live.engine_v3.wc2026_patterns import (
    MatchContext,
    OddsSnapshot,
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class AvoidVerdict(BaseModel):
    """Output of one C-rule. ``blocked=True`` means drop the pick."""

    model_config = _MODEL_CONFIG
    rule_id: str
    blocked: bool
    reason: str


# ── C1 — Lock-soft favorite, asymmetric AH spread ────────────────────────

C1_LOCK_SOFT_MAX_H = 0.45
# Asian-handicap selection naming convention used in the dossiers:
#   "ah_-1.5_home", "ah_-2.0_away", etc. We block lines ≤ -1.0 (i.e. the
#   favorite must win by 2+ goals).
C1_BLOCK_AH_VALUE_MAX = -1.0


def _parse_ah_value(selection: str) -> float | None:
    """Returns the handicap value from selection strings like
    'ah_-1.5_home' or 'ah_-2_home'. Returns None when unparseable."""
    parts = selection.split("_")
    for token in parts:
        try:
            if token.startswith("-") or token.startswith("+"):
                return float(token)
        except ValueError:
            continue
    return None


def avoid_c1_lock_soft_ah(
    ctx: MatchContext,
    odds: OddsSnapshot,
    lock: LockPrediction,
) -> AvoidVerdict:
    """Block asymmetric AH (≤ -1.0) on the favorite when lock has H ≤ 0.45.

    Rationale: a lock that gives the favorite only ~40% to win is already
    conservative. Stacking an asymmetric -1.5/-2 spread on top compounds
    risk against a tail outcome the model already flags as plausible.
    """
    if odds.market_family != MarketFamily.ASIAN_HANDICAP:
        return AvoidVerdict(
            rule_id="C1", blocked=False,
            reason=f"market_family={odds.market_family} not asian_handicap",
        )

    ah_value = _parse_ah_value(odds.selection)
    if ah_value is None or ah_value > C1_BLOCK_AH_VALUE_MAX:
        return AvoidVerdict(
            rule_id="C1", blocked=False,
            reason=f"selection {odds.selection!r} not an AH ≤ -1.0",
        )

    # Determine which side has the favorite (per market value).
    if ctx.home_market_value_eur is None or ctx.away_market_value_eur is None:
        return AvoidVerdict(
            rule_id="C1", blocked=False,
            reason="no market-value data to identify favorite",
        )
    home_is_fav = ctx.home_market_value_eur > ctx.away_market_value_eur

    # selection ending '_home' or '_away' decides which team takes the AH.
    take_home = odds.selection.endswith("_home")
    take_away = odds.selection.endswith("_away")
    if not (take_home or take_away):
        return AvoidVerdict(
            rule_id="C1", blocked=False,
            reason=f"selection {odds.selection!r} side suffix not recognized",
        )

    fav_taking_ah = (home_is_fav and take_home) or ((not home_is_fav) and take_away)
    if not fav_taking_ah:
        return AvoidVerdict(
            rule_id="C1", blocked=False,
            reason="AH is on the underdog side, not blocked",
        )

    fav_lock_prob = lock.p_home_win if home_is_fav else lock.p_away_win
    if fav_lock_prob > C1_LOCK_SOFT_MAX_H:
        return AvoidVerdict(
            rule_id="C1", blocked=False,
            reason=f"lock favorite_prob {fav_lock_prob:.2f} > {C1_LOCK_SOFT_MAX_H}",
        )

    return AvoidVerdict(
        rule_id="C1",
        blocked=True,
        reason=(
            f"lock favorite_prob {fav_lock_prob:.2f} ≤ {C1_LOCK_SOFT_MAX_H} + "
            f"asymmetric AH {ah_value} stacks tail risk"
        ),
    )


# ── C2 — Extreme TM gap, oversized AH spread ─────────────────────────────

C2_TM_EXTREME_RATIO = 15.0
C2_BLOCK_AH_VALUE_MAX = -2.5


def avoid_c2_tm_extreme_oversized_ah(
    ctx: MatchContext,
    odds: OddsSnapshot,
) -> AvoidVerdict:
    """Block AH ≤ -2.5 on extreme TM mismatches (≥ 15x).

    Rationale: at TM ratios this lopsided the market over-prices the
    spread. Public hammers Spain -2.5 / France -3 lines; expected value
    erodes. Prefer derivatives (TT, corners, CS).
    """
    if odds.market_family != MarketFamily.ASIAN_HANDICAP:
        return AvoidVerdict(
            rule_id="C2", blocked=False,
            reason=f"market_family={odds.market_family} not asian_handicap",
        )
    ah_value = _parse_ah_value(odds.selection)
    if ah_value is None or ah_value > C2_BLOCK_AH_VALUE_MAX:
        return AvoidVerdict(
            rule_id="C2", blocked=False,
            reason=f"selection {odds.selection!r} not an AH ≤ {C2_BLOCK_AH_VALUE_MAX}",
        )
    if ctx.home_market_value_eur is None or ctx.away_market_value_eur is None:
        return AvoidVerdict(
            rule_id="C2", blocked=False,
            reason="no market-value data",
        )
    h, a = ctx.home_market_value_eur, ctx.away_market_value_eur
    if h <= 0 or a <= 0:
        return AvoidVerdict(
            rule_id="C2", blocked=False, reason="zero market value",
        )
    ratio = max(h, a) / min(h, a)
    if ratio < C2_TM_EXTREME_RATIO:
        return AvoidVerdict(
            rule_id="C2", blocked=False,
            reason=f"TM ratio {ratio:.2f} < {C2_TM_EXTREME_RATIO}",
        )
    return AvoidVerdict(
        rule_id="C2",
        blocked=True,
        reason=(
            f"TM ratio {ratio:.2f} ≥ {C2_TM_EXTREME_RATIO} + AH {ah_value} "
            "is a public trap; prefer derivatives"
        ),
    )


# ── C3 — Cinderella ML ───────────────────────────────────────────────────

C3_CINDERELLA_ML_MAX_ODDS = 2.60


def avoid_c3_cinderella_ml(
    ctx: MatchContext,
    odds: OddsSnapshot,
) -> AvoidVerdict:
    """Block moneyline picks on Cinderella / debutant teams at short prices.

    Rationale: sentiment-driven cuotas. After a single friendly upset
    (NZL 4-1 ENG), public hammers the underdog ML, shortening it well
    below true probability.
    """
    if odds.market_family != MarketFamily.RESULT_1X2:
        return AvoidVerdict(
            rule_id="C3", blocked=False,
            reason=f"market_family={odds.market_family} not result_1x2",
        )
    take_home = odds.selection == "home"
    take_away = odds.selection == "away"
    if not (take_home or take_away):
        return AvoidVerdict(
            rule_id="C3", blocked=False,
            reason=f"selection {odds.selection!r} not a moneyline side",
        )
    backed_team = ctx.home_team if take_home else ctx.away_team
    if backed_team not in WC_DEBUTANT_OR_WEAK:
        return AvoidVerdict(
            rule_id="C3", blocked=False,
            reason=f"backed team {backed_team!r} not Cinderella",
        )
    if odds.decimal_odds > C3_CINDERELLA_ML_MAX_ODDS:
        return AvoidVerdict(
            rule_id="C3", blocked=False,
            reason=f"odds {odds.decimal_odds:.2f} > {C3_CINDERELLA_ML_MAX_ODDS} (price is honest)",
        )
    return AvoidVerdict(
        rule_id="C3",
        blocked=True,
        reason=(
            f"backed team {backed_team} is Cinderella/debutant; "
            f"ML at {odds.decimal_odds:.2f} ≤ {C3_CINDERELLA_ML_MAX_ODDS} is a sentiment trap"
        ),
    )


# ── C4 — Sentiment-driven line move (manual hook) ────────────────────────


@dataclass(frozen=True)
class LineMoveContext:
    """Optional input for C4. Caller supplies the % move since open and
    whether the move was tied to a sentimental news event."""

    pct_move_since_open: float  # negative = shortening, positive = drifting
    sentiment_event_tagged: bool


C4_BLOCK_PCT_THRESHOLD = 5.0  # percentage points


def avoid_c4_sentiment_line_move(
    line_ctx: LineMoveContext | None,
) -> AvoidVerdict:
    """Block when the cuota moved ≥5% in 24h tied to a sentiment news event.

    Returns a non-block ``blocked=False`` when ``line_ctx`` is None — the
    upstream line-history feed is not wired yet, so this is opt-in.
    """
    if line_ctx is None:
        return AvoidVerdict(
            rule_id="C4", blocked=False,
            reason="line-history context not provided (feed not wired)",
        )
    if not line_ctx.sentiment_event_tagged:
        return AvoidVerdict(
            rule_id="C4", blocked=False,
            reason="line move not tagged to sentiment event",
        )
    if abs(line_ctx.pct_move_since_open) < C4_BLOCK_PCT_THRESHOLD:
        return AvoidVerdict(
            rule_id="C4", blocked=False,
            reason=(
                f"move |{line_ctx.pct_move_since_open:.1f}%| < "
                f"{C4_BLOCK_PCT_THRESHOLD}% threshold"
            ),
        )
    return AvoidVerdict(
        rule_id="C4",
        blocked=True,
        reason=(
            f"line moved {line_ctx.pct_move_since_open:+.1f}% on sentiment event; "
            "value likely already absorbed"
        ),
    )


# ── Orchestrator ─────────────────────────────────────────────────────────


def run_avoids(
    ctx: MatchContext,
    odds: OddsSnapshot,
    lock: LockPrediction | None = None,
    *,
    line_move: LineMoveContext | None = None,
) -> list[AvoidVerdict]:
    """Run all C-rules; returns the list of verdicts (one per rule).

    Caller can inspect any ``blocked=True`` and reject the pick. Rules
    that need missing inputs return ``blocked=False`` with a reason.
    """
    results: list[AvoidVerdict] = []
    if lock is not None:
        results.append(avoid_c1_lock_soft_ah(ctx, odds, lock))
    else:
        results.append(AvoidVerdict(
            rule_id="C1", blocked=False, reason="no lock prediction provided",
        ))
    results.append(avoid_c2_tm_extreme_oversized_ah(ctx, odds))
    results.append(avoid_c3_cinderella_ml(ctx, odds))
    results.append(avoid_c4_sentiment_line_move(line_move))
    return results


def is_blocked(verdicts: list[AvoidVerdict]) -> bool:
    """Convenience: True if any verdict blocks."""
    return any(v.blocked for v in verdicts)


__all__ = [
    "AvoidVerdict",
    "C1_BLOCK_AH_VALUE_MAX",
    "C1_LOCK_SOFT_MAX_H",
    "C2_BLOCK_AH_VALUE_MAX",
    "C2_TM_EXTREME_RATIO",
    "C3_CINDERELLA_ML_MAX_ODDS",
    "C4_BLOCK_PCT_THRESHOLD",
    "LineMoveContext",
    "avoid_c1_lock_soft_ah",
    "avoid_c2_tm_extreme_oversized_ah",
    "avoid_c3_cinderella_ml",
    "avoid_c4_sentiment_line_move",
    "is_blocked",
    "run_avoids",
]
