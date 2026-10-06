"""Market Expression Score (MES) — section 2.3 of the v3 engine design notes.

::

    MES(thesis, market, gsv) =
          base_edge(thesis, market, gsv)
        × signal_clarity(thesis, market)
        × book_slowness(market, recent_event)
        × liquidity_score(market, book)
        ÷ conditional_variance(market, gsv, horizon)

The non-obvious factor is ``signal_clarity``: how directly does the
market's outcome measure the mechanism the thesis describes? Example:
if the thesis is "leader presses with numerical superiority", the
mechanism is *sustained pressure*. That is measured **more directly**
in corners-2H (corners = entries to the last third without conversion)
than in Over 2.5 goals (which requires both pressure AND conversion).
Same thesis → corners has higher signal_clarity → higher MES.

The MES is a routing score, not a probability. Section 6 rule #5 uses
it as a threshold (``MES < 0.6 → no pick``) to enforce the principle
"valid thesis + no good market expression = no bet". That is what
prevents the system from defaulting back to the 3 efficient markets it
calibrates well — the failure mode that the v3 design is built to fix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bip.evaluation.live.engine_v3.gsv import GameStateVector, MarketLine
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis, ThesisArchetype

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.calibrator import IsotonicCalibrator

# ──────────────────────────────────────────────────────────────────────
# Signal-clarity matrix: (archetype × market_family) → clarity in [0, 1]
# ──────────────────────────────────────────────────────────────────────
#
# Values come from the sec 4-bis "Mercados óptimos (top-3)" column. The
# top market for an archetype gets 1.00; the second-best 0.85; third
# 0.70; anything off the top-3 list 0.40. We DO NOT score zero — even
# off-archetype markets can sometimes express the thesis indirectly,
# but the threshold (rule #5 in sec 6) is 0.60, so 0.40 effectively
# means "do not route here unless backed up by another archetype".
#
# This is the single highest-impact tunable surface in v3. Calibration
# strategy is documented in Phase 2 of sec 10.

_SIGNAL_CLARITY: dict[ThesisArchetype, dict[MarketFamily, float]] = {
    ThesisArchetype.RED_CARD_AWAY_EARLY: {
        MarketFamily.CORNERS: 1.00,
        MarketFamily.CARDS: 0.85,
        MarketFamily.GOALS: 0.70,
    },
    ThesisArchetype.DOMINANT_LOSING_NAPOLI: {
        MarketFamily.NEXT_GOAL: 1.00,
        MarketFamily.BTTS: 0.85,
        MarketFamily.GOALS: 0.70,
    },
    ThesisArchetype.LATE_CAGEY_ZERO_ZERO: {
        MarketFamily.GOALS: 1.00,
        MarketFamily.NEXT_GOAL: 0.85,
        MarketFamily.CORNERS: 0.70,
    },
    ThesisArchetype.LEAD_TWO_DEFENSIVE_SUB: {
        MarketFamily.CARDS: 1.00,
        MarketFamily.CORNERS: 0.85,
        MarketFamily.GOALS: 0.70,
    },
    ThesisArchetype.REGRESSION_TO_XG: {
        MarketFamily.NEXT_GOAL: 1.00,
        MarketFamily.ASIAN_HANDICAP: 0.85,
        MarketFamily.BTTS: 0.70,
    },
    ThesisArchetype.CARDS_MOMENTUM_STRICT_REF: {
        MarketFamily.CARDS: 1.00,
        MarketFamily.PROPS: 0.85,
        MarketFamily.RESULT_1X2: 0.40,
    },
    ThesisArchetype.UNDERDOG_LEADS_SIEGE: {
        MarketFamily.CARDS: 1.00,
        MarketFamily.CORNERS: 0.85,
        MarketFamily.GOALS: 0.70,
    },
    ThesisArchetype.OPEN_GAME_FORMATIONS: {
        MarketFamily.GOALS: 1.00,
        MarketFamily.BTTS: 0.85,
        MarketFamily.NEXT_GOAL: 0.70,
    },
    ThesisArchetype.KEY_PLAYMAKER_OFF: {
        MarketFamily.PROPS: 1.00,
        MarketFamily.GOALS: 0.85,
        MarketFamily.NEXT_GOAL: 0.70,
    },
    ThesisArchetype.SECOND_HALF_RESET: {
        MarketFamily.NEXT_GOAL: 1.00,
        MarketFamily.GOALS: 0.85,
        MarketFamily.CORNERS: 0.70,
    },
    ThesisArchetype.NUMERICAL_SUSTAINED: {
        MarketFamily.CORNERS: 1.00,
        MarketFamily.CARDS: 0.85,
        MarketFamily.GOALS: 0.70,
    },
    ThesisArchetype.CRUISE_MODE: {
        MarketFamily.GOALS: 1.00,
        MarketFamily.BTTS: 0.85,
        MarketFamily.NEXT_GOAL: 0.70,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Book slowness — empirical lag bands per market_family from sec 2.4
# ──────────────────────────────────────────────────────────────────────
#
# These are HYPOTHESES (sec 2.4 explicitly labels them such) that will
# get measured against recorded line snapshots once Phase 1 has 2-4
# weeks of data. For now they encode the operator's domain knowledge
# of which markets the book updates fastest vs slowest.
#
# Returns: a multiplier in [0.5, 1.5]. 1.0 = neutral. > 1 = the book is
# expected to lag here (good for us); < 1 = book is fast (bad for us).

_BOOK_SLOWNESS_BY_FAMILY: dict[MarketFamily, float] = {
    MarketFamily.RESULT_1X2: 0.55,         # 1X2 lags 5-15s — book is fast
    MarketFamily.DOUBLE_CHANCE: 0.60,
    MarketFamily.GOALS: 0.80,
    MarketFamily.BTTS: 0.85,
    MarketFamily.ASIAN_HANDICAP: 0.85,
    MarketFamily.NEXT_GOAL: 1.20,          # 30-90s lag
    MarketFamily.NEXT_CORNER: 1.30,
    MarketFamily.CORNERS: 1.40,            # 60-180s post-event lag
    MarketFamily.CARDS: 1.45,              # slowest non-prop family
    MarketFamily.PROPS: 1.50,              # heavily manual on Betano
    MarketFamily.RACE_TO_X: 1.30,
}


def book_slowness(family: MarketFamily, recent_event_age_sec: float | None) -> float:
    """Multiplier capturing book-update lag.

    If there has been NO recent critical event, slowness collapses to 1.0
    (no fresh mispricing to chase regardless of family). If the event
    is too old (>300s), the book has had time to catch up — the slowness
    advantage decays linearly toward 1.0.
    """
    if recent_event_age_sec is None:
        return 1.0
    if recent_event_age_sec > 300:
        return 1.0
    base = _BOOK_SLOWNESS_BY_FAMILY.get(family, 1.0)
    # Decay from base at age 0 to 1.0 at age 300 (linear).
    decay = recent_event_age_sec / 300.0
    return base + (1.0 - base) * decay


# ──────────────────────────────────────────────────────────────────────
# Conditional variance — predictive uncertainty over the horizon
# ──────────────────────────────────────────────────────────────────────


def _squashed_goals_cvar(raw: float) -> float:
    """Squashed GOALS conditional variance: raw / (1 + raw).

    This is a monotone squash that maps [0, ∞) → [0, 1). Used ONLY
    inside rule_8's shadow comparison for half-time GOALS theses to
    check whether the squashed value would pass the uncertainty band.

    The raw formula (λ_total × horizon / 90) is structurally high for
    normal half-time horizons (~40 min): a λ=2.5 total with 40 min
    remaining gives raw ≈ 1.11, squashed ≈ 0.53. The squash reflects
    that the remaining horizon matters less near HT — the first-half
    info has already resolved much of the uncertainty.

    DO NOT use this as the return value of conditional_variance —
    that would silently change MES scoring for all GOALS picks.
    This helper exists ONLY for the rule_8 shadow branch.
    """
    return raw / (1.0 + raw)


def conditional_variance(family: MarketFamily, gsv: GameStateVector, horizon: int) -> float:
    """Rough variance estimate. Higher = less confident → divides MES.

    The values are anchored to empirical match-end variances:
    - Goals: variance ~ λ × (horizon / 90)  (Poisson count quantity)
    - BTTS:  binary outcome (yes/no) — variance bounded by p(1-p) ≤ 0.25.
             We use a fixed 0.5 baseline that scales mildly with horizon
             so a 90-min-ahead bet is treated as less certain than a
             5-min-ahead one. This matches the Phase-1 contract that
             ``cvar`` divides MES (cvar must be ≥ 0.5 to stay
             well-behaved).
    - Corners: similar to goals but ~3× higher absolute scale
    - Cards: ~0.5× corners
    - Next-X: variance roughly inverse to current rate

    IMPORTANT: the return value of this function for GOALS is the RAW
    formula (λ_total × horizon / 90). Do NOT change it to the squashed
    variant (_squashed_goals_cvar) — that would silently change pick
    selection for all GOALS candidates (constraint #2).
    """
    if family == MarketFamily.GOALS:
        lam = gsv.priors.lambda_home_prematch + gsv.priors.lambda_away_prematch
        return max(0.5, lam * horizon / 90.0)
    if family == MarketFamily.BTTS:
        # Binary outcome: scale a small base by horizon-fraction. Never
        # explodes the way a goal-count variance does.
        horizon_frac = min(1.0, horizon / 90.0)
        return max(0.5, 0.5 + 0.3 * horizon_frac)
    if family == MarketFamily.CORNERS or family == MarketFamily.NEXT_CORNER:
        return max(0.5, gsv.priors.expected_corners_total * horizon / 90.0 / 3.0)
    if family == MarketFamily.CARDS:
        return max(0.5, gsv.priors.expected_cards_total * horizon / 90.0 / 2.0)
    if family == MarketFamily.NEXT_GOAL or family == MarketFamily.RACE_TO_X:
        return max(0.5, horizon / 30.0)
    return 1.0  # neutral fallback


# ──────────────────────────────────────────────────────────────────────
# Liquidity — hard floor + a smoothing curve
# ──────────────────────────────────────────────────────────────────────


def liquidity_score(line: MarketLine, target_stake: float) -> float:
    """Stake-cap based liquidity in [0, 1].

    ``target_stake`` is the operator's Kelly-fraction sized stake (USD-
    equivalent). A liquidity_score of 0 = market is closed or capped
    below 50% of target stake (rule #7 hard gate). A score of 1.0 = the
    cap is at least 4× the target (full liquidity, irrelevant cap).

    Phase-1 placeholder contract: when ``max_stake_cap == 1.0`` exactly,
    the line came from ``dual_write.market_snapshot_from_odds`` which
    hard-codes 1.0 because the Sportmonks feed does not emit caps. That
    placeholder must NOT collide with the USD-vs-USD comparison below
    (which would zero every MES — see internal research notes). We
    treat it as full liquidity until Phase 2 wires real caps; any cap
    value ≠ 1.0 still goes through the schedule.
    """
    if line.max_stake_cap == 1.0:
        return 1.0
    cap = max(0.0, line.max_stake_cap)
    if cap < 0.5 * target_stake:
        return 0.0
    if cap >= 4.0 * target_stake:
        return 1.0
    return min(1.0, (cap / target_stake - 0.5) / 3.5)


# ──────────────────────────────────────────────────────────────────────
# MES result dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MESResult:
    """Per-(thesis × market) routing score with full breakdown.

    The breakdown is preserved so the audit log can answer "why was this
    market chosen?" — required by sec 7.2 (causal audit).

    ``calibrated_winprob``: when a fitted MES→win-prob IsotonicCalibrator
    is available at compute_mes call time, this field carries the
    calibrated win probability (calibrator.transform(fair_prob, family,
    minute)). None → calibrator not available; gate logic falls back to
    raw-score thresholds (byte-identical backward-compatible behavior).
    """

    thesis_id: str
    market_id: str
    family: MarketFamily
    base_edge: float
    signal_clarity: float
    book_slowness: float
    liquidity_score: float
    conditional_variance: float
    score: float
    calibrated_winprob: float | None = None

    def passes_threshold(self, threshold: float = 0.6) -> bool:
        return self.score >= threshold


def signal_clarity(archetype: ThesisArchetype, family: MarketFamily) -> float:
    """Lookup from the static matrix. 0.40 fallback (= below threshold)."""
    return _SIGNAL_CLARITY.get(archetype, {}).get(family, 0.40)


def compute_mes(
    thesis: Thesis,
    market_id: str,
    family: MarketFamily,
    line: MarketLine,
    gsv: GameStateVector,
    *,
    fair_prob: float,
    target_stake: float = 100.0,
    mes_calibrator: "IsotonicCalibrator | None" = None,
) -> MESResult:
    """Compute MES for one (thesis, market) pair.

    Arguments:
        fair_prob: probability the conditional predictor assigns to the
            outcome the thesis points to.
        target_stake: the operator's pre-Kelly stake target.
        mes_calibrator: optional fitted ``IsotonicCalibrator`` mapping raw
            fair_prob → calibrated win probability for the (family,
            minute) cell. When provided, populates
            ``MESResult.calibrated_winprob`` so rule_5 / rule_12 can gate
            on the calibrated value. None → ``calibrated_winprob=None``
            (byte-identical behavior, no change to any existing gate).
    """
    implied = 1.0 / line.side_a_decimal if line.side_a_decimal > 1.0 else 0.0
    base_edge = max(0.0, fair_prob - implied)
    clarity = signal_clarity(thesis.archetype, family)
    slowness = book_slowness(family, gsv.last_critical_event_age_sec)
    liq = liquidity_score(line, target_stake)
    horizon = max(1, thesis.prediction.horizon.horizon_minutes)
    cvar = conditional_variance(family, gsv, horizon)

    if cvar <= 0:
        score = 0.0
    else:
        score = (base_edge * clarity * slowness * liq) / cvar
        # Scale so a "good" routing (e.g., edge 0.05 × clarity 1.0 ×
        # slowness 1.4 × liquidity 0.8 / variance 1.0) lands around 0.6
        # — the threshold from rule #5. The /0.1 normalisation comes
        # from the same calibration anchor.
        score = score / 0.1

    # Calibrated win probability — populated only when a fitted calibrator
    # is available. Does NOT affect score or any raw-threshold gates.
    cal_winprob: float | None = None
    if mes_calibrator is not None and mes_calibrator.is_fitted:
        minute = gsv.time.minute if gsv is not None else 45
        cal_winprob = mes_calibrator.transform(fair_prob, family, minute)

    return MESResult(
        thesis_id=thesis.id,
        market_id=market_id,
        family=family,
        base_edge=base_edge,
        signal_clarity=clarity,
        book_slowness=slowness,
        liquidity_score=liq,
        conditional_variance=cvar,
        score=score,
        calibrated_winprob=cal_winprob,
    )


__all__ = [
    "MESResult",
    "_squashed_goals_cvar",
    "book_slowness",
    "compute_mes",
    "conditional_variance",
    "liquidity_score",
    "signal_clarity",
]
