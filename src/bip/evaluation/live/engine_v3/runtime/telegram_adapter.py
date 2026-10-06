"""Adapter: v3 ``ShadowPick`` → v2 ``LivePick`` for Telegram delivery.

Why an adapter and not a parallel sender:
    The Bot v2 alert pipeline (``LiveAlertSender``) already handles
    tier classification, throttling, keyboard layout, message-id
    tracking for outcome replies, burst flushing, and CLV resampling.
    Re-implementing all that for v3 picks would be a maintenance trap.

    The adapter packs the v3 thesis + MES + market line into the
    ``LivePick`` shape that ``LiveAlertSender.send_pick_safe`` expects.
    Tier-1/2/3 classification then drives the same templates the
    operator already knows. The adapter is the ONLY place v3-specific
    concepts (archetype, MES factors, calibration drift) cross into
    the alert layer.

Mapping decisions:

    LivePick.market          ← thesis.prediction.family.value
                               (e.g. 'goals', 'corners', 'btts')

    LivePick.selection       ← '<direction> <line_value>'
                               (e.g. 'under 2.5', 'over 10.5', 'yes')

    LivePick.bookmaker_odd   ← side_a/side_b of the matched MarketLine
                               (same logic as ShadowLogger._derive_stake_fields)

    LivePick.our_probability ← candidate.fair_prob (post-calibration)
    LivePick.edge_pct        ← ((bookmaker_odd × fair_prob) − 1) × 100

    LivePick.logical_score   ← clamp(mes.score / 5.0, 0, 1)
                               5.0 is the empirical p99 of MES across
                               Day-3+Day-4 picks; this gives a 0-1 scale
                               compatible with tier classification.

    LivePick.flagged_reason  ← None
                               Anything that's a "flag" in v2 vocab is
                               already a gate denial in v3 — flagged
                               picks don't reach this adapter.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bip.evaluation.live.value_detector import LivePick

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.gsv import GameStateVector
    from bip.evaluation.live.engine_v3.pipeline import ShadowPick


log = logging.getLogger(__name__)

# Empirical MES-to-logical-score scale. Day-3+Day-4 sample observed
# MES p99 ≈ 5.0; we treat that as the "near-maximum logical confidence"
# anchor so the tier classifier sees ~1.0 for top picks.
_MES_LOGICAL_DIVISOR = 5.0


def _derive_book_odd(
    market_line, direction: str,
) -> tuple[float | None, float | None]:
    """Mirror of ``shadow_logger._derive_stake_fields`` for the odd-only
    half. Returns ``(bookmaker_odd, line_value)``."""
    if market_line is None:
        return None, None
    line_value = (
        float(market_line.line_value)
        if market_line.line_value is not None else None
    )
    d = (direction or "").lower()
    if d in ("over", "yes", "home"):
        odd = market_line.side_a_decimal
    elif d in ("under", "no", "away", "draw"):
        odd = market_line.side_b_decimal or market_line.side_a_decimal
    else:
        odd = market_line.side_a_decimal
    if odd is None or odd <= 1.0:
        return None, line_value
    return float(odd), line_value


def _selection_label(direction: str, line_value: float | None) -> str:
    """Human-readable selection string for the alert template."""
    d = (direction or "").lower()
    if line_value is None:
        return d
    # Format the line crisply — Sportmonks emits .0 / .5 / .25 / .75 lines
    if abs(line_value - round(line_value)) < 1e-6:
        line_str = f"{int(round(line_value))}"
    else:
        line_str = f"{line_value:g}"
    return f"{d} {line_str}"


def shadow_pick_to_live_pick(
    pick: "ShadowPick",
    gsv: "GameStateVector",
) -> LivePick | None:
    """Convert a v3 ShadowPick into a v2 LivePick for Telegram delivery.

    Returns ``None`` when the conversion can't produce a coherent
    LivePick (missing market line, sub-1.0 odd, etc.). Caller should
    skip silently — the gate already approved the pick at the v3 layer,
    so a missing line at the alert stage is a transient odds feed gap,
    not a quality concern.
    """
    cand = pick.candidate
    thesis = cand.thesis
    line = gsv.markets.lines.get(cand.market_id)
    book_odd, line_value = _derive_book_odd(line, thesis.prediction.direction)
    if book_odd is None:
        log.debug(
            "v3_adapter_skip fixture=%d market=%s reason=no_book_odd",
            gsv.fixture_id, cand.market_id,
        )
        return None

    fair_prob = float(cand.fair_prob)
    edge_pct = ((book_odd * fair_prob) - 1.0) * 100.0
    if edge_pct <= 0.0:
        # The v3 gate may have allowed a pick on MES-composite grounds
        # even with thin nominal edge (e.g., high clarity × low base_edge).
        # The alert layer's tier classifier requires positive edge_pct.
        log.debug(
            "v3_adapter_skip fixture=%d market=%s edge_pct=%.2f",
            gsv.fixture_id, cand.market_id, edge_pct,
        )
        return None

    fair_odd = 1.0 / fair_prob if fair_prob > 0 else float("inf")

    # MES-derived logical score, clamped to [0, 1].
    logical_score = min(1.0, max(0.0, float(cand.mes.score) / _MES_LOGICAL_DIVISOR))

    # MES factor breakdown — useful in the alert footer.
    logical_components = {
        "base_edge": float(cand.mes.base_edge),
        "signal_clarity": float(cand.mes.signal_clarity),
        "book_slowness": float(cand.mes.book_slowness),
        "liquidity": float(cand.mes.liquidity_score),
        "cvar": float(cand.mes.conditional_variance),
    }

    # Full-Kelly fraction — operator typically multiplies by their own
    # fraction (1/4 by convention) before placing the bet.
    b = book_odd - 1.0
    kelly_full = (fair_prob * b - (1.0 - fair_prob)) / b if b > 0 else 0.0
    kelly_full = max(0.0, kelly_full)
    suggested_stake_pct = kelly_full * 100.0 * 0.25  # default 1/4 Kelly

    return LivePick(
        fixture_id=gsv.fixture_id,
        minute=gsv.time.minute,
        home_team=gsv.home_team_name or "",
        away_team=gsv.away_team_name or "",
        market=thesis.prediction.family.value,
        selection=_selection_label(thesis.prediction.direction, line_value),
        bookmaker_id=0,  # v3 doesn't track bookmaker_id per-line yet
        bookmaker_odd=book_odd,
        our_probability=fair_prob,
        fair_odd=fair_odd,
        edge_pct=edge_pct,
        kelly_fraction_full=kelly_full,
        suggested_stake_pct=suggested_stake_pct,
        market_description=cand.market_id,
        snapshot_kind="live",
        flagged_reason=None,
        logical_score=logical_score,
        logical_components=logical_components,
        confidence_half_width=0.0,
        model_probability_raw=fair_prob,
    )


__all__ = ["shadow_pick_to_live_pick"]
