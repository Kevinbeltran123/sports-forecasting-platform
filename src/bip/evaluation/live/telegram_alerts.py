"""Telegram alerts for the Sportmonks live-edge spike (Bot v2).

Public API surface (preserved from v1):
- ``format_pick_alert(pick, *, home_score, away_score, league_name)``
- ``format_jornada_summary(...)``
- ``format_pre_jornada_brief(...)``
- ``send_pick_alert(bot, pick, ...)``
- ``send_jornada_summary(bot, **kwargs)``
- ``send_pre_jornada_brief(bot, **kwargs)``

New in v2:
- ``classify_tier(pick) -> int``  — Tier 1/2/3 routing rule
- ``format_pick_alert_with_tier(pick, tier, ...)`` — explicit override
- ``format_outcome_reply(...)``   — Tier 4 settlement reply
- ``format_burst_digest(...)``    — §E batched-send digest
- ``format_scoreboard(...)``      — pinned live P/L message

Style: no emojis (operator directive). Labels are textual; visual
hierarchy is carried by ``<b>``, ``<i>``, ``<blockquote>``.

Telegram HTML mode whitelist: ``<b> <i> <u> <s> <code> <pre> <a>
<blockquote>``. Everything else is escaped or stripped at parse time.
"""

from __future__ import annotations

import html
from typing import Any

from bip.evaluation.live.value_detector import LivePick


# ── Tier classification thresholds (tunable) ────────────────────────────────
#
# Tier 1: high-confidence CLEAN — pops visually, never silenced
# Tier 2: standard CLEAN, or borderline flagged with strong logical score
# Tier 3: flagged + weak logical score — diagnostic only, silent push

TIER1_MIN_LOGICAL = 0.85
TIER1_MIN_EDGE_PCT = 8.0
TIER2_FLAGGED_MIN_LOGICAL = 0.80


def _e(s: Any) -> str:
    """HTML-escape (Telegram parse_mode='HTML')."""
    return html.escape(str(s), quote=False)


# ── Tier classification ─────────────────────────────────────────────────────


def classify_tier(pick: LivePick) -> int:
    """Map a LivePick onto a visual tier (1, 2, or 3).

    Rules:
    - Tier 1: not flagged AND logical_score >= 0.85 AND edge_pct >= 8.0
    - Tier 2: not flagged (and not Tier 1), OR flagged with logical >= 0.80
    - Tier 3: flagged AND logical_score < 0.80
    """
    flagged = pick.flagged_reason is not None
    L = pick.logical_score
    if not flagged:
        if L >= TIER1_MIN_LOGICAL and pick.edge_pct >= TIER1_MIN_EDGE_PCT:
            return 1
        return 2
    # flagged
    if L >= TIER2_FLAGGED_MIN_LOGICAL:
        return 2
    return 3


# ── Shared helpers ──────────────────────────────────────────────────────────


def _score_str(home_score: int | None, away_score: int | None) -> str:
    if home_score is None or away_score is None:
        return ""
    return f" · Score <b>{home_score}-{away_score}</b>"


def _prob_line(pick: LivePick) -> str:
    cal = pick.our_probability
    raw = pick.model_probability_raw
    if raw and abs(raw - cal) > 1e-4:
        return f"Cal prob <b>{cal:.3f}</b> (raw {raw:.3f})"
    return f"Prob <b>{cal:.3f}</b>"


def _top_components_str(pick: LivePick, k: int = 2) -> str:
    if not pick.logical_components:
        return ""
    comps = sorted(
        pick.logical_components.items(), key=lambda kv: -abs(kv[1]),
    )[:k]
    return " · ".join(f"<i>{_e(k_)}</i>={v:.2f}" for k_, v in comps)


def _fixture_line(pick: LivePick, league_name: str | None) -> str:
    line = f"{_e(pick.home_team)} vs {_e(pick.away_team)}"
    if league_name:
        line = f"{_e(league_name)} — {line}"
    return line


# ── Tier 1 — high-confidence CLEAN ──────────────────────────────────────────


def _format_tier1(
    pick: LivePick, *, home_score: int | None, away_score: int | None,
    league_name: str | None,
) -> str:
    components = _top_components_str(pick)
    components_block = f"\nTop: {components}" if components else ""
    return (
        "<b>TIER 1 — HIGH-CONVICTION PICK</b>\n"
        f"<b>{_fixture_line(pick, league_name)}</b>\n"
        "\n"
        f"Min <b>{pick.minute}'</b>{_score_str(home_score, away_score)}\n"
        f"Market <code>{_e(pick.market)}</code> · "
        f"Selection <b>{_e(pick.selection)}</b>\n"
        f"Odds <b>{pick.bookmaker_odd:.2f}</b> · "
        f"Edge <b>+{pick.edge_pct:.2f}%</b> · "
        f"Stake <b>{pick.suggested_stake_pct:.2f}%</b>\n"
        "\n"
        f"{_prob_line(pick)} · Logical <b>{pick.logical_score:.2f}</b>\n"
        f"Kelly full <b>{pick.kelly_fraction_full:.2f}</b>"
        f"{components_block}"
    )


# ── Tier 2 — standard CLEAN or top-flagged ──────────────────────────────────


def _format_tier2(
    pick: LivePick, *, home_score: int | None, away_score: int | None,
    league_name: str | None,
) -> str:
    flag_suffix = ""
    if pick.flagged_reason:
        flag_suffix = (
            f"\n<b>FLAGGED:</b> <code>{_e(pick.flagged_reason)}</code>"
        )
    components = _top_components_str(pick)
    components_block = f"\nTop: {components}" if components else ""
    return (
        f"<b>PICK</b> · <i>{_fixture_line(pick, league_name)}</i>\n"
        "\n"
        f"Min <b>{pick.minute}'</b>{_score_str(home_score, away_score)}\n"
        f"<code>{_e(pick.market)}</code> · "
        f"<b>{_e(pick.selection)}</b> @ <b>{pick.bookmaker_odd:.2f}</b>\n"
        f"Edge <b>+{pick.edge_pct:.2f}%</b> · "
        f"Stake <b>{pick.suggested_stake_pct:.2f}%</b> · "
        f"L=<b>{pick.logical_score:.2f}</b>\n"
        f"{_prob_line(pick)} · "
        f"Kelly full <b>{pick.kelly_fraction_full:.2f}</b>"
        f"{components_block}"
        f"{flag_suffix}"
    )


# ── Tier 3 — flagged, informational only ────────────────────────────────────


def _format_tier3(
    pick: LivePick, *, home_score: int | None, away_score: int | None,
    league_name: str | None,
) -> str:
    score = _score_str(home_score, away_score).replace(" · ", " · ")
    return (
        "<b>INFO — FLAGGED PICK</b>\n"
        "<blockquote>"
        f"{_fixture_line(pick, league_name)} · "
        f"Min {pick.minute}'{score}\n"
        f"<code>{_e(pick.market)}</code> · "
        f"<b>{_e(pick.selection)}</b> @ {pick.bookmaker_odd:.2f}\n"
        f"Edge +{pick.edge_pct:.2f}% · "
        f"L={pick.logical_score:.2f}\n"
        f"<b>FLAGGED:</b> <code>{_e(pick.flagged_reason or 'unknown')}</code>"
        "</blockquote>"
    )


# ── Public dispatcher ───────────────────────────────────────────────────────


def format_pick_alert(
    pick: LivePick,
    *,
    home_score: int | None = None,
    away_score: int | None = None,
    league_name: str | None = None,
) -> str:
    """Format a LivePick alert. Selects template by ``classify_tier(pick)``."""
    tier = classify_tier(pick)
    return format_pick_alert_with_tier(
        pick, tier=tier,
        home_score=home_score, away_score=away_score, league_name=league_name,
    )


def format_pick_alert_with_tier(
    pick: LivePick,
    *,
    tier: int,
    home_score: int | None = None,
    away_score: int | None = None,
    league_name: str | None = None,
) -> str:
    """Explicit-tier formatter. Used by promoted Tier-3-to-Tier-2 flows."""
    if tier == 1:
        return _format_tier1(
            pick, home_score=home_score, away_score=away_score,
            league_name=league_name,
        )
    if tier == 2:
        return _format_tier2(
            pick, home_score=home_score, away_score=away_score,
            league_name=league_name,
        )
    if tier == 3:
        return _format_tier3(
            pick, home_score=home_score, away_score=away_score,
            league_name=league_name,
        )
    raise ValueError(f"unknown tier {tier!r}, expected 1|2|3")


# ── Tier 4 — outcome reply ──────────────────────────────────────────────────


def format_outcome_reply(
    *,
    status: str,                # 'won' | 'lost' | 'void'
    market: str,
    selection: str,
    bookmaker_odd: float,
    profit_units: float,
    placed_stake_pct: float | None = None,
    placed_odd: float | None = None,
    actual_profit_units: float | None = None,
) -> str:
    """Tier 4 — reply to the original pick message after settlement.

    The ``placed_*`` block appears only when the operator marked the pick
    as placed (via ``tg_actions`` with ``action='placed'``).
    """
    verdict = {"won": "WON", "lost": "LOST", "void": "VOID"}.get(
        status, status.upper(),
    )
    placement = ""
    if placed_stake_pct is not None and placed_odd is not None:
        placement = (
            f"\nPlaced <b>{placed_stake_pct:.2f}%</b> @ <b>{placed_odd:.2f}</b>"
        )
        if actual_profit_units is not None:
            placement += f" → <b>{actual_profit_units:+.2f}u</b>"
    return (
        f"<b>{verdict}</b> · <code>{_e(market)}</code> "
        f"<b>{_e(selection)}</b> @ {bookmaker_odd:.2f}\n"
        f"P/L (emit) <b>{profit_units:+.2f}u</b>"
        f"{placement}"
    )


# ── Burst digest (§E) ───────────────────────────────────────────────────────


def format_burst_digest(
    picks: list[LivePick],
    *,
    window_seconds: int,
) -> str:
    """Compact digest for §E batch-send when the token bucket is empty.

    Each row is a single line so 5-10 picks fit on one phone screen
    without scrolling. Inline keyboard for placement is attached by the
    caller (one row of buttons per pick).
    """
    if not picks:
        return ""
    n = len(picks)
    rows = []
    for p in picks:
        rows.append(
            f"<b>{_e(p.home_team)}</b> vs <b>{_e(p.away_team)}</b> · "
            f"<code>{_e(p.market)}</code> <b>{_e(p.selection)}</b> "
            f"@ {p.bookmaker_odd:.2f} · "
            f"+{p.edge_pct:.1f}% · L={p.logical_score:.2f}"
        )
    return (
        f"<b>BURST</b> · {n} picks in last {window_seconds}s\n"
        + "\n".join(rows)
        + "\n\n<i>Tap a row's [PLACE] button to act.</i>"
    )


# ── Scoreboard (§F) — pinned, edited in-place ───────────────────────────────


def format_scoreboard(
    *,
    date: str,
    n_emit: int,
    n_placed: int,
    n_pending: int,
    n_settled: int,
    n_won: int,
    pl_emit_units: float,
    pl_placed_units: float | None = None,
    drawdown_pct: float = 0.0,
    updated_at: str = "",
) -> str:
    wr = (n_won / n_settled * 100) if n_settled else 0.0
    placed_line = ""
    if pl_placed_units is not None:
        placed_line = f"\nP/L (placed only) <b>{pl_placed_units:+.2f}u</b>"
    dd_line = f"\nDrawdown from peak <b>{drawdown_pct:.1f}%</b>" if drawdown_pct else ""
    return (
        f"<b>TODAY</b> · {_e(date)}\n"
        "\n"
        f"Picks emitted <b>{n_emit}</b> · "
        f"Placed <b>{n_placed}</b> · Pending <b>{n_pending}</b>\n"
        f"Settled <b>{n_settled}</b> · Won <b>{n_won}</b> ({wr:.1f}%)\n"
        f"P/L (emit) <b>{pl_emit_units:+.2f}u</b>"
        f"{placed_line}"
        f"{dd_line}\n"
        "\n"
        f"<i>Updated {_e(updated_at)}</i>"
    )


# ── CLV drift footer ────────────────────────────────────────────────────────


def format_clv_footer(
    *,
    emit_odd: float,
    current_odd: float,
    drift_pct: float,
    backing_side: str = "selection",
) -> str:
    """Short italicized note appended to a pick alert when the line moves.

    Drift sign tells the operator whether the market moved WITH or
    AGAINST our pick:
    - Backing an Over / Under: drift > 0 = price up = line moving WITH us
    - Backing a result (1X2, BTTS): same — higher current odd = better entry

    ``backing_side`` is unused in the text (the operator already knows
    what they backed); kept as a parameter so future versions can route
    it to a side-aware steam marker.
    """
    direction = "with us" if drift_pct > 0 else "against us"
    if abs(drift_pct) < 0.5:
        direction = "flat"
    return (
        f"<i>CLV: emitted {emit_odd:.2f} → now {current_odd:.2f} "
        f"({drift_pct:+.1f}% · {direction})</i>"
    )


# ── Drawdown alert ──────────────────────────────────────────────────────────


def format_drawdown_alert(
    *,
    today_pl_units: float,
    threshold_pct: float,
    bankroll_baseline_units: float | None = None,
    n_settled_today: int = 0,
    loss_streak: int = 0,
    worst_pick_summary: str | None = None,
) -> str:
    """Alert when today's P/L crosses a drawdown threshold.

    ``threshold_pct`` is the absolute percentage (e.g. ``3.0`` for -3%).
    If ``bankroll_baseline_units`` is provided, P/L is contextualized
    as a percentage; otherwise raw units only.
    """
    pct_line = ""
    if bankroll_baseline_units and bankroll_baseline_units > 0:
        pct = today_pl_units / bankroll_baseline_units * 100.0
        pct_line = f" ({pct:+.2f}% bankroll)"
    streak_line = (
        f"\nLoss streak: <b>{loss_streak}</b>" if loss_streak >= 2 else ""
    )
    worst_line = (
        f"\nLargest loser today: {_e(worst_pick_summary)}"
        if worst_pick_summary else ""
    )
    return (
        "<b>DRAWDOWN ALERT</b>\n"
        "\n"
        f"Today P/L <b>{today_pl_units:+.2f}u</b>{pct_line}\n"
        f"Crossed <b>-{threshold_pct:.1f}%</b> threshold."
        f"{streak_line}{worst_line}\n"
        f"\n<i>Picks settled today: {n_settled_today}. "
        "Consider /mute 30 and re-evaluate next jornada.</i>"
    )


# ── Streak alert (HOT / COLD) ───────────────────────────────────────────────


def format_streak_alert(
    *,
    kind: str,                  # 'win' | 'loss'
    count: int,
    recent_picks_summary: list[str] | None = None,
) -> str:
    """3+ wins or 3+ losses in a row — operator confidence signal."""
    label = "HOT STREAK" if kind == "win" else "COLD STREAK"
    suffix = (
        "Confidence check — review stake sizing."
        if kind == "win" else
        "Possible model drift — check /bankroll and consider /mute 30."
    )
    recent_block = ""
    if recent_picks_summary:
        rows = "\n".join(f"  {_e(s)}" for s in recent_picks_summary[:3])
        recent_block = f"\n\nRecent:\n{rows}"
    return (
        f"<b>{label}</b>\n"
        "\n"
        f"<b>{count}</b> {kind}s in a row.{recent_block}\n"
        f"\n<i>{suffix}</i>"
    )


# ── Pre-jornada brief ───────────────────────────────────────────────────────


def format_pre_jornada_brief(
    *,
    jornada_date: str,
    n_fixtures: int,
    calibrator_fitted_at: str | None = None,
    expected_roi_band: str | None = None,
    profile_name: str | None = None,
    bankroll_baseline_units: float | None = None,
) -> str:
    lines = [
        f"<b>PRE-JORNADA</b> · {_e(jornada_date)}",
        "",
        f"Fixtures to monitor <b>{n_fixtures}</b>",
    ]
    if profile_name:
        lines.append(f"Stack profile <code>{_e(profile_name)}</code>")
    if calibrator_fitted_at:
        lines.append(
            f"Calibrator fitted <code>{_e(calibrator_fitted_at)}</code>"
        )
    if expected_roi_band:
        lines.append(f"Expected ROI band <b>{_e(expected_roi_band)}</b>")
    if bankroll_baseline_units is not None:
        lines.append(
            f"Bankroll baseline <b>{bankroll_baseline_units:.1f}u</b>"
        )
    return "\n".join(lines)


# ── Post-jornada summary ────────────────────────────────────────────────────


def format_jornada_summary(
    *,
    jornada_date: str,
    n_emit_total: int,
    n_emit_settled: int,
    n_won: int,
    profit_units: float,
    stake_pct_total: float,
    drops_by_reason: dict[str, int] | None = None,
    top_n_stats: dict[str, Any] | None = None,
    calibrator_ece_cv: float | None = None,
    calibrator_n_markets_own_fit: int | None = None,
    best_market: tuple[str, int, int, float] | None = None,
) -> str:
    win_rate = (n_won / n_emit_settled * 100) if n_emit_settled else 0.0
    roi = (profit_units / stake_pct_total * 100) if stake_pct_total > 0 else 0.0

    lines = [
        f"<b>JORNADA SUMMARY</b> · {_e(jornada_date)}",
        "",
        "<b>Emit universe</b>",
        f"Picks emitidos <b>{n_emit_total}</b> "
        f"(settled <b>{n_emit_settled}</b>)",
        f"Won <b>{n_won}</b> ({win_rate:.1f}%)",
        f"Profit <b>{profit_units:+.2f} u</b>",
        f"ROI <b>{roi:+.2f}%</b>",
    ]

    if top_n_stats:
        n = top_n_stats.get("n", 0)
        w = top_n_stats.get("won", 0)
        pr = top_n_stats.get("profit", 0.0)
        r = top_n_stats.get("roi", 0.0)
        lines.extend([
            "",
            "<b>Top-N (placeable)</b>",
            f"Picks <b>{n}</b> · Won <b>{w}</b> "
            f"({w/n*100 if n else 0:.1f}%)",
            f"P/L <b>{pr:+.2f} u</b> · ROI <b>{r:+.2f}%</b>",
        ])

    if best_market:
        mkt, mn, mw, mroi = best_market
        lines.extend([
            "",
            f"<b>Best market</b> <code>{_e(mkt)}</code> — "
            f"{mw}/{mn} ({mw/mn*100 if mn else 0:.0f}%) · "
            f"ROI <b>{mroi:+.1f}%</b>",
        ])

    if drops_by_reason:
        interesting = {
            k: v for k, v in drops_by_reason.items()
            if k != "below_min_edge"
        }
        top_drops = sorted(
            interesting.items(), key=lambda kv: -kv[1],
        )[:6]
        if top_drops:
            lines.extend([
                "",
                "<b>Drops by gate (excl. below_min_edge)</b>",
            ])
            for reason, count in top_drops:
                lines.append(
                    f"<code>{_e(reason)}</code> <b>{count}</b>"
                )

    if calibrator_ece_cv is not None or calibrator_n_markets_own_fit is not None:
        lines.append("")
        lines.append("<b>Calibrator</b>")
        if calibrator_ece_cv is not None:
            lines.append(f"ECE (5-fold CV) <b>{calibrator_ece_cv:.4f}</b>")
        if calibrator_n_markets_own_fit is not None:
            lines.append(
                f"Markets with own fit <b>{calibrator_n_markets_own_fit}</b>"
            )

    return "\n".join(lines)


# ── Send helpers (async, wrap TelegramBot) ──────────────────────────────────


async def send_pick_alert(
    bot: Any,
    pick: LivePick,
    *,
    home_score: int | None = None,
    away_score: int | None = None,
    league_name: str | None = None,
) -> None:
    text = format_pick_alert(
        pick, home_score=home_score, away_score=away_score,
        league_name=league_name,
    )
    await bot.send_html(text)


async def send_jornada_summary(bot: Any, **kwargs: Any) -> None:
    text = format_jornada_summary(**kwargs)
    await bot.send_html(text)


async def send_pre_jornada_brief(bot: Any, **kwargs: Any) -> None:
    text = format_pre_jornada_brief(**kwargs)
    await bot.send_html(text)
