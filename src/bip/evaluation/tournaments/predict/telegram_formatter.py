"""Format a MatchPrediction as a Telegram-ready alert message.

The operator receives this message, compares the "odd justa" (fair odds)
against the current bookmaker line, and decides whether to bet.

Fair odds = 1 / probability. A bookmaker offering more than this has
positive expected value for the bettor.

Usage::

    from bip.evaluation.tournaments.predict.telegram_formatter import format_match_alert

    text = format_match_alert(prediction, corners=corners_dist)
    await bot.send_message(chat_id=..., text=text, parse_mode="Markdown")
"""

from __future__ import annotations

import math

import numpy as np

from bip.evaluation.tournaments.predict.output import MatchPrediction
from bip.evaluation.tournaments.predictors.corners_poisson import CornersDistribution

_LINEUP_TAG = {
    "confirmed": "✅ Confirmada",
    "heuristic_qualifier": "🟡 Heurística",
    "fallback_squad": "⚠️ Provisional",
}


def format_match_alert(
    pred: MatchPrediction,
    corners: CornersDistribution | None = None,
) -> str:
    """Return a Telegram Markdown string for one fixture.

    Use pred.corners if already attached to the prediction, or pass
    `corners` explicitly to override / supply it separately.
    """
    c = corners if corners is not None else pred.corners
    home = pred.home
    away = pred.away
    g = pred.goals
    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines.append(f"⚽ *{home.team_name}* vs *{away.team_name}*")
    lines.append(f"🏆 {pred.tournament_slug.replace('_', ' ').title()}")
    lines.append(f"🕐 {pred.kickoff.strftime('%d %b %Y  %H:%M')} UTC")
    h_tag = _LINEUP_TAG.get(home.lineup_confidence, home.lineup_confidence)
    a_tag = _LINEUP_TAG.get(away.lineup_confidence, away.lineup_confidence)
    lines.append(f"Lineup  Local: {h_tag}  |  Visitante: {a_tag}")
    lines.append("")

    # ── 1X2 ──────────────────────────────────────────────────────────────────
    lines.append("*── RESULTADO 1X2 ──*")
    lines.append(f"  {home.team_name}: {_pct(g.p_home_win)}  →  odd justa *{_odd(g.p_home_win)}*")
    lines.append(f"  Empate:              {_pct(g.p_draw)}  →  odd justa *{_odd(g.p_draw)}*")
    lines.append(f"  {away.team_name}: {_pct(g.p_away_win)}  →  odd justa *{_odd(g.p_away_win)}*")
    lines.append("")

    # ── Handicap asiático ─────────────────────────────────────────────────────
    lines.append("*── HÁNDICAP ASIÁTICO ──*")
    for label, prob in _handicap_rows(pred):
        lines.append(f"  {label}: {_pct(prob)}  →  *{_odd(prob)}*")
    lines.append("")

    # ── Goles ─────────────────────────────────────────────────────────────────
    lines.append("*── GOLES ──*")
    lines.append(
        f"  λ {home.team_name}: *{g.lambda_home:.2f}*  |  "
        f"λ {away.team_name}: *{g.lambda_away:.2f}*  |  Total: *{g.expected_total_goals:.2f}*"
    )
    over_1_5 = _p_over_from_grid(g, 1)
    over_3_5 = _p_over_from_grid(g, 3)
    lines.append(f"  Over 1.5: {_pct(over_1_5)}  →  *{_odd(over_1_5)}*")
    lines.append(f"  Over 2.5: {_pct(g.p_over_2_5)}  →  *{_odd(g.p_over_2_5)}*")
    lines.append(f"  Over 3.5: {_pct(over_3_5)}  →  *{_odd(over_3_5)}*")
    lines.append(f"  BTTS:     {_pct(g.p_btts)}  →  *{_odd(g.p_btts)}*")
    lines.append("")

    # ── Primer tiempo (goles) ────────────────────────────────────────────────
    λ_fh = (g.lambda_home + g.lambda_away) * 0.42
    p_fh_over_0_5 = _poisson_over(λ_fh, 0)
    p_fh_over_1_5 = _poisson_over(λ_fh, 1)
    lines.append("*── PRIMER TIEMPO ──*")
    lines.append(f"  λ goles 1T: *{λ_fh:.2f}*")
    lines.append(f"  Over 0.5 (1T): {_pct(p_fh_over_0_5)}  →  *{_odd(p_fh_over_0_5)}*")
    lines.append(f"  Over 1.5 (1T): {_pct(p_fh_over_1_5)}  →  *{_odd(p_fh_over_1_5)}*")
    lines.append("")

    # ── Corners ───────────────────────────────────────────────────────────────
    if c is not None:
        lines.append("*── CORNERS ──*")
        lines.append(
            f"  λ {home.team_name}: *{c.lambda_home:.1f}*  |  "
            f"λ {away.team_name}: *{c.lambda_away:.1f}*  |  Total: *{c.lambda_total:.1f}*"
        )
        lines.append(f"  Over 8.5:  {_pct(c.p_over_8_5)}  →  *{_odd(c.p_over_8_5)}*")
        lines.append(f"  Over 9.5:  {_pct(c.p_over_9_5)}  →  *{_odd(c.p_over_9_5)}*")
        lines.append(f"  Over 10.5: {_pct(c.p_over_10_5)}  →  *{_odd(c.p_over_10_5)}*")
        lines.append(f"  Más corners {home.team_name}: {_pct(c.p_home_more)}  →  *{_odd(c.p_home_more)}*")
        lines.append(f"  Más corners {away.team_name}: {_pct(c.p_away_more)}  →  *{_odd(c.p_away_more)}*")
        lines.append(
            f"  {home.team_name} -1.5 corners: "
            f"{_pct(c.p_home_ahc_minus_1_5)}  →  *{_odd(c.p_home_ahc_minus_1_5)}*"
        )
        lines.append(
            f"  Corners 1T Over 4.5: {_pct(c.p_total_fh_over_4_5)}  →  *{_odd(c.p_total_fh_over_4_5)}*"
        )
        lines.append(
            f"  Corners 1T Over 5.5: {_pct(c.p_total_fh_over_5_5)}  →  *{_odd(c.p_total_fh_over_5_5)}*"
        )
        lines.append("")

    # ── P(marcar) — top 6 jugadores ─────────────────────────────────────────
    top = _top_scorers(pred, n=6)
    if top:
        lines.append("*── P(MARCAR) ──*")
        for name, p_score in top:
            lines.append(f"  {name}: {_pct(p_score)}  →  *{_odd(p_score)}*")
        lines.append("")

    # ── Advertencias ─────────────────────────────────────────────────────────
    if pred.warnings:
        lines.append("*── ⚠️ ADVERTENCIAS ──*")
        for w in pred.warnings:
            lines.append(f"  • {w}")
        lines.append("")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _odd(p: float) -> str:
    if p <= 0.0:
        return "∞"
    return f"{1.0 / p:.2f}"


def _p_over_from_grid(g, n: int) -> float:
    """P(total goals > n) from the score_grid (e.g., n=1 → Over 1.5)."""
    grid = g.score_grid_as_array()
    total = 0.0
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if i + j > n:
                total += grid[i, j]
    return min(1.0, float(total))


def _poisson_over(lam: float, k_min: int) -> float:
    """P(Poisson(lam) > k_min) via exact PMF sum."""
    p_le = sum(
        math.exp(-lam) * (lam ** k) / math.factorial(k)
        for k in range(k_min + 1)
    )
    return max(0.0, 1.0 - p_le)


def _handicap_rows(pred: MatchPrediction) -> list[tuple[str, float]]:
    """Asian handicap and Draw No Bet rows derived from the score grid."""
    g = pred.goals
    grid = g.score_grid_as_array()
    n = grid.shape[0]
    hn = pred.home.team_name
    an = pred.away.team_name

    def p_home_margin(min_margin: int) -> float:
        return float(sum(
            grid[i, j] for i in range(n) for j in range(n)
            if i - j >= min_margin
        ))

    def p_away_margin(min_margin: int) -> float:
        return float(sum(
            grid[i, j] for i in range(n) for j in range(n)
            if j - i >= min_margin
        ))

    # Draw No Bet: renormalize excluding draws
    no_draw_total = g.p_home_win + g.p_away_win
    p_home_dnb = g.p_home_win / max(1e-9, no_draw_total)
    p_away_dnb = g.p_away_win / max(1e-9, no_draw_total)

    return [
        (f"{hn} DNB (+0.5)", p_home_dnb),
        (f"{an} DNB (+0.5)", p_away_dnb),
        (f"{hn} -1.5", p_home_margin(2)),
        (f"{hn} -2.5", p_home_margin(3)),
        (f"{an} -1.5", p_away_margin(2)),
        (f"{an} -2.5", p_away_margin(3)),
    ]


def _top_scorers(pred: MatchPrediction, *, n: int = 6) -> list[tuple[str, float]]:
    all_starters = list(pred.home.starters) + list(pred.away.starters)
    ranked = sorted(all_starters, key=lambda p: p.p_anytime_scorer, reverse=True)
    return [(p.name, p.p_anytime_scorer) for p in ranked[:n]]
