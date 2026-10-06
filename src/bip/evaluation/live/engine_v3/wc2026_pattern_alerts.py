"""Telegram alert formatting + dispatch for WC2026 pattern picks.

Mirrors the style of ``bip.evaluation.live.telegram_alerts``:
- HTML parse_mode, no emojis (project convention)
- Per-pick alert with rule ID, reason, max Kelly recommendation
- Promotion filter: A1 is alerted live; A2/A3/A4 emit shadow-only by default

Wiring (caller-controlled): after the runner produces picks, iterate and
call ``await send_pattern_pick_alert(bot, pick)`` for those whose rule
is in ``LIVE_ALERT_RULES``.
"""
from __future__ import annotations

import html
import logging
from typing import Any

from bip.evaluation.live.engine_v3.wc2026_pattern_runner import PatternPick

_logger = logging.getLogger(__name__)

# Promotion policy: A1 (HT 0-0 → live U2.5) is alert-eligible since its
# expected-value math has +EV @ odds 1.30. A2 and A3 are pre-match tilts
# whose value depends on operator line shopping — kept shadow-only here.
# The operator can override by passing ``force_alert=True`` to the
# dispatcher.
LIVE_ALERT_RULES: frozenset[str] = frozenset({"A1"})


def _e(s: Any) -> str:
    return html.escape(str(s), quote=False)


def format_pattern_pick_alert(pick: PatternPick) -> str:
    """Format a PatternPick as Telegram HTML (no emojis)."""
    header = f"PATTERN RULE {pick.rule_id}"
    fixture_line = (
        f"<b>{_e(pick.home_team)}</b> vs <b>{_e(pick.away_team)}</b>"
    )
    context_bits = [
        _e(pick.tournament_slug.replace("_", " ").title()),
        _e(pick.phase.title()),
    ]
    context_line = " · ".join(context_bits)

    market_line = (
        f"Market: <code>{_e(pick.market_family.value)}</code> "
        f"direction <code>{_e(pick.direction)}</code> @ <b>{pick.decimal_odds:.2f}</b>"
    )

    sizing_line = f"Max Kelly: <b>{pick.max_kelly:.0%}</b>"
    if pick.breakeven_odds is not None:
        sizing_line += f" · Breakeven odds: <i>{pick.breakeven_odds:.3f}</i>"

    reason_block = f"<blockquote>{_e(pick.reason)}</blockquote>"

    return "\n".join(
        [
            f"<b>{header}</b>",
            "",
            fixture_line,
            context_line,
            "",
            market_line,
            sizing_line,
            "",
            reason_block,
        ]
    )


def should_alert_live(pick: PatternPick, *, force: bool = False) -> bool:
    """Returns True when this pick should produce a live Telegram alert.

    Shadow rules (A2, A3) are skipped unless ``force=True``."""
    return force or pick.rule_id in LIVE_ALERT_RULES


async def send_pattern_pick_alert(
    bot: Any,
    pick: PatternPick,
    *,
    chat_id: int | str,
    force: bool = False,
    disable_notification: bool = False,
) -> bool:
    """Send a pattern pick to Telegram if the promotion filter allows.

    Returns True if a message was sent. The bot argument is duck-typed
    to support both ``python-telegram-bot`` and the test fakes used in
    tests/evaluation/live/test_telegram_*.py.

    Mirrors the no-emoji + HTML parse_mode contract of
    ``send_pick_alert``.
    """
    if not should_alert_live(pick, force=force):
        _logger.debug(
            "wc2026_pattern_alerts: skipping shadow rule %s for %s",
            pick.rule_id,
            pick.fixture_id,
        )
        return False
    text = format_pattern_pick_alert(pick)
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="HTML",
        disable_notification=disable_notification,
    )
    return True


__all__ = [
    "LIVE_ALERT_RULES",
    "format_pattern_pick_alert",
    "send_pattern_pick_alert",
    "should_alert_live",
]
