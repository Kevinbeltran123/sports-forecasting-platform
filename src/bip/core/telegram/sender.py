"""Telegram pick sender — renders Pick rows through Jinja2 and pushes to TelegramBot.

D-12: parse_mode='HTML' (handled by TelegramBot.send_html).
D-14: FLAG verdict → WARNING marker (template branch).
D-15: bullets sourced from claude_summary, split on ' * ', capped at 3.
"""

from __future__ import annotations

from typing import Any

import structlog
from jinja2 import Environment, PackageLoader, select_autoescape

from bip.core.storage.models import Pick
from bip.core.telegram.bot import TelegramBot

logger = structlog.get_logger(__name__)

_env = Environment(
    loader=PackageLoader("bip.core.telegram", "templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _build_context(pick: Pick, home_team: str, away_team: str, model_version: str,
                   reason_code: str | None) -> dict[str, Any]:
    return {
        "claude_validation": pick.claude_validation,
        "fixture": f"{home_team} vs {away_team}",
        "market": pick.market,
        "selection": pick.selection,
        "best_odds": pick.best_odds,
        "edge": pick.edge,
        "suggested_stake": pick.suggested_stake,
        "model_probability": pick.model_probability,
        "bookmaker": pick.bookmaker,
        "model_version": model_version,
        "reason_code": reason_code or "",
    }


def render_pick(
    pick: Pick,
    home_team: str,
    away_team: str,
    model_version: str,
    reason_code: str | None = None,
) -> str:
    raw = pick.claude_summary or ""
    bullets = [b.strip() for b in raw.split(" * ") if b.strip()]
    ctx = _build_context(pick, home_team, away_team, model_version, reason_code)
    return _env.get_template("pick.html").render(pick=ctx, summary_bullets=bullets)


class TelegramSender:
    """Pushes rendered picks through TelegramBot. One message per pick (D-12)."""

    def __init__(self, bot: TelegramBot) -> None:
        self._bot = bot

    async def send_pick(
        self,
        pick: Pick,
        home_team: str,
        away_team: str,
        model_version: str,
        reason_code: str | None = None,
    ) -> None:
        text = render_pick(pick, home_team, away_team, model_version, reason_code)
        await self._bot.send_html(text)
        logger.info(
            "telegram_pick_sent",
            fixture_id=pick.fixture_id,
            market=pick.market,
            verdict=pick.claude_validation,
            length=len(text),
        )
