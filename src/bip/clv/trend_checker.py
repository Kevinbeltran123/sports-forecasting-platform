"""ClvTrendChecker (CLV-03 — D-09 through D-12).

Hourly cron wrapper around bip.clv.recorder.compute_rolling_clv_average. Queries
last 50 settled CLV values, alerts ops channel when rolling avg <
Settings.clv_trend_alert_threshold. 12h in-memory cooldown per scope.

Concurrency note (RESEARCH §CLV Trend Cooldown):
The cooldown dict is touched only from the asyncio event loop via the async def
check() method. Do NOT call from a sync context — the dict is not lock-protected.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

import structlog

from bip.clv.recorder import compute_rolling_clv_average
from bip.core.storage.repositories import ClvRecordRepository

logger = structlog.get_logger(__name__)


class _OpsSender(Protocol):
    """TelegramSender / TelegramBot shape — duck-typed for testability."""

    async def send_html(self, text: str) -> None: ...


class ClvTrendChecker:
    """D-09–D-12: hourly check, 12h in-memory cooldown.

    Iterates global + per-market scopes. Per-market only alerts when >=50 settled
    picks exist for that market (D-10 — no false-alarm noise from small samples).
    Phase 4 v1 = 1X2-only; Phase 6 corners adds a second market without code
    changes here (extend the markets tuple).
    """

    # (canonical_db_value, display_name) — DB stores MarketKey.ONEXTWO.value="onextwo";
    # alerts read by humans use the legacy "1X2" tag. D-10: extend in Phase 6 for corners.
    _MARKETS: tuple[tuple[str, str], ...] = (("onextwo", "1X2"),)

    def __init__(
        self,
        repo: ClvRecordRepository,
        ops_sender: _OpsSender,
        threshold: float = 1.0,
        cooldown_hours: int = 12,
    ) -> None:
        self._repo = repo
        self._sender = ops_sender
        self._threshold = threshold
        self._cooldown = timedelta(hours=cooldown_hours)
        # D-11: in-memory cooldown — restart resets, at most one duplicate alert per restart.
        self._last_alert: dict[str, datetime] = {}

    async def check(self) -> None:
        """Run hourly. Global rolling-50 + per-market rolling-50 (when >=50 picks)."""
        await self._check_scope(scope="global", market=None, market_display=None)
        for canonical, display in self._MARKETS:
            await self._check_scope(scope="market", market=canonical, market_display=display)

    async def _check_scope(
        self, scope: str, market: str | None, market_display: str | None
    ) -> None:
        try:
            rows = self._repo.last_n_settled(n=50, market=market)
        except Exception as exc:
            logger.error("clv_trend_query_failed", scope=scope, market=market, error=str(exc))
            return

        if len(rows) < 50:
            logger.info(
                "clv_trend_skip_insufficient", scope=scope, market=market, count=len(rows)
            )
            return

        values = [
            float(r["clv_percentage"])
            for r in rows
            if r.get("clv_percentage") is not None
        ]
        avg = compute_rolling_clv_average(values)

        if avg >= self._threshold:
            logger.info("clv_trend_ok", scope=scope, market=market, avg=avg)
            return

        # Cooldown gate (D-11) — keyed by f"{scope}:{market or '*'}" per CONTEXT.md.
        key = f"{scope}:{market or '*'}"
        now = datetime.now(UTC)
        last = self._last_alert.get(key)
        if last is not None and now - last < self._cooldown:
            logger.info("clv_trend_in_cooldown", scope=scope, market=market, avg=avg)
            return

        # Fire alert — D-12 minimal format. Display the human-friendly market tag.
        suffix = f" [market: {market_display}]" if market_display else ""
        text = f"⚠️ CLV +{avg:.1f}% < +{self._threshold:.0f}% threshold{suffix}"
        await self._sender.send_html(text)
        self._last_alert[key] = now
        logger.info("clv_trend_alert_fired", scope=scope, market=market, avg=avg)
