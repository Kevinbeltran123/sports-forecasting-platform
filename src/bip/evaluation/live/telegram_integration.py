"""Live-watch Telegram integration with failure-safe semantics (v2).

Wraps :class:`TelegramBot` with logic specific to the Sportmonks spike:

- Start/shutdown lifecycle managed by an async context manager.
- All sends are swallowed-on-failure: a Telegram outage must NEVER crash
  the watch loop. We log and continue.
- Throttling: optionally enforce a minimum interval between sends to
  avoid flooding the channel when the cascade emits many picks at once.
- Send filters: only emit Telegram alerts above an edge threshold,
  honor mute state (with Tier-1 bypass), Tier-3 sent silently.

v2 additions:
- ``state: TelegramState | None`` — when present, message_ids are
  persisted to ``tg_messages`` for the outcome-reply loop.
- ``pick_id`` parameter on ``send_pick_safe`` enables inline keyboards
  and tier-aware routing.
- Tier-3 picks routed to ``TELEGRAM_DIAG_CHANNEL_ID`` when available;
  otherwise fall through to the primary channel.

Used by an offline research script (not published) when the operator passes
``--telegram``. Falls back to stdout-only logging when bot init fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.core.telegram.bandwidth import TokenBucket
from bip.evaluation.live.pick_tracker import DEFAULT_DB_PATH
from bip.evaluation.live.telegram_alerts import (
    classify_tier,
    format_burst_digest,
    format_jornada_summary,
    format_pick_alert_with_tier,
    format_pre_jornada_brief,
)
from bip.evaluation.live.telegram_state import (
    BurstReason,
    Role,
    TelegramState,
)
from bip.evaluation.live.value_detector import LivePick

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SendResult:
    """Result of ``LiveAlertSender.send_pick_safe``.

    Replaces the bare ``bool`` return so callers can thread the
    ``telegram_message_id`` out for delivery observability (e.g.,
    writing deliveries.parquet). Backward-compatible: ``bool(result)``
    is equivalent to the old ``return True/False`` contract.

    send_result values:
        "sent"    — message delivered; message_id is the Telegram message id.
        "skipped" — pick filtered before network call; message_id is None.
        "failed"  — network/API error during send; message_id is None.
    """

    ok: bool
    message_id: int | None
    status: str  # "sent" | "skipped" | "failed"

    def __bool__(self) -> bool:  # noqa: D105
        return self.ok


class LiveAlertSender:
    """Failure-safe Telegram alert sender for the live watch loop.

    Use as an async context manager so start/shutdown are paired::

        async with LiveAlertSender.from_env() as sender:
            if sender:
                await sender.send_pick_safe(
                    pick, pick_id=42, home_score=1, away_score=0,
                )
    """

    def __init__(
        self,
        bot: Any,
        *,
        min_edge_pct_for_alert: float = 5.0,
        skip_flagged: bool = True,
        min_interval_seconds: float = 2.0,
        state: TelegramState | None = None,
        diag_channel_id: str | None = None,
        enable_keyboards: bool = True,
        db_path: Path | None = None,
        bandwidth: TokenBucket | None = None,
    ) -> None:
        self._bot = bot
        self.min_edge_pct_for_alert = min_edge_pct_for_alert
        self.skip_flagged = skip_flagged
        self.min_interval_seconds = min_interval_seconds
        self.state = state
        self.diag_channel_id = diag_channel_id
        self.enable_keyboards = enable_keyboards and state is not None
        self._db_path: Path | None = (
            Path(db_path) if db_path is not None
            else (DEFAULT_DB_PATH if state is not None else None)
        )
        # Bandwidth: token bucket replaces min_interval_seconds when state
        # is configured (durable queue available). v1-only callers fall
        # back to the legacy throttle.
        self._bandwidth: TokenBucket | None = (
            bandwidth if bandwidth is not None
            else (TokenBucket() if state is not None else None)
        )
        self._last_send_ts: float = 0.0
        self.n_sent = 0
        self.n_skipped = 0
        self.n_failed = 0
        self.n_muted = 0
        self.n_queued = 0
        self.n_digests_sent = 0

    @classmethod
    async def from_env(
        cls,
        *,
        min_edge_pct_for_alert: float = 5.0,
        skip_flagged: bool = True,
        min_interval_seconds: float = 2.0,
        db_path: Path = DEFAULT_DB_PATH,
        enable_interactivity: bool = True,
    ) -> "LiveAlertSender | None":
        """Build from env vars. Returns None when init fails (failure-safe).

        Env vars consumed:
        - ``TELEGRAM_BOT_TOKEN``         (required)
        - ``TELEGRAM_CHANNEL_ID`` or ``TELEGRAM_PRIMARY_CHANNEL_ID`` (required)
        - ``TELEGRAM_DIAG_CHANNEL_ID``   (optional, falls back to primary)
        - ``TELEGRAM_OPERATOR_USER_ID``  (optional; enables interactivity)
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        primary = (
            os.environ.get("TELEGRAM_PRIMARY_CHANNEL_ID", "").strip()
            or os.environ.get("TELEGRAM_CHANNEL_ID", "").strip()
        )
        diag = (
            os.environ.get("TELEGRAM_DIAG_CHANNEL_ID", "").strip() or None
        )
        operator_user_id_str = os.environ.get(
            "TELEGRAM_OPERATOR_USER_ID", "",
        ).strip()
        if not token or not primary:
            logger.warning(
                "telegram_alerts_disabled "
                "(TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID missing)"
            )
            return None
        try:
            from bip.core.telegram.bot import TelegramBot
            bot = TelegramBot(token=token, channel_id=primary)
            await bot.start()
            state = TelegramState(db_path=db_path)

            if enable_interactivity and operator_user_id_str:
                try:
                    op_uid = int(operator_user_id_str)
                    await bot.enable_interactivity(
                        state=state, db_path=db_path,
                        operator_user_id=op_uid,
                    )
                except (ValueError, Exception) as exc:  # noqa: BLE001
                    logger.warning(
                        "telegram_interactivity_skipped err=%s", exc,
                    )

            return cls(
                bot,
                min_edge_pct_for_alert=min_edge_pct_for_alert,
                skip_flagged=skip_flagged,
                min_interval_seconds=min_interval_seconds,
                state=state,
                diag_channel_id=diag,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_bot_init_failed err=%s", exc)
            return None

    @property
    def bot(self) -> Any:
        """The underlying TelegramBot — exposed for OutcomeWatcher reuse."""
        return self._bot

    async def __aenter__(self) -> "LiveAlertSender":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def shutdown(self) -> None:
        try:
            await self._bot.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_bot_shutdown_failed err=%s", exc)

    # ── filtering / routing ─────────────────────────────────────────────

    def _should_send(self, pick: LivePick, *, tier: int) -> tuple[bool, str]:
        """Apply per-pick filters before attempting a send.

        Returns ``(should_send, reason_if_skipped)``.
        """
        if self.skip_flagged and pick.flagged_reason and tier != 3:
            return False, "flagged"
        if pick.edge_pct < self.min_edge_pct_for_alert:
            return False, "below_min_edge"
        if self.state is not None and self.state.is_muted():
            # Tier 1 bypasses mute by default (design memo §A).
            if tier != 1:
                return False, "muted"
        # Per-market mute — applies to ALL tiers including Tier 1, since
        # the operator's intent is "do not alert me on this market at all".
        if (
            self.state is not None
            and self.state.is_market_muted(pick.market)
        ):
            return False, "market_muted"
        return True, ""

    def _channel_for_tier(self, tier: int) -> str | None:
        """Pick the right channel for a tier. None = use bot's primary.

        Tier 3 → diag channel when configured; otherwise None (primary).
        Tier 1/2 → primary (returns None).
        """
        if tier == 3 and self.diag_channel_id:
            return self.diag_channel_id
        return None

    async def _throttle(self) -> None:
        """v1 legacy throttle. Skipped when token-bucket governor is active."""
        if self._bandwidth is not None:
            return
        if self.min_interval_seconds <= 0:
            return
        now = datetime.now(timezone.utc).timestamp()
        elapsed = now - self._last_send_ts
        if elapsed < self.min_interval_seconds:
            wait = self.min_interval_seconds - elapsed
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                pass
        self._last_send_ts = datetime.now(timezone.utc).timestamp()

    # ── public send API ─────────────────────────────────────────────────

    async def send_pick_safe(
        self,
        pick: LivePick,
        *,
        pick_id: int | None = None,
        home_score: int | None = None,
        away_score: int | None = None,
        league_name: str | None = None,
    ) -> SendResult:
        """Send a pick alert with tier-aware routing + optional keyboard.

        Returns a ``SendResult`` with ``ok``, ``message_id``, and
        ``status`` (sent|skipped|failed). ``bool(result)`` is backward-
        compatible with the old ``return True/False`` contract.

        ``pick_id`` enables interactive features (inline keyboard,
        message-id tracking). When None, falls back to v1 send-only
        behavior — useful for callers that pre-date Bot v2.
        """
        tier = classify_tier(pick)
        ok, reason = self._should_send(pick, tier=tier)
        if not ok:
            self.n_skipped += 1
            if reason == "muted":
                self.n_muted += 1
                if pick_id is not None and self.state is not None:
                    # Queue muted pick so /resume can replay.
                    self.state.enqueue_burst(pick_id, BurstReason.MUTED)
            return SendResult(ok=False, message_id=None, status="skipped")

        # Bandwidth gate: if the token bucket is exhausted and we have a
        # durable queue available, defer instead of throttling-and-sending.
        # Tier 1 picks bypass the bucket — never delay high-conviction.
        if (
            self._bandwidth is not None
            and self.state is not None
            and pick_id is not None
            and tier != 1
            and not self._bandwidth.try_consume(cost=1.0)
        ):
            self.state.enqueue_burst(pick_id, BurstReason.BURST)
            self.n_queued += 1
            return SendResult(ok=False, message_id=None, status="skipped")

        await self._throttle()
        try:
            text = format_pick_alert_with_tier(
                pick, tier=tier,
                home_score=home_score, away_score=away_score,
                league_name=league_name,
            )
            reply_markup = None
            if self.enable_keyboards and pick_id is not None:
                # Lazy import to keep keyboards.py optional in test contexts
                from bip.core.telegram.keyboards import build_pick_keyboard
                reply_markup = build_pick_keyboard(pick_id, tier=tier)

            target_channel = self._channel_for_tier(tier)
            disable_notification = (tier == 3)

            message_id = await self._bot.send_html(
                text,
                chat_id=target_channel,
                reply_markup=reply_markup,
                disable_notification=disable_notification,
            )

            # Persist message_id for outcome reply loop.
            if pick_id is not None and self.state is not None:
                channel = (
                    str(target_channel) if target_channel
                    else str(self._bot.channel_id)
                )
                self.state.record_message(
                    pick_id=pick_id,
                    channel_id=channel,
                    message_id=message_id,
                    role=Role.PICK,
                    tier=tier,
                )

            self.n_sent += 1
            return SendResult(ok=True, message_id=message_id, status="sent")
        except Exception as exc:  # noqa: BLE001
            self.n_failed += 1
            logger.warning(
                "telegram_send_failed pick=%s/%s err=%s",
                pick.market, pick.selection, exc,
            )
            return SendResult(ok=False, message_id=None, status="failed")

    async def run_burst_flush_forever(
        self, *, period_seconds: float = 30.0,
        window_seconds: int = 60, batch_limit: int = 10,
    ) -> None:
        """Long-running task: periodically drain the burst queue.

        Failure-safe: any exception in one tick is logged and the loop
        continues. Cancel via the caller's task object.
        """
        while True:
            try:
                await self.flush_burst_queue(
                    window_seconds=window_seconds, batch_limit=batch_limit,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("burst_flush_tick_failed err=%s", exc)
            try:
                await asyncio.sleep(period_seconds)
            except asyncio.CancelledError:
                return

    async def flush_burst_queue(
        self, *, window_seconds: int = 60, batch_limit: int = 10,
    ) -> int:
        """Drain queued picks and send a single burst-digest message.

        Returns the number of picks included in the digest (0 if queue
        empty or no tokens available). Failure-safe.

        Requires both ``state`` and ``db_path`` to be configured;
        otherwise it's a no-op.
        """
        if self.state is None or self._db_path is None:
            return 0
        if self.state.burst_queue_size() == 0:
            return 0
        # Each digest message costs 1 token. Skip if bucket dry; flush will
        # be retried by the next periodic call.
        if (
            self._bandwidth is not None
            and not self._bandwidth.try_consume(cost=1.0)
        ):
            return 0

        pick_ids = self.state.drain_burst_queue(limit=batch_limit)
        if not pick_ids:
            return 0

        picks, pid_order = self._load_picks_for_digest(pick_ids)
        if not picks:
            return 0

        try:
            from bip.core.telegram.keyboards import build_burst_digest_keyboard
            text = format_burst_digest(picks, window_seconds=window_seconds)
            keyboard = build_burst_digest_keyboard(pid_order)
            message_id = await self._bot.send_html(
                text,
                reply_markup=keyboard,
                disable_notification=True,
            )
            # Persist message_id per pick so outcome replies thread to
            # the digest. Use role=BURST_DIGEST.
            channel = str(self._bot.channel_id)
            for pid in pid_order:
                self.state.record_message(
                    pick_id=pid, channel_id=channel,
                    message_id=message_id,
                    role=Role.BURST_DIGEST,
                )
            self.n_digests_sent += 1
            return len(picks)
        except Exception as exc:  # noqa: BLE001
            # Re-queue the drained pick_ids so they're not lost.
            for pid in pick_ids:
                self.state.enqueue_burst(pid, BurstReason.RATE_LIMIT)
            self.n_failed += 1
            logger.warning("burst_digest_send_failed err=%s", exc)
            return 0

    def _load_picks_for_digest(
        self, pick_ids: list[int],
    ) -> tuple[list[LivePick], list[int]]:
        """Reconstruct LivePick objects from picks.db rows.

        Returns ``(picks, pid_order)`` where ``pid_order`` is parallel
        to ``picks`` and carries the DB rowids — used downstream to
        build the inline keyboard and write tg_messages rows.
        """
        if not pick_ids or self._db_path is None:
            return [], []
        placeholders = ",".join("?" * len(pick_ids))
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT id, fixture_id, minute, home_team, away_team,
                       market, selection, bookmaker_id, bookmaker_odd,
                       our_probability, fair_odd, edge_pct,
                       kelly_fraction_full, suggested_stake_pct,
                       market_description, snapshot_kind,
                       flagged_reason, logical_score
                FROM picks
                WHERE id IN ({placeholders})
                ORDER BY edge_pct DESC
                """,
                pick_ids,
            ).fetchall()
        picks: list[LivePick] = []
        pid_order: list[int] = []
        for r in rows:
            picks.append(LivePick(
                fixture_id=int(r["fixture_id"]),
                minute=int(r["minute"]),
                home_team=str(r["home_team"]),
                away_team=str(r["away_team"]),
                market=str(r["market"]),
                selection=str(r["selection"]),
                bookmaker_id=int(r["bookmaker_id"]),
                bookmaker_odd=float(r["bookmaker_odd"]),
                our_probability=float(r["our_probability"]),
                fair_odd=float(r["fair_odd"]),
                edge_pct=float(r["edge_pct"]),
                kelly_fraction_full=float(r["kelly_fraction_full"]),
                suggested_stake_pct=float(r["suggested_stake_pct"]),
                market_description=r["market_description"],
                snapshot_kind=str(r["snapshot_kind"]),
                flagged_reason=r["flagged_reason"],
                logical_score=float(r["logical_score"] or 1.0),
            ))
            pid_order.append(int(r["id"]))
        return picks, pid_order

    async def send_brief_safe(self, **kwargs: Any) -> bool:
        try:
            text = format_pre_jornada_brief(**kwargs)
            await self._bot.send_html(text)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_brief_failed err=%s", exc)
            return False

    async def send_summary_safe(self, **kwargs: Any) -> bool:
        try:
            text = format_jornada_summary(**kwargs)
            await self._bot.send_html(text)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_summary_failed err=%s", exc)
            return False
