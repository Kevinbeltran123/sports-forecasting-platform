"""Periodic outcome-reply + scoreboard maintenance task.

Decoupling rationale (per design memo §F): the grader writes only
``picks.status`` + ``picks.profit_units``. This watcher polls
``picks.db`` for newly-settled picks, replies to the original Telegram
message, and refreshes the pinned scoreboard.

Lives in the watch loop's asyncio task — failure-safe: any exception
in ``tick()`` is logged but never raised. Outage of Telegram does not
prevent the next tick from trying again on the same pick set (the
outcome-reply row only gets written on a successful send).

Pinned scoreboard: derived from ``picks.db`` on every render — no
persistent state beyond ``tg_state['scoreboard_msg_id']``. Cannot drift.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.evaluation.live.telegram_alerts import (
    format_drawdown_alert,
    format_outcome_reply,
    format_scoreboard,
    format_streak_alert,
)
from bip.evaluation.live.telegram_state import (
    Action,
    Role,
    TelegramState,
)


logger = logging.getLogger(__name__)


# ── Alert thresholds (tunable) ──────────────────────────────────────────────
#
# Drawdown alert fires once per day when today's P/L crosses below
# -DRAWDOWN_THRESHOLD_PCT of bankroll baseline. When no baseline is
# configured, the absolute-units fallback is used.
DRAWDOWN_THRESHOLD_PCT: float = 3.0
DRAWDOWN_FALLBACK_UNITS: float = -3.0

# Streak alert fires when a consecutive win/loss run reaches this many.
STREAK_ALERT_THRESHOLD: int = 3


class OutcomeWatcher:
    """Periodic task: outcome replies + scoreboard maintenance.

    Usage::

        watcher = OutcomeWatcher(
            bot=alert_sender._bot, state=state, db_path=picks_db,
            primary_channel_id=os.environ["TELEGRAM_CHANNEL_ID"],
        )
        task = asyncio.create_task(watcher.run_forever(period_seconds=60))
        # ... watcher.cancel() on shutdown

    Or, when integrated with the watch loop's own scheduling, call
    ``await watcher.tick()`` once per period.
    """

    def __init__(
        self,
        *,
        bot: Any,
        state: TelegramState,
        db_path: Path,
        primary_channel_id: str,
        diag_channel_id: str | None = None,
        scoreboard_enabled: bool = True,
        alerts_enabled: bool = True,
    ) -> None:
        self._bot = bot
        self._state = state
        self._db_path = Path(db_path)
        self._primary_channel_id = str(primary_channel_id)
        self._diag_channel_id = (
            str(diag_channel_id) if diag_channel_id else None
        )
        self._scoreboard_enabled = scoreboard_enabled
        self._alerts_enabled = alerts_enabled
        self.n_outcome_replies_sent = 0
        self.n_scoreboard_updates = 0
        self.n_drawdown_alerts_sent = 0
        self.n_streak_alerts_sent = 0
        self.n_failures = 0
        self._stopped = False

    @property
    def _alert_channel(self) -> str:
        """Diag if configured, else primary (drawdown/streak alerts)."""
        return self._diag_channel_id or self._primary_channel_id

    # ── one-tick API ────────────────────────────────────────────────────

    async def tick(self) -> int:
        """Run one cycle. Returns the number of outcome replies sent.

        Failure-safe. The scoreboard update is best-effort; an edit
        failure (e.g. message-too-old) triggers a fresh pin on the next
        tick.
        """
        try:
            n = await self._send_outcome_replies()
        except Exception as exc:  # noqa: BLE001
            self.n_failures += 1
            logger.warning("outcome_replies_failed err=%s", exc)
            n = 0
        if self._scoreboard_enabled:
            try:
                await self.update_scoreboard()
            except Exception as exc:  # noqa: BLE001
                self.n_failures += 1
                logger.warning("scoreboard_update_failed err=%s", exc)
        return n

    async def run_forever(self, *, period_seconds: float = 60.0) -> None:
        """Blocking infinite loop. Cancel via the caller's task object."""
        while not self._stopped:
            await self.tick()
            try:
                await asyncio.sleep(period_seconds)
            except asyncio.CancelledError:
                self._stopped = True
                return

    def stop(self) -> None:
        self._stopped = True

    # ── outcome replies ─────────────────────────────────────────────────

    async def _send_outcome_replies(self) -> int:
        """Send one reply per pick that settled but has no outcome yet."""
        rows = self._fetch_pending_outcome_rows()
        n_sent = 0
        for row in rows:
            try:
                await self._send_one_outcome_reply(row)
                n_sent += 1
                self.n_outcome_replies_sent += 1
            except Exception as exc:  # noqa: BLE001
                self.n_failures += 1
                logger.warning(
                    "outcome_reply_send_failed pick_id=%s err=%s",
                    row["pick_id"], exc,
                )
                continue
            # Best-effort: update streak + check drawdown after each
            # successful outcome reply. Failures here don't affect the
            # outcome-reply contract.
            if self._alerts_enabled:
                try:
                    await self._update_streak_and_alert(row)
                except Exception as exc:  # noqa: BLE001
                    self.n_failures += 1
                    logger.warning(
                        "streak_alert_failed pick_id=%s err=%s",
                        row["pick_id"], exc,
                    )
                try:
                    await self._check_drawdown_and_alert()
                except Exception as exc:  # noqa: BLE001
                    self.n_failures += 1
                    logger.warning(
                        "drawdown_alert_failed err=%s", exc,
                    )
        return n_sent

    def _fetch_pending_outcome_rows(self) -> list[sqlite3.Row]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                  p.id              AS pick_id,
                  p.status          AS status,
                  p.profit_units    AS profit_units,
                  p.market          AS market,
                  p.selection       AS selection,
                  p.bookmaker_odd   AS bookmaker_odd,
                  p.actual_stake_units AS actual_stake_units,
                  p.betano_odd      AS betano_odd,
                  m.channel_id      AS channel_id,
                  m.message_id      AS message_id,
                  a.stake_pct       AS placed_stake_pct
                FROM picks p
                JOIN tg_messages m
                  ON m.pick_id = p.id AND m.role = ?
                LEFT JOIN tg_messages r
                  ON r.pick_id = p.id AND r.role = ?
                LEFT JOIN tg_actions a
                  ON a.pick_id = p.id AND a.action = ?
                WHERE p.status IN ('won', 'lost', 'void')
                  AND r.pick_id IS NULL
                ORDER BY p.settled_at
                LIMIT 100
                """,
                (Role.PICK, Role.OUTCOME, Action.PLACED),
            ).fetchall()
        return list(rows)

    async def _send_one_outcome_reply(self, row: sqlite3.Row) -> None:
        placed_stake_pct = row["placed_stake_pct"]
        placed_odd = row["betano_odd"]
        actual_pl = None
        if (
            placed_stake_pct is not None
            and placed_odd is not None
            and row["status"] in ("won", "lost")
        ):
            actual_pl = _actual_profit_units(
                status=row["status"],
                stake_pct=float(placed_stake_pct),
                odd=float(placed_odd),
            )

        text = format_outcome_reply(
            status=row["status"],
            market=row["market"],
            selection=row["selection"],
            bookmaker_odd=float(row["bookmaker_odd"]),
            profit_units=float(row["profit_units"] or 0.0),
            placed_stake_pct=(
                float(placed_stake_pct)
                if placed_stake_pct is not None else None
            ),
            placed_odd=(
                float(placed_odd) if placed_odd is not None else None
            ),
            actual_profit_units=actual_pl,
        )

        reply_message_id = await self._bot.send_html(
            text,
            chat_id=row["channel_id"],
            reply_to_message_id=row["message_id"],
            disable_notification=True,
        )
        self._state.record_message(
            pick_id=row["pick_id"],
            channel_id=row["channel_id"],
            message_id=reply_message_id,
            role=Role.OUTCOME,
        )

    # ── streak detection ────────────────────────────────────────────────

    async def _update_streak_and_alert(self, row: sqlite3.Row) -> None:
        """Update consecutive-outcome streak; alert when crossing threshold.

        Void picks are neutral: streak is paused, neither extended nor
        broken. Alert fires once per streak when count crosses
        ``STREAK_ALERT_THRESHOLD`` (not on subsequent picks of the same
        streak).
        """
        status = row["status"]
        if status == "void":
            return
        new_kind = "win" if status == "won" else "loss"

        prev = self._state.get_state("current_streak", default=None) or {}
        prev_kind = prev.get("kind")
        prev_count = int(prev.get("count", 0))
        prev_alerted_at = int(prev.get("alerted_at_count", 0))

        if prev_kind == new_kind:
            new_count = prev_count + 1
            alerted_at = prev_alerted_at
        else:
            new_count = 1
            alerted_at = 0  # streak reset → re-arm alert

        self._state.set_state("current_streak", {
            "kind": new_kind,
            "count": new_count,
            "alerted_at_count": alerted_at,
        })

        # Fire alert at the threshold-crossing only.
        if new_count == STREAK_ALERT_THRESHOLD and alerted_at < STREAK_ALERT_THRESHOLD:
            recent = self._recent_settled_summary(
                kind_filter=status, limit=STREAK_ALERT_THRESHOLD,
            )
            text = format_streak_alert(
                kind=new_kind, count=new_count,
                recent_picks_summary=recent,
            )
            try:
                await self._bot.send_html(
                    text, chat_id=self._alert_channel,
                    disable_notification=True,
                )
                self.n_streak_alerts_sent += 1
            except Exception as exc:  # noqa: BLE001
                self.n_failures += 1
                logger.warning("streak_alert_send_failed err=%s", exc)
                return
            # Mark this streak as already alerted at this count so future
            # picks in the same streak don't re-fire.
            self._state.set_state("current_streak", {
                "kind": new_kind,
                "count": new_count,
                "alerted_at_count": new_count,
            })

    def _recent_settled_summary(
        self, *, kind_filter: str, limit: int,
    ) -> list[str]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT market, selection, bookmaker_odd, profit_units
                FROM picks
                WHERE status = ?
                ORDER BY settled_at DESC
                LIMIT ?
                """,
                (kind_filter, limit),
            ).fetchall()
        return [
            f"{r['market']}/{r['selection']} @ {r['bookmaker_odd']:.2f} "
            f"-> {float(r['profit_units'] or 0):+.2f}u"
            for r in rows
        ]

    # ── drawdown detection ──────────────────────────────────────────────

    async def _check_drawdown_and_alert(self) -> None:
        """Fire one drawdown alert per day when today P/L crosses threshold."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self._state.get_state("drawdown_alerted_date") == today:
            return

        snap = self._compute_scoreboard_snapshot()
        today_pl = snap["pl_emit"]
        baseline = self._state.get_state("bankroll_baseline", default=None)

        crossed = False
        baseline_units = None
        if baseline is not None:
            try:
                baseline_units = float(baseline)
                if baseline_units > 0:
                    pct = today_pl / baseline_units * 100.0
                    crossed = pct <= -DRAWDOWN_THRESHOLD_PCT
            except (ValueError, TypeError):
                pass
        if baseline_units is None or baseline_units <= 0:
            # Fallback: absolute-units threshold (e.g., -3.0u).
            crossed = today_pl <= DRAWDOWN_FALLBACK_UNITS

        if not crossed:
            return

        loss_streak = self._current_streak_count_if("loss")
        worst = self._worst_pick_today_summary()
        text = format_drawdown_alert(
            today_pl_units=today_pl,
            threshold_pct=DRAWDOWN_THRESHOLD_PCT,
            bankroll_baseline_units=baseline_units,
            n_settled_today=snap["n_settled"],
            loss_streak=loss_streak,
            worst_pick_summary=worst,
        )
        try:
            await self._bot.send_html(
                text, chat_id=self._alert_channel,
                disable_notification=False,  # drawdown wants attention
            )
            self.n_drawdown_alerts_sent += 1
            self._state.set_state("drawdown_alerted_date", today)
        except Exception as exc:  # noqa: BLE001
            self.n_failures += 1
            logger.warning("drawdown_alert_send_failed err=%s", exc)

    def _current_streak_count_if(self, kind: str) -> int:
        streak = self._state.get_state("current_streak", default=None) or {}
        if streak.get("kind") == kind:
            return int(streak.get("count", 0))
        return 0

    def _worst_pick_today_summary(self) -> str | None:
        today = datetime.now(timezone.utc).date().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT market, selection, bookmaker_odd, profit_units
                FROM picks
                WHERE status IN ('won','lost') AND DATE(emitted_at) = ?
                ORDER BY profit_units ASC
                LIMIT 1
                """,
                (today,),
            ).fetchone()
        if row is None or row["profit_units"] is None:
            return None
        return (
            f"{row['market']}/{row['selection']} @ "
            f"{row['bookmaker_odd']:.2f} "
            f"({float(row['profit_units']):+.2f}u)"
        )

    # ── scoreboard ──────────────────────────────────────────────────────

    async def update_scoreboard(self) -> None:
        """Refresh (or create) the pinned scoreboard for today.

        Day-rollover: when ``tg_state['scoreboard_date']`` is older than
        today (UTC), the existing pin is abandoned (best-effort unpin),
        per-day drawdown peak is reset, and a fresh message is sent and
        pinned.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        existing_id = self._state.get_state("scoreboard_msg_id")
        existing_date = self._state.get_state("scoreboard_date")
        rolled_over = (
            existing_id is not None
            and existing_date is not None
            and existing_date != today
        )

        if rolled_over:
            # New day — reset daily peak (drawdown is per-day) and try
            # to unpin yesterday's scoreboard. Failure is non-fatal.
            self._state.delete_state("today_peak_pl")
            self._state.delete_state("drawdown_alerted_date")
            await self._unpin_safely(int(existing_id))
            existing_id = None  # force re-post path below

        snapshot = self._compute_scoreboard_snapshot()
        text = format_scoreboard(
            date=snapshot["date"],
            n_emit=snapshot["n_emit"],
            n_placed=snapshot["n_placed"],
            n_pending=snapshot["n_pending"],
            n_settled=snapshot["n_settled"],
            n_won=snapshot["n_won"],
            pl_emit_units=snapshot["pl_emit"],
            pl_placed_units=snapshot["pl_placed"],
            drawdown_pct=snapshot["drawdown_pct"],
            updated_at=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        )

        if existing_id:
            try:
                await self._bot.edit_html(
                    chat_id=self._primary_channel_id,
                    message_id=int(existing_id),
                    text=text,
                )
                self.n_scoreboard_updates += 1
                return
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "scoreboard_edit_failed_reposting err=%s", exc,
                )
                # Fall through to repost.

        message_id = await self._bot.send_html(
            text, chat_id=self._primary_channel_id,
            disable_notification=True,
        )
        self._state.set_state("scoreboard_msg_id", message_id)
        self._state.set_state("scoreboard_date", today)
        # Best-effort pin so it stays at the top of the channel.
        await self._pin_safely(message_id)
        self.n_scoreboard_updates += 1

    async def _pin_safely(self, message_id: int) -> None:
        bot = getattr(self._bot, "bot", None)
        if bot is None:
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(self._primary_channel_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("scoreboard_pin_skipped err=%s", exc)

    async def _unpin_safely(self, message_id: int) -> None:
        bot = getattr(self._bot, "bot", None)
        if bot is None:
            return
        try:
            await bot.unpin_chat_message(
                chat_id=int(self._primary_channel_id),
                message_id=int(message_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("scoreboard_unpin_skipped err=%s", exc)

    def _compute_scoreboard_snapshot(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT
                  COUNT(*)                                            AS n_emit,
                  SUM(CASE WHEN placed_at_betano=1 THEN 1 ELSE 0 END) AS n_placed,
                  SUM(CASE WHEN status='pending'   THEN 1 ELSE 0 END) AS n_pending,
                  SUM(CASE WHEN status IN ('won','lost')
                           THEN 1 ELSE 0 END)                         AS n_settled,
                  SUM(CASE WHEN status='won'       THEN 1 ELSE 0 END) AS n_won,
                  COALESCE(SUM(CASE WHEN status IN ('won','lost')
                                    THEN profit_units END), 0)        AS pl_emit
                FROM picks
                WHERE DATE(emitted_at) = ?
                """,
                (today,),
            ).fetchone()
            placed_row = conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                      WHEN p.status='won'  AND p.placed_at_betano=1
                        THEN p.actual_stake_units * (p.betano_odd - 1)
                      WHEN p.status='lost' AND p.placed_at_betano=1
                        THEN -p.actual_stake_units
                      ELSE 0
                    END
                ), 0) AS pl_placed
                FROM picks p
                WHERE DATE(p.emitted_at) = ?
                """,
                (today,),
            ).fetchone()

        # Drawdown computed against today's peak P/L.
        peak_pl = self._state.get_state("today_peak_pl", default=0.0) or 0.0
        pl_emit = float(row["pl_emit"] or 0.0)
        if pl_emit > peak_pl:
            self._state.set_state("today_peak_pl", pl_emit)
            peak_pl = pl_emit
        drawdown_pct = (peak_pl - pl_emit) if peak_pl > 0 else 0.0

        return {
            "date": today,
            "n_emit": int(row["n_emit"] or 0),
            "n_placed": int(row["n_placed"] or 0),
            "n_pending": int(row["n_pending"] or 0),
            "n_settled": int(row["n_settled"] or 0),
            "n_won": int(row["n_won"] or 0),
            "pl_emit": pl_emit,
            "pl_placed": float(placed_row["pl_placed"] or 0.0),
            "drawdown_pct": drawdown_pct,
        }


# ── helpers ─────────────────────────────────────────────────────────────────


def _actual_profit_units(*, status: str, stake_pct: float, odd: float) -> float:
    """Operator's actual P/L in stake-percent units.

    Mirrors how ``picks.profit_units`` is computed in the grader:
    won → stake_pct * (odd - 1); lost → -stake_pct; void → 0.
    """
    if status == "won":
        return stake_pct * (odd - 1.0)
    if status == "lost":
        return -stake_pct
    return 0.0
