"""CLV (closing-line-value) drift sampler for Telegram Bot v2 (§G #1).

Periodic task that re-queries the bookmaker odd for open picks and:

- Records every sample into ``tg_clv_track`` (full trajectory available
  later for analyse_clv reports).
- When |drift_pct| exceeds the threshold AND the operator has not yet
  acted (no ``placed`` or ``skipped`` in ``tg_actions``), edits the
  original alert message in-place to append a CLV footer line. This
  surfaces market movement to the operator without spamming new
  messages.

Decoupling from data source: takes a pluggable ``odds_resolver``
callback so the sampler is testable in isolation and the operator can
wire in any source (Sportmonks live cache, The Odds API, a fixture
table, etc.).

Failure-safe end-to-end: resolver exceptions, edit failures, and missing
messages all log + skip — never raise into the watch loop.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from bip.evaluation.live.telegram_alerts import (
    classify_tier,
    format_clv_footer,
    format_pick_alert_with_tier,
)
from bip.evaluation.live.telegram_state import (
    Action,
    Role,
    TelegramState,
)
from bip.evaluation.live.value_detector import LivePick


logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────
#
# Drift threshold (absolute %) above which we annotate the original
# alert. Lower → more noisy edits; higher → only steam moves shown.
CLV_DRIFT_THRESHOLD_PCT: float = 5.0

# How far back (hours) to look for open picks worth sampling. Picks
# emitted long ago are stale — don't waste resolver calls on them.
CLV_LOOKBACK_HOURS: float = 3.0


# Resolver signature: (fixture_id, market, selection, bookmaker_id) -> odd | None
OddsResolver = Callable[
    [int, str, str, int],
    Awaitable[float | None],
]


class CLVSampler:
    """Periodic CLV-drift sampler + footer-edit task."""

    def __init__(
        self,
        *,
        bot: Any,
        state: TelegramState,
        db_path: Path,
        odds_resolver: OddsResolver,
        threshold_pct: float = CLV_DRIFT_THRESHOLD_PCT,
        lookback_hours: float = CLV_LOOKBACK_HOURS,
    ) -> None:
        self._bot = bot
        self._state = state
        self._db_path = Path(db_path)
        self._resolve_odd = odds_resolver
        self._threshold_pct = float(threshold_pct)
        self._lookback_hours = float(lookback_hours)
        self.n_samples = 0
        self.n_edits = 0
        self.n_resolver_errors = 0
        self.n_edit_errors = 0
        self._stopped = False

    # ── one-tick API ────────────────────────────────────────────────────

    async def tick(self) -> int:
        """Sample all open picks once. Returns number successfully sampled."""
        rows = self._fetch_open_picks()
        n_sampled = 0
        for row in rows:
            try:
                got = await self._sample_one(row)
                if got is not None:
                    n_sampled += 1
                    self.n_samples += 1
            except Exception as exc:  # noqa: BLE001
                self.n_resolver_errors += 1
                logger.warning(
                    "clv_sample_failed pick_id=%s err=%s",
                    row["id"], exc,
                )
        return n_sampled

    async def run_forever(self, *, period_seconds: float = 120.0) -> None:
        while not self._stopped:
            await self.tick()
            try:
                await asyncio.sleep(period_seconds)
            except asyncio.CancelledError:
                self._stopped = True
                return

    def stop(self) -> None:
        self._stopped = True

    # ── per-pick processing ─────────────────────────────────────────────

    def _fetch_open_picks(self) -> list[sqlite3.Row]:
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(hours=self._lookback_hours)
        ).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, fixture_id, minute, home_team, away_team,
                       market, selection, bookmaker_id, bookmaker_odd,
                       our_probability, fair_odd, edge_pct,
                       kelly_fraction_full, suggested_stake_pct,
                       market_description, snapshot_kind,
                       flagged_reason, logical_score
                FROM picks
                WHERE status = 'pending' AND emitted_at >= ?
                ORDER BY emitted_at DESC
                LIMIT 100
                """,
                (cutoff,),
            ).fetchall()
        return list(rows)

    async def _sample_one(self, row: sqlite3.Row) -> float | None:
        cur_odd = await self._resolve_odd(
            int(row["fixture_id"]),
            str(row["market"]),
            str(row["selection"]),
            int(row["bookmaker_id"]),
        )
        if cur_odd is None or cur_odd <= 0:
            return None
        emit_odd = float(row["bookmaker_odd"])
        try:
            drift_pct = self._state.record_clv_sample(
                pick_id=int(row["id"]),
                bookmaker_odd=float(cur_odd),
                emit_odd=emit_odd,
            )
        except ValueError:
            return None

        if abs(drift_pct) >= self._threshold_pct:
            await self._maybe_edit_footer(
                row=row, current_odd=float(cur_odd),
                emit_odd=emit_odd, drift_pct=drift_pct,
            )
        return drift_pct

    async def _maybe_edit_footer(
        self, *, row: sqlite3.Row, current_odd: float,
        emit_odd: float, drift_pct: float,
    ) -> None:
        pick_id = int(row["id"])
        # Skip if operator already acted — the message has lost its
        # keyboard and our edit would resurrect it.
        if (
            self._state.get_action(pick_id=pick_id, action=Action.PLACED)
            or self._state.get_action(pick_id=pick_id, action=Action.SKIPPED)
        ):
            return
        msg_ref = self._state.find_message(
            pick_id=pick_id, role=Role.PICK,
        )
        if msg_ref is None:
            return

        # Re-render the alert with the same tier the original used.
        pick = self._row_to_live_pick(row)
        tier = msg_ref.tier or classify_tier(pick)
        body = format_pick_alert_with_tier(pick, tier=tier)
        footer = format_clv_footer(
            emit_odd=emit_odd, current_odd=current_odd, drift_pct=drift_pct,
        )
        full_text = f"{body}\n\n{footer}"

        # Preserve interactivity — rebuild the keyboard for the same pick.
        # (Skipping if not enabled is implicit — bot.edit_html accepts
        # reply_markup=None.)
        try:
            from bip.core.telegram.keyboards import build_pick_keyboard
            keyboard = build_pick_keyboard(pick_id, tier=tier)
        except Exception:  # noqa: BLE001
            keyboard = None

        try:
            await self._bot.edit_html(
                chat_id=msg_ref.channel_id,
                message_id=msg_ref.message_id,
                text=full_text,
                reply_markup=keyboard,
            )
            self.n_edits += 1
        except Exception as exc:  # noqa: BLE001
            self.n_edit_errors += 1
            logger.warning(
                "clv_footer_edit_failed pick_id=%s err=%s",
                pick_id, exc,
            )

    @staticmethod
    def _row_to_live_pick(row: sqlite3.Row) -> LivePick:
        return LivePick(
            fixture_id=int(row["fixture_id"]),
            minute=int(row["minute"]),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            market=str(row["market"]),
            selection=str(row["selection"]),
            bookmaker_id=int(row["bookmaker_id"]),
            bookmaker_odd=float(row["bookmaker_odd"]),
            our_probability=float(row["our_probability"]),
            fair_odd=float(row["fair_odd"]),
            edge_pct=float(row["edge_pct"]),
            kelly_fraction_full=float(row["kelly_fraction_full"]),
            suggested_stake_pct=float(row["suggested_stake_pct"]),
            market_description=row["market_description"],
            snapshot_kind=str(row["snapshot_kind"]),
            flagged_reason=row["flagged_reason"],
            logical_score=float(row["logical_score"] or 1.0),
        )
