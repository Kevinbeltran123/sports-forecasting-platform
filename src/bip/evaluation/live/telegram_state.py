"""Persistent state for Telegram Bot v2.

Extends the existing ``picks.db`` (owned by ``PickTracker``) with additive
tables for the interactive bot:

- ``tg_messages``     — pick_id <-> Telegram message_id mapping per role
- ``tg_actions``      — operator decisions captured via inline buttons
- ``tg_state``        — scalar key/value bag (mute, scoreboard ptr, etc.)
- ``tg_burst_queue``  — picks queued during burst / mute / outage
- ``tg_clv_track``    — line-drift samples for open picks

All tables use ``CREATE TABLE IF NOT EXISTS`` and never modify existing
``picks`` / ``pick_decisions`` schemas — backwards compatible with
analyze_decisions, topn_filter, and discover_policies.

Restart contract: every piece of bot state lives here. The bot retains
no in-memory-only state across restarts — see §D of the v2 design memo.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.evaluation.live.pick_tracker import DEFAULT_DB_PATH


# ── Tunables: bandwidth governance (§E) ─────────────────────────────────────
#
# Higher capacity = more individual messages allowed in a burst before
# batching kicks in. Lower refill_seconds = more permissive long-term
# cadence.
#
# Safe envelope: capacity in [3, 8], refill_seconds in [4, 15]. Below /
# above either floods or starves under realistic jornada cadence.
BANDWIDTH_BUCKET_CAPACITY: int = 5
BANDWIDTH_REFILL_SECONDS: float = 6.0
BANDWIDTH_DIGEST_FLUSH_SECONDS: float = 60.0


# ── Action / role enums (string-typed for SQLite compatibility) ─────────────


class Role:
    PICK = "pick"
    OUTCOME = "outcome"
    REMINDER = "reminder"
    CLV_EDIT = "clv_edit"
    PROMOTED = "promoted"
    BURST_DIGEST = "burst_digest"
    SCOREBOARD = "scoreboard"


class Action:
    PLACED = "placed"
    SKIPPED = "skipped"
    REMINDED = "reminded"
    PROMOTED = "promoted"
    DISMISSED = "dismissed"
    MUTED_FIXTURE = "muted_fixture"


class BurstReason:
    MUTED = "muted"
    BURST = "burst"
    RATE_LIMIT = "rate_limit"
    OUTAGE = "outage"


_TG_SCHEMA = """
CREATE TABLE IF NOT EXISTS tg_messages (
    pick_id      INTEGER NOT NULL,
    channel_id   TEXT    NOT NULL,
    message_id   INTEGER NOT NULL,
    role         TEXT    NOT NULL,
    tier         INTEGER,
    sent_at      TEXT    NOT NULL,
    PRIMARY KEY (pick_id, role, channel_id)
);
CREATE INDEX IF NOT EXISTS idx_tg_messages_pick ON tg_messages (pick_id);
CREATE INDEX IF NOT EXISTS idx_tg_messages_role ON tg_messages (role);

CREATE TABLE IF NOT EXISTS tg_actions (
    pick_id          INTEGER NOT NULL,
    action           TEXT    NOT NULL,
    stake_pct        REAL,
    operator_user_id INTEGER NOT NULL,
    callback_id      TEXT    NOT NULL,
    acted_at         TEXT    NOT NULL,
    PRIMARY KEY (pick_id, action),
    UNIQUE (callback_id)
);
CREATE INDEX IF NOT EXISTS idx_tg_actions_user ON tg_actions (operator_user_id, acted_at);

CREATE TABLE IF NOT EXISTS tg_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tg_burst_queue (
    pick_id     INTEGER PRIMARY KEY,
    enqueued_at TEXT    NOT NULL,
    reason      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tg_clv_track (
    pick_id       INTEGER NOT NULL,
    sampled_at    TEXT    NOT NULL,
    bookmaker_odd REAL    NOT NULL,
    drift_pct     REAL    NOT NULL,
    PRIMARY KEY (pick_id, sampled_at)
);
CREATE INDEX IF NOT EXISTS idx_tg_clv_drift ON tg_clv_track (pick_id, drift_pct);
"""


# ── Dataclasses for typed reads ─────────────────────────────────────────────


@dataclass(frozen=True)
class MessageRef:
    pick_id: int
    channel_id: str
    message_id: int
    role: str
    tier: int | None
    sent_at: str


@dataclass(frozen=True)
class ActionRecord:
    pick_id: int
    action: str
    stake_pct: float | None
    operator_user_id: int
    callback_id: str
    acted_at: str


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelegramState:
    """Persistence layer for Telegram Bot v2 state.

    Co-located in ``picks.db`` to keep grader writes (``picks.status``)
    and bot reads (``tg_messages.message_id``) in a single transactional
    namespace.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_TG_SCHEMA)

    # ── tg_messages ─────────────────────────────────────────────────────

    def record_message(
        self,
        *,
        pick_id: int,
        channel_id: str,
        message_id: int,
        role: str,
        tier: int | None = None,
        sent_at: str | None = None,
    ) -> None:
        """Upsert a message reference. Idempotent on (pick_id, role, channel_id)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tg_messages
                  (pick_id, channel_id, message_id, role, tier, sent_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pick_id, role, channel_id) DO UPDATE SET
                  message_id = excluded.message_id,
                  tier       = excluded.tier,
                  sent_at    = excluded.sent_at
                """,
                (pick_id, str(channel_id), int(message_id), role,
                 int(tier) if tier is not None else None,
                 sent_at or _utc_iso()),
            )

    def find_message(
        self, *, pick_id: int, role: str = Role.PICK,
        channel_id: str | None = None,
    ) -> MessageRef | None:
        """Return the first matching message ref or None."""
        query = (
            "SELECT pick_id, channel_id, message_id, role, tier, sent_at "
            "FROM tg_messages WHERE pick_id = ? AND role = ?"
        )
        params: list[Any] = [pick_id, role]
        if channel_id is not None:
            query += " AND channel_id = ?"
            params.append(str(channel_id))
        query += " LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return MessageRef(
            pick_id=row["pick_id"],
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            role=row["role"],
            tier=row["tier"],
            sent_at=row["sent_at"],
        )

    def picks_awaiting_outcome_reply(self, limit: int = 100) -> list[int]:
        """Return pick_ids that have a 'pick' message but no 'outcome' message
        and whose picks.status has been graded.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.id AS pick_id
                FROM picks p
                JOIN tg_messages m
                  ON m.pick_id = p.id AND m.role = ?
                LEFT JOIN tg_messages r
                  ON r.pick_id = p.id AND r.role = ?
                WHERE p.status IN ('won', 'lost', 'void')
                  AND r.pick_id IS NULL
                ORDER BY p.settled_at
                LIMIT ?
                """,
                (Role.PICK, Role.OUTCOME, limit),
            ).fetchall()
        return [r["pick_id"] for r in rows]

    # ── tg_actions ──────────────────────────────────────────────────────

    def record_action(
        self,
        *,
        pick_id: int,
        action: str,
        operator_user_id: int,
        callback_id: str,
        stake_pct: float | None = None,
        acted_at: str | None = None,
    ) -> tuple[bool, ActionRecord | None]:
        """Atomically record an operator action.

        Returns ``(was_new, existing_row_or_None)``:

        - ``(True,  None)`` — first time this (pick_id, action) is recorded.
        - ``(False, ActionRecord)`` — already recorded; row carries the
          original action. Callers should treat this as "ack already, do
          not double-process".

        Idempotent across two redelivery modes:
        1. Same ``callback_id`` redelivered (TG network retry).
        2. Different ``callback_id`` for the same (pick_id, action) — the
           operator double-tapped the same button shape.

        Both collide on the PRIMARY KEY ``(pick_id, action)``.
        """
        acted_at = acted_at or _utc_iso()
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO tg_actions
                      (pick_id, action, stake_pct, operator_user_id,
                       callback_id, acted_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (pick_id, action, stake_pct, int(operator_user_id),
                     callback_id, acted_at),
                )
                return (True, None)
            except sqlite3.IntegrityError:
                row = conn.execute(
                    """
                    SELECT pick_id, action, stake_pct, operator_user_id,
                           callback_id, acted_at
                    FROM tg_actions
                    WHERE pick_id = ? AND action = ?
                    """,
                    (pick_id, action),
                ).fetchone()
                if row is None:
                    # Collision was on UNIQUE(callback_id), not the PK —
                    # look it up by callback_id.
                    row = conn.execute(
                        """
                        SELECT pick_id, action, stake_pct, operator_user_id,
                               callback_id, acted_at
                        FROM tg_actions WHERE callback_id = ?
                        """,
                        (callback_id,),
                    ).fetchone()
                return (False, _row_to_action(row) if row else None)

    def get_action(
        self, *, pick_id: int, action: str,
    ) -> ActionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT pick_id, action, stake_pct, operator_user_id,
                       callback_id, acted_at
                FROM tg_actions
                WHERE pick_id = ? AND action = ?
                """,
                (pick_id, action),
            ).fetchone()
        return _row_to_action(row) if row else None

    # ── tg_state ────────────────────────────────────────────────────────

    def set_state(self, key: str, value: Any) -> None:
        """Upsert a scalar state value. ``value`` is JSON-encoded for
        round-trip fidelity; primitives in ⇒ primitives out via
        ``get_state``.
        """
        payload = json.dumps(value)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tg_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value      = excluded.value,
                  updated_at = excluded.updated_at
                """,
                (key, payload, _utc_iso()),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM tg_state WHERE key = ?", (key,),
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def delete_state(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM tg_state WHERE key = ?", (key,))

    # Convenience: mute window ───────────────────────────────────────────

    def is_muted(self, *, now: datetime | None = None) -> bool:
        until = self.get_state("mute_until")
        if until is None:
            return False
        try:
            until_dt = datetime.fromisoformat(until)
        except (TypeError, ValueError):
            return False
        now = now or datetime.now(timezone.utc)
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        return now < until_dt

    def set_mute_until(self, until: datetime) -> None:
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        self.set_state("mute_until", until.isoformat())
        # Track the window's start for /missed reporting.
        self.set_state(
            "mute_started_at",
            datetime.now(timezone.utc).isoformat(),
        )

    def clear_mute(self) -> None:
        self.delete_state("mute_until")
        # mute_started_at is preserved so /missed can look back at the
        # most recent window even after /resume.

    def last_mute_window(self) -> tuple[str, str] | None:
        """Return (started_at_iso, until_iso) of the most recent mute,
        or None if no mute has ever been set.
        """
        started = self.get_state("mute_started_at")
        until = self.get_state("mute_until")
        if started is None or until is None:
            return None
        return (started, until)

    # Convenience: per-market mute ────────────────────────────────────────

    def set_market_mute(self, market: str, until: datetime) -> None:
        """Mute a specific market key (e.g. 'ou_2_5') until ``until``."""
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        muted = self.get_state("muted_markets", default={}) or {}
        muted[market] = until.isoformat()
        self.set_state("muted_markets", muted)

    def is_market_muted(
        self, market: str, *, now: datetime | None = None,
    ) -> bool:
        muted = self.get_state("muted_markets", default={}) or {}
        until_iso = muted.get(market)
        if until_iso is None:
            return False
        try:
            until_dt = datetime.fromisoformat(until_iso)
        except (TypeError, ValueError):
            return False
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        return now < until_dt

    def clear_market_mute(self, market: str) -> None:
        muted = self.get_state("muted_markets", default={}) or {}
        if market in muted:
            del muted[market]
            self.set_state("muted_markets", muted)

    def muted_markets(self) -> dict[str, str]:
        """Return active market mutes as {market: until_iso}, expired pruned."""
        muted = self.get_state("muted_markets", default={}) or {}
        now = datetime.now(timezone.utc)
        active: dict[str, str] = {}
        for market, until_iso in muted.items():
            try:
                until_dt = datetime.fromisoformat(until_iso)
                if until_dt.tzinfo is None:
                    until_dt = until_dt.replace(tzinfo=timezone.utc)
                if now < until_dt:
                    active[market] = until_iso
            except (TypeError, ValueError):
                continue
        return active

    # ── tg_burst_queue ──────────────────────────────────────────────────

    def enqueue_burst(self, pick_id: int, reason: str) -> bool:
        """Add pick to burst queue. Returns True if newly queued,
        False if already in queue.
        """
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO tg_burst_queue (pick_id, enqueued_at, reason)
                    VALUES (?, ?, ?)
                    """,
                    (pick_id, _utc_iso(), reason),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def drain_burst_queue(self, limit: int | None = None) -> list[int]:
        """Atomically pop up to ``limit`` queued pick_ids (oldest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pick_id FROM tg_burst_queue "
                "ORDER BY enqueued_at "
                + (f"LIMIT {int(limit)}" if limit else ""),
            ).fetchall()
            pick_ids = [r["pick_id"] for r in rows]
            if pick_ids:
                placeholders = ",".join("?" * len(pick_ids))
                conn.execute(
                    f"DELETE FROM tg_burst_queue WHERE pick_id IN ({placeholders})",
                    pick_ids,
                )
        return pick_ids

    def burst_queue_size(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM tg_burst_queue"
            ).fetchone()
        return int(row["n"]) if row else 0

    # ── tg_clv_track ────────────────────────────────────────────────────

    def record_clv_sample(
        self, *, pick_id: int, bookmaker_odd: float, emit_odd: float,
        sampled_at: str | None = None,
    ) -> float:
        """Record a CLV sample. Returns the drift_pct just recorded.

        drift_pct = ((current - emitted) / emitted) * 100
        """
        if emit_odd <= 0:
            raise ValueError("emit_odd must be positive")
        drift_pct = ((bookmaker_odd - emit_odd) / emit_odd) * 100.0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tg_clv_track
                  (pick_id, sampled_at, bookmaker_odd, drift_pct)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(pick_id, sampled_at) DO UPDATE SET
                  bookmaker_odd = excluded.bookmaker_odd,
                  drift_pct     = excluded.drift_pct
                """,
                (pick_id, sampled_at or _utc_iso(),
                 bookmaker_odd, drift_pct),
            )
        return drift_pct

    def latest_clv_drift(self, pick_id: int) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT drift_pct FROM tg_clv_track
                WHERE pick_id = ?
                ORDER BY sampled_at DESC
                LIMIT 1
                """,
                (pick_id,),
            ).fetchone()
        return float(row["drift_pct"]) if row else None


# ── Helpers ─────────────────────────────────────────────────────────────────


def _row_to_action(row: sqlite3.Row) -> ActionRecord:
    return ActionRecord(
        pick_id=row["pick_id"],
        action=row["action"],
        stake_pct=row["stake_pct"],
        operator_user_id=row["operator_user_id"],
        callback_id=row["callback_id"],
        acted_at=row["acted_at"],
    )
