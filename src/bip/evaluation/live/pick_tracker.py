"""SQLite pick tracker — persists every emitted LivePick + outcome.

The DB is the experimental log: without it, we can't tell post-trial
whether the system actually had edge or whether we just remember the
wins. Schema is deliberately flat, easy to query from a notebook.

Stable pick identity: a pick is uniquely identified by
``(fixture_id, market, selection, bookmaker_id, snapshot_minute_bucket)``
where ``snapshot_minute_bucket`` is the floor of (minute / 5) — picks
within the same 5-minute window count as the same pick (avoid emitting
the "Lanús X2" pick 30 times in 5 minutes).

In addition to the ``picks`` table, this module owns ``pick_decisions``,
a debug-grade audit log of EVERY decision the value detector makes —
emit, flag, drop. That table is the input for §E falsifiability metrics
(per-component breakdown, hit-rate-by-bucket, gate-rejection rate).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.evaluation.live.value_detector import LivePick


DEFAULT_DB_PATH = Path("data/cache/sportmonks/picks.db")
DEDUP_BUCKET_MINUTES = 5


_SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker_id INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    minute_bucket INTEGER NOT NULL,
    bookmaker_odd REAL NOT NULL,
    our_probability REAL NOT NULL,
    fair_odd REAL NOT NULL,
    edge_pct REAL NOT NULL,
    kelly_fraction_full REAL NOT NULL,
    suggested_stake_pct REAL NOT NULL,
    market_description TEXT,
    snapshot_kind TEXT NOT NULL,
    emitted_at TEXT NOT NULL,           -- ISO 8601 UTC
    flagged_reason TEXT,                -- sanity filter trip (NULL = clean)
    logical_score REAL,                  -- §D composite, [0,1]
    logical_components_json TEXT,        -- JSON: subscore breakdown
    confidence_half_width REAL,          -- predictor CI half-width
    -- Outcome columns, populated post-match
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | won | lost | void | unknown
    settled_at TEXT,
    profit_units REAL,
    -- Operator-tracked fields (manual placement)
    placed_at_betano BOOLEAN NOT NULL DEFAULT 0,
    betano_odd REAL,
    actual_stake_units REAL,
    notes TEXT,
    -- Composite uniqueness
    UNIQUE (fixture_id, market, selection, bookmaker_id, minute_bucket)
);

CREATE INDEX IF NOT EXISTS idx_picks_status ON picks (status);
CREATE INDEX IF NOT EXISTS idx_picks_fixture ON picks (fixture_id);
CREATE INDEX IF NOT EXISTS idx_picks_emitted_at ON picks (emitted_at);
CREATE INDEX IF NOT EXISTS idx_picks_flagged ON picks (flagged_reason);

-- Decision audit: one row per (market, selection, bookmaker_id) PER detector
-- pass, regardless of whether it ended in emit / flag / drop. Powers the
-- per-gate rejection-rate analysis from §E.
CREATE TABLE IF NOT EXISTS pick_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker_id INTEGER,
    bookmaker_odd REAL,
    our_probability REAL,
    edge_pct REAL,
    decision TEXT NOT NULL,         -- 'emit' | 'flag' | 'drop'
    drop_reason TEXT,                -- specific gate that fired
    sm_marginal REAL,                -- Sportmonks correct-score marginal
    valuebet_agrees INTEGER,         -- nullable bool
    informational_density REAL,
    logical_score REAL,
    logical_components_json TEXT,
    confidence_half_width REAL,
    pick_id INTEGER,                 -- FK into picks.id when decision='emit'
    snapshot_taken_at TEXT,
    audited_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_fixture ON pick_decisions (fixture_id);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON pick_decisions (decision);
CREATE INDEX IF NOT EXISTS idx_decisions_drop_reason ON pick_decisions (drop_reason);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial release.

    SQLite ALTER TABLE ADD COLUMN is idempotent in spirit but doesn't
    have IF NOT EXISTS. We use PRAGMA table_info to detect.
    """
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(picks)")
    }
    if "flagged_reason" not in existing_cols:
        conn.execute("ALTER TABLE picks ADD COLUMN flagged_reason TEXT")
    if "logical_score" not in existing_cols:
        conn.execute("ALTER TABLE picks ADD COLUMN logical_score REAL")
    if "logical_components_json" not in existing_cols:
        conn.execute("ALTER TABLE picks ADD COLUMN logical_components_json TEXT")
    if "confidence_half_width" not in existing_cols:
        conn.execute("ALTER TABLE picks ADD COLUMN confidence_half_width REAL")


@dataclass(frozen=True)
class TrackedPick:
    """A row from the picks table with all columns."""

    id: int
    fixture_id: int
    home_team: str
    away_team: str
    market: str
    selection: str
    bookmaker_id: int
    minute: int
    minute_bucket: int
    bookmaker_odd: float
    our_probability: float
    fair_odd: float
    edge_pct: float
    suggested_stake_pct: float
    snapshot_kind: str
    emitted_at: str
    status: str
    profit_units: float | None
    flagged_reason: str | None = None
    market_description: str | None = None
    logical_score: float | None = None
    confidence_half_width: float | None = None


class PickTracker:
    """Thin wrapper around a SQLite picks DB."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _migrate_schema(conn)

    # ── insert / update ─────────────────────────────────────────────────

    def record(self, pick: LivePick) -> tuple[int | None, bool]:
        """Insert pick if new, otherwise no-op.

        Returns ``(picks_id_or_None, is_new)``. ``is_new=False`` means a
        row with the same dedup key already exists.
        """
        bucket = pick.minute // DEDUP_BUCKET_MINUTES
        now_iso = datetime.now(timezone.utc).isoformat()
        components_json = (
            json.dumps(pick.logical_components, sort_keys=True)
            if pick.logical_components else None
        )
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO picks (
                        fixture_id, home_team, away_team, market, selection,
                        bookmaker_id, minute, minute_bucket, bookmaker_odd,
                        our_probability, fair_odd, edge_pct, kelly_fraction_full,
                        suggested_stake_pct, market_description, snapshot_kind,
                        emitted_at, flagged_reason,
                        logical_score, logical_components_json, confidence_half_width
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        pick.fixture_id, pick.home_team, pick.away_team,
                        pick.market, pick.selection, pick.bookmaker_id,
                        pick.minute, bucket, pick.bookmaker_odd,
                        pick.our_probability, pick.fair_odd, pick.edge_pct,
                        pick.kelly_fraction_full, pick.suggested_stake_pct,
                        pick.market_description, pick.snapshot_kind, now_iso,
                        pick.flagged_reason,
                        pick.logical_score, components_json,
                        pick.confidence_half_width,
                    ),
                )
                return cursor.lastrowid, True
        except sqlite3.IntegrityError:
            return None, False

    def record_decision(
        self,
        *,
        fixture_id: int,
        minute: int,
        market: str,
        selection: str,
        decision: str,
        bookmaker_id: int | None = None,
        bookmaker_odd: float | None = None,
        our_probability: float | None = None,
        edge_pct: float | None = None,
        drop_reason: str | None = None,
        sm_marginal: float | None = None,
        valuebet_agrees: bool | None = None,
        informational_density: float | None = None,
        logical_score: float | None = None,
        logical_components: dict[str, float] | None = None,
        confidence_half_width: float | None = None,
        pick_id: int | None = None,
        snapshot_taken_at: datetime | None = None,
    ) -> None:
        """Append one row to ``pick_decisions``.

        Always succeeds (no UNIQUE constraint on this table — append-only).
        Powers the §E falsifiability metrics: gate-rejection rates, hit-rate
        bucketing by edge or logical_score, per-component decomposition.
        """
        if decision not in ("emit", "flag", "drop"):
            raise ValueError(f"invalid decision: {decision}")
        components_json = (
            json.dumps(logical_components, sort_keys=True)
            if logical_components else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pick_decisions (
                    fixture_id, minute, market, selection,
                    bookmaker_id, bookmaker_odd, our_probability, edge_pct,
                    decision, drop_reason,
                    sm_marginal, valuebet_agrees, informational_density,
                    logical_score, logical_components_json,
                    confidence_half_width, pick_id, snapshot_taken_at,
                    audited_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fixture_id, minute, market, selection,
                    bookmaker_id, bookmaker_odd, our_probability, edge_pct,
                    decision, drop_reason,
                    sm_marginal,
                    None if valuebet_agrees is None else int(valuebet_agrees),
                    informational_density,
                    logical_score, components_json,
                    confidence_half_width, pick_id,
                    snapshot_taken_at.isoformat() if snapshot_taken_at else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def gate_rejection_rates(self, *, hours: int = 24) -> list[dict[str, Any]]:
        """Return per-drop_reason counts over the last N hours.

        Useful for §E "each gate ≥ 5% of total filter, ≤ 60%" check.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT drop_reason, COUNT(*) AS n
                FROM pick_decisions
                WHERE decision='drop'
                  AND audited_at >= datetime('now', ?)
                GROUP BY drop_reason
                ORDER BY n DESC
                """,
                (f"-{int(hours)} hours",),
            ).fetchall()
        return [{"drop_reason": r["drop_reason"] or "unflagged", "n": r["n"]}
                for r in rows]

    def update_outcome(
        self, pick_id: int, *,
        status: str, profit_units: float | None = None,
    ) -> None:
        if status not in ("won", "lost", "void", "unknown"):
            raise ValueError(f"Bad status: {status}")
        settled_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE picks
                SET status = ?, settled_at = ?, profit_units = ?
                WHERE id = ?
                """,
                (status, settled_at, profit_units, pick_id),
            )

    def mark_placed(
        self, pick_id: int, *,
        betano_odd: float, stake_units: float, notes: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE picks
                SET placed_at_betano = 1, betano_odd = ?,
                    actual_stake_units = ?, notes = ?
                WHERE id = ?
                """,
                (betano_odd, stake_units, notes, pick_id),
            )

    # ── queries ─────────────────────────────────────────────────────────

    def list_pending(self) -> list[TrackedPick]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM picks WHERE status='pending' ORDER BY emitted_at DESC"
            ).fetchall()
        return [_row_to_tracked(r) for r in rows]

    def list_for_fixture(self, fixture_id: int) -> list[TrackedPick]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM picks WHERE fixture_id=? ORDER BY emitted_at",
                (fixture_id,),
            ).fetchall()
        return [_row_to_tracked(r) for r in rows]

    def list_recent(self, limit: int = 50) -> list[TrackedPick]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM picks ORDER BY emitted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_tracked(r) for r in rows]

    def stats_summary(self) -> dict[str, Any]:
        """Aggregate ROI / win rate / counts. Excludes pending picks."""
        with self._connect() as conn:
            graded = conn.execute(
                """
                SELECT status, COUNT(*) as n, SUM(profit_units) as total_profit
                FROM picks
                WHERE status IN ('won', 'lost', 'void')
                GROUP BY status
                """
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) as n FROM picks"
            ).fetchone()
            placed = conn.execute(
                "SELECT COUNT(*) as n FROM picks WHERE placed_at_betano=1"
            ).fetchone()
            by_market = conn.execute(
                """
                SELECT market,
                       COUNT(*) as n,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                       COALESCE(SUM(profit_units), 0) as profit
                FROM picks
                WHERE status IN ('won', 'lost', 'void')
                GROUP BY market
                """
            ).fetchall()

        n_total_graded = sum(g["n"] for g in graded)
        n_won = sum(g["n"] for g in graded if g["status"] == "won")
        total_profit = sum(g["total_profit"] or 0 for g in graded)
        return {
            "n_total": total["n"],
            "n_graded": n_total_graded,
            "n_won": n_won,
            "n_placed_betano": placed["n"],
            "win_rate": (n_won / n_total_graded) if n_total_graded else None,
            "total_profit_units": total_profit,
            "roi_pct": (total_profit / n_total_graded * 100) if n_total_graded else None,
            "by_market": [
                {
                    "market": r["market"],
                    "n": r["n"],
                    "wins": r["wins"],
                    "win_rate": r["wins"] / r["n"] if r["n"] else 0,
                    "profit": r["profit"],
                    "roi_pct": (r["profit"] / r["n"] * 100) if r["n"] else 0,
                }
                for r in by_market
            ],
        }


def _row_to_tracked(row: sqlite3.Row) -> TrackedPick:
    keys = row.keys()
    return TrackedPick(
        id=row["id"], fixture_id=row["fixture_id"],
        home_team=row["home_team"], away_team=row["away_team"],
        market=row["market"], selection=row["selection"],
        bookmaker_id=row["bookmaker_id"], minute=row["minute"],
        minute_bucket=row["minute_bucket"],
        bookmaker_odd=row["bookmaker_odd"],
        our_probability=row["our_probability"],
        fair_odd=row["fair_odd"],
        edge_pct=row["edge_pct"],
        suggested_stake_pct=row["suggested_stake_pct"],
        snapshot_kind=row["snapshot_kind"],
        emitted_at=row["emitted_at"],
        status=row["status"],
        profit_units=row["profit_units"],
        flagged_reason=row["flagged_reason"] if "flagged_reason" in keys else None,
        market_description=row["market_description"],
        logical_score=row["logical_score"] if "logical_score" in keys else None,
        confidence_half_width=(
            row["confidence_half_width"] if "confidence_half_width" in keys else None
        ),
    )
