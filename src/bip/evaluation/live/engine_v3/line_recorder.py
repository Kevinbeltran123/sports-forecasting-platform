"""Market Line Recorder — Phase 1 deliverable from sec 10.

Persists 30-60s snapshots of every market line per fixture to local
parquet partitioned by date. This is the **opt-B truth source** from
sec 7.6: without ``/odds_history`` from Sportmonks, we have to grow
the dataset ourselves. Recording from day 1 means in 4-8 weeks we
have a counter-factual backtest substrate.

Schema (one row per (fixture, market, timestamp)):

    fixture_id        int
    timestamp_utc     datetime
    market_id         str
    side_a_decimal    float
    side_b_decimal    float | None
    line_value        float | None
    max_stake_cap     float
    bookmaker         str   (defaults to "betano")
    league_id         int | None

Partitioning: ``data/cache/live_lines/dt=YYYY-MM-DD/lines.parquet``.
Append per snapshot; readers reconstruct line history by sorting on
(fixture_id, market_id, timestamp_utc).

The recorder is intentionally thin — no schema validation downstream,
no schema migrations. Phase 2 will add a polars-based reader with the
proper schema-on-read semantics.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.evaluation.live.engine_v3.gsv import MarketLine, MarketSnapshot


DEFAULT_OUTPUT_ROOT = Path("data/cache/live_lines")


def _date_partition(ts: datetime) -> str:
    return ts.strftime("dt=%Y-%m-%d")


def _flatten_line(
    fixture_id: int,
    market_id: str,
    line: MarketLine,
    timestamp_utc: datetime,
    *,
    bookmaker: str,
    league_id: int | None,
) -> dict[str, Any]:
    return {
        "fixture_id": int(fixture_id),
        "timestamp_utc": timestamp_utc,
        "market_id": market_id,
        "side_a_decimal": float(line.side_a_decimal),
        "side_b_decimal": float(line.side_b_decimal) if line.side_b_decimal is not None else None,
        "line_value": float(line.line_value) if line.line_value is not None else None,
        "max_stake_cap": float(line.max_stake_cap),
        "bookmaker": bookmaker,
        "league_id": int(league_id) if league_id is not None else None,
    }


def snapshot_rows(
    fixture_id: int,
    markets: MarketSnapshot,
    timestamp_utc: datetime | None = None,
    *,
    bookmaker: str = "betano",
    league_id: int | None = None,
) -> list[dict[str, Any]]:
    """Flatten a ``MarketSnapshot`` to one dict per market line.

    Centralised so writer and tests share schema. ``timestamp_utc``
    defaults to ``datetime.now(timezone.utc)`` so the caller doesn't
    have to set it on every snapshot."""
    ts = timestamp_utc or datetime.now(timezone.utc)
    return [
        _flatten_line(fixture_id, mid, line, ts, bookmaker=bookmaker, league_id=league_id)
        for mid, line in markets.lines.items()
    ]


class LineRecorder:
    """Append-only parquet recorder.

    ``record(...)`` appends to a daily partition file. The implementation
    uses polars (already in the dependency set) and rewrites the daily
    file each call — fine at 30-60s cadence; a future optimisation is
    delta-write but we don't need it yet.
    """

    def __init__(
        self,
        *,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        bookmaker: str = "betano",
    ) -> None:
        self.output_root = Path(output_root)
        self.bookmaker = bookmaker
        self._buffer: list[dict[str, Any]] = []

    # ── public API ──────────────────────────────────────────────────────

    def record(
        self,
        fixture_id: int,
        markets: MarketSnapshot,
        *,
        timestamp_utc: datetime | None = None,
        league_id: int | None = None,
    ) -> int:
        """Append one snapshot. Returns row count written this call."""
        rows = snapshot_rows(
            fixture_id, markets, timestamp_utc,
            bookmaker=self.bookmaker, league_id=league_id,
        )
        if not rows:
            return 0
        self._buffer.extend(rows)
        return len(rows)

    def flush(self, timestamp_utc: datetime | None = None) -> Path | None:
        """Write the buffer to the day's parquet file. Returns the path
        written, or None when buffer is empty."""
        if not self._buffer:
            return None
        ts = timestamp_utc or datetime.now(timezone.utc)
        path = self._daily_path(ts)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_parquet(path, self._buffer)
        self._buffer.clear()
        return path

    def buffer_size(self) -> int:
        return len(self._buffer)

    # ── internals ───────────────────────────────────────────────────────

    def _daily_path(self, ts: datetime) -> Path:
        return self.output_root / _date_partition(ts) / "lines.parquet"

    def _write_parquet(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        import polars as pl

        new_df = pl.DataFrame(list(rows))
        if path.exists():
            try:
                existing = pl.read_parquet(path)
                combined = pl.concat([existing, new_df], how="diagonal_relaxed")
            except Exception:
                # corrupted file → start fresh; the day's snapshot history is best-effort
                combined = new_df
        else:
            combined = new_df
        combined.write_parquet(path)


__all__ = ["LineRecorder", "snapshot_rows", "DEFAULT_OUTPUT_ROOT"]
