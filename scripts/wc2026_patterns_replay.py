"""WC2026 pattern-rules replay CLI — shadow-mode batch evaluation.

Reads a CSV of odds snapshots and replays them through the
``WC2026PatternRunner``, writing all triggered picks to the
date-partitioned parquet under ``data/cache/wc2026_patterns/``.

Input CSV schema (one row per (fixture, market_snapshot)):

    fixture_id, stage, market_family, selection, decimal_odds,
    minute, home_goals_ht, away_goals_ht

Example CSV row:

    wc2026_grp_000,live,goals,under_2_5,1.25,46,0,0

USAGE:
    uv run python scripts/wc2026_patterns_replay.py \\
        --input data/cache/wc2026_patterns/replay_input.csv \\
        --output-root data/cache/wc2026_patterns

The script prints a summary table of trigger counts per rule. Useful for:
- Validating the runner end-to-end against synthetic streams.
- Counter-factual backtests over recorded line history once the
  Sportmonks recorder ingests WC2026 fixtures live.

EXITS:
  0 — replay completed (even if no picks fired)
  1 — input file missing or malformed
  2 — lock JSON missing
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_pattern_runner import (
    DEFAULT_LOCK_PATH,
    DEFAULT_PATTERN_OUTPUT_ROOT,
    PatternPickRecorder,
    WC2026PatternRunner,
)
from bip.evaluation.live.engine_v3.wc2026_patterns import OddsSnapshot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, required=True, help="CSV input file")
    p.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Path to WC2026 lock JSON (default: production lock)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PATTERN_OUTPUT_ROOT,
        help="Parquet output root (default: data/cache/wc2026_patterns)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing parquet (dry-run; summary only)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _row_to_snapshot(row: dict) -> OddsSnapshot:
    return OddsSnapshot(
        stage=row["stage"],
        market_family=MarketFamily(row["market_family"]),
        selection=row["selection"],
        decimal_odds=float(row["decimal_odds"]),
        minute=int(row["minute"]) if row.get("minute") not in (None, "") else None,
        home_goals_ht=(
            int(row["home_goals_ht"])
            if row.get("home_goals_ht") not in (None, "")
            else None
        ),
        away_goals_ht=(
            int(row["away_goals_ht"])
            if row.get("away_goals_ht") not in (None, "")
            else None
        ),
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("wc2026_patterns_replay")

    if not args.input.exists():
        log.error("input not found: %s", args.input)
        return 1
    if not args.lock.exists():
        log.error("lock not found: %s", args.lock)
        return 2

    try:
        df = pl.read_csv(args.input)
    except Exception as exc:
        log.error("failed to parse CSV %s: %s", args.input, exc)
        return 1

    runner = WC2026PatternRunner.from_lock_json(args.lock)
    recorder = PatternPickRecorder(output_root=args.output_root)
    log.info("loaded %d fixtures from %s", len(runner.contexts), args.lock)
    log.info("replaying %d odds rows from %s", df.height, args.input)

    rule_counts: Counter[str] = Counter()
    unregistered = 0
    now = datetime.now(timezone.utc)

    for row in df.iter_rows(named=True):
        fixture_id = str(row["fixture_id"])
        if fixture_id not in runner.contexts:
            unregistered += 1
            continue
        snapshot = _row_to_snapshot(row)
        picks = runner.evaluate(fixture_id, snapshot, now_utc=now)
        for pick in picks:
            rule_counts[pick.rule_id] += 1
        recorder.record(picks)

    path: Path | None = None
    if not args.no_write:
        path = recorder.flush(ts=now)

    print()
    print("=== WC2026 Pattern Replay Summary ===")
    print(f"input rows:           {df.height}")
    print(f"unregistered fixtures: {unregistered}")
    print(f"picks total:           {sum(rule_counts.values())}")
    for rule_id in sorted(rule_counts):
        print(f"  {rule_id}: {rule_counts[rule_id]}")
    if path is not None:
        print(f"wrote parquet:         {path}")
    elif args.no_write:
        print("dry-run: parquet not written")
    else:
        print("no picks to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
