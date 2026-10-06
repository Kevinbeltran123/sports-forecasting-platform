"""LineRecorder smoke + parquet roundtrip tests.

Phase-1 deliverable verification: recorder must persist line history
in a format we can read back into polars for backtest reconstruction
(sec 7.6 opt-B)."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from bip.evaluation.live.engine_v3 import LineRecorder, MarketLine, MarketSnapshot
from bip.evaluation.live.engine_v3.line_recorder import snapshot_rows


@pytest.fixture
def sample_snapshot() -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    return MarketSnapshot(lines={
        "match_corners_over_10.5": MarketLine(
            market_id="match_corners_over_10.5",
            side_a_decimal=1.95, side_b_decimal=1.85,
            line_value=10.5, max_stake_cap=300.0,
            last_update_utc=now,
        ),
        "btts_yes": MarketLine(
            market_id="btts_yes",
            side_a_decimal=1.90, max_stake_cap=400.0,
            last_update_utc=now,
        ),
    })


def test_snapshot_rows_emits_one_per_line(sample_snapshot):
    rows = snapshot_rows(101, sample_snapshot)
    assert len(rows) == 2
    market_ids = {r["market_id"] for r in rows}
    assert market_ids == {"match_corners_over_10.5", "btts_yes"}
    for r in rows:
        assert r["fixture_id"] == 101
        assert "timestamp_utc" in r
        assert r["bookmaker"] == "betano"


def test_recorder_flush_creates_parquet(tmp_path, sample_snapshot):
    rec = LineRecorder(output_root=tmp_path)
    n = rec.record(101, sample_snapshot, league_id=384)
    assert n == 2
    assert rec.buffer_size() == 2
    path = rec.flush()
    assert path is not None
    assert path.exists()
    # Reload and verify
    df = pl.read_parquet(path)
    assert df.height == 2
    assert "fixture_id" in df.columns
    assert "league_id" in df.columns
    assert (df["league_id"] == 384).all()


def test_recorder_appends_to_existing_partition(tmp_path, sample_snapshot):
    rec = LineRecorder(output_root=tmp_path)
    rec.record(101, sample_snapshot)
    rec.flush()
    rec.record(102, sample_snapshot)
    path = rec.flush()
    assert path is not None
    df = pl.read_parquet(path)
    assert df.height == 4  # 2 fixtures × 2 markets


def test_recorder_flush_noop_when_empty(tmp_path):
    rec = LineRecorder(output_root=tmp_path)
    assert rec.flush() is None
