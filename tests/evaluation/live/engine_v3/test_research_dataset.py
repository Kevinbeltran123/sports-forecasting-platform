"""Tests for the research dataset builder.

Covers:
- Empty shadow root → 0-row output, no crash.
- gsv_log only (no picks, no denials, no outcomes) → 1 row per frame
  with NULL pick fields.
- Full pipeline (gsv + picks + denials + outcomes) → joined rows.
- Date range filtering at partition granularity.
- include_gsv_json=False drops the JSON column.
- Flattened GSV features populated from the JSON blob.
- Denial aggregation: n_denials_this_frame + denial_rules_fired list.
- BuildReport counts match the materialised output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import polars as pl
import pytest

from bip.evaluation.live.engine_v3.runtime.research_dataset import (
    BuildReport,
    _flatten_gsv_blob,
    build_research_dataset,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers — synthetic shadow root layout
# ──────────────────────────────────────────────────────────────────────


def _ts(day=12, hour=12, minute=0, second=0) -> datetime:
    return datetime(2026, 5, day, hour, minute, second, tzinfo=timezone.utc)


def _gsv_json_blob(
    *, dominant_losing=False, xg_diff=0.5, total_yellows=2, game_phase="balanced",
) -> str:
    """Minimal GSV blob that exercises _flatten_gsv_blob."""
    return json.dumps({
        "score": {
            "dominant_team_id": 100, "dominant_losing": dominant_losing,
            "home_goals": 1, "away_goals": 0,
        },
        "xg": {
            "xg_diff": xg_diff, "xg_total": 1.8,
            "xg_vs_score_divergence": 0.4,
            "xg_per_min_home_last_15": 0.02,
            "xg_per_min_away_last_15": 0.01,
        },
        "tactical": {
            "pressing_intensity_home": "high",
            "pressing_intensity_away": "medium",
            "game_phase": game_phase,
        },
        "numerical": {"numerical_advantage": 0, "red_cards_home": 0, "red_cards_away": 0},
        "cards": {"yellows": [total_yellows // 2, total_yellows - total_yellows // 2]},
        "corners": {"corners_home": 4, "corners_away": 3},
    })


def _write_gsv_log(root: Path, day: int, rows: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "gsv_log.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _gsv_row(*, fixture_id=12345, sv=1, ts=None, minute=30, gsv_json=None):
    return {
        "fixture_id": fixture_id,
        "state_version": sv,
        "timestamp_utc": ts or _ts(minute=minute),
        "home_team_id": 100,
        "away_team_id": 200,
        "minute": minute,
        "period": "1H",
        "home_goals": 1,
        "away_goals": 0,
        "gsv_json": gsv_json or _gsv_json_blob(),
    }


def _write_picks(root: Path, day: int, rows: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(part / "picks.parquet")
    return part / "picks.parquet"


def _write_denials(root: Path, day: int, rows: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(part / "gate_denials.parquet")
    return part / "gate_denials.parquet"


def _write_outcomes(root: Path, day: int, rows: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(part / "picks_outcomes.parquet")
    return part / "picks_outcomes.parquet"


# ──────────────────────────────────────────────────────────────────────
# _flatten_gsv_blob
# ──────────────────────────────────────────────────────────────────────


def test_flatten_gsv_pulls_score_and_xg():
    blob = _gsv_json_blob(dominant_losing=True, xg_diff=1.2)
    flat = _flatten_gsv_blob(blob)
    assert flat["score_dominant_losing"] is True
    assert flat["xg_diff"] == 1.2
    assert flat["total_yellows"] == 2  # 1 + 1 from default _gsv_json_blob


def test_flatten_gsv_handles_malformed_json():
    """Bad JSON → all-None dict, no crash."""
    flat = _flatten_gsv_blob('{"score":}')  # malformed
    assert all(v is None for v in flat.values())


def test_flatten_gsv_handles_missing_sections():
    """If GSV blob misses sections, flat dict has Nones."""
    flat = _flatten_gsv_blob('{}')
    assert flat["xg_diff"] is None
    assert flat["game_phase"] is None
    assert flat["total_yellows"] == 0
    assert flat["total_corners"] == 0


# ──────────────────────────────────────────────────────────────────────
# Empty / degenerate paths
# ──────────────────────────────────────────────────────────────────────


def test_empty_root_returns_zero_frames(tmp_path):
    report = build_research_dataset(
        shadow_root=tmp_path / "empty",
        output_root=tmp_path / "out",
    )
    assert report.n_frames == 0
    assert report.n_picks == 0
    assert report.output_path.exists()  # writes 0-row parquet


def test_gsv_only_no_picks_one_row_per_frame(tmp_path):
    """3 frames, no picks/denials/outcomes → 3 rows with NULL pick fields."""
    _write_gsv_log(tmp_path, day=12, rows=[
        _gsv_row(sv=1, minute=20),
        _gsv_row(sv=2, minute=30),
        _gsv_row(sv=3, minute=40),
    ])
    out_dir = tmp_path / "out"
    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=out_dir,
    )
    assert report.n_frames == 3
    df = pl.read_parquet(report.output_path)
    assert df.height == 3


# ──────────────────────────────────────────────────────────────────────
# Full join
# ──────────────────────────────────────────────────────────────────────


def _pick_row(*, fixture_id=12345, ts=None, thesis_id="t1",
              market_id="match_goals_over_2.5"):
    ts = ts or _ts(minute=30)
    return {
        "fixture_id": fixture_id,
        "timestamp_utc": ts,
        "thesis_id": thesis_id,
        "archetype": "regression_to_xg",
        "thesis_layer": "rule",
        "rule_id": "A5",
        "family": "goals",
        "direction": "over",
        "magnitude_pp": 0.07,
        "horizon_minutes": 30,
        "market_id": market_id,
        "line_value": 2.5,
        "bookmaker_odd": 1.95,
        "kelly_full_pct": 0.12,
        "fair_prob": 0.55,
        "base_edge": 0.04,
        "signal_clarity": 0.65,
        "book_slowness": 0.50,
        "liquidity_score": 0.80,
        "conditional_variance": 0.10,
        "mes_score": 0.72,
        "confidence_prior": 0.55,
        "activated_at_minute": 30,
    }


def _outcome_row(*, fixture_id=12345, ts=None, thesis_id="t1",
                 market_id="match_goals_over_2.5", status="won"):
    return {
        "fixture_id": fixture_id,
        "thesis_id": thesis_id,
        "market_id": market_id,
        "pick_timestamp_utc": ts or _ts(minute=30),
        "status": status,
        "profit_units": 0.95 if status == "won" else -1.0,
        "bookmaker_odd": 1.95,
        "settled_at": _ts(hour=15),
    }


def test_full_join_picks_and_outcomes_present(tmp_path):
    """3 frames + 1 pick at frame 2 + outcome for that pick.
    Expected output: 3 rows (1 per frame), the middle one with all
    pick + outcome fields populated."""
    frame_ts = [_ts(minute=20), _ts(minute=30), _ts(minute=40)]
    _write_gsv_log(tmp_path, day=12, rows=[
        _gsv_row(sv=i+1, ts=t, minute=20+i*10)
        for i, t in enumerate(frame_ts)
    ])
    _write_picks(tmp_path, day=12, rows=[
        _pick_row(ts=frame_ts[1]),  # pick at minute 30
    ])
    _write_outcomes(tmp_path, day=12, rows=[
        _outcome_row(ts=frame_ts[1], status="won"),
    ])

    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=tmp_path / "out",
    )
    assert report.n_frames == 3
    assert report.n_picks == 1
    assert report.n_picks_with_outcome == 1

    df = pl.read_parquet(report.output_path)
    # Frame at minute 30 must have status="won" and profit_units > 0
    row_30 = df.filter(pl.col("minute") == 30).row(0, named=True)
    assert row_30["status"] == "won"
    assert row_30["profit_units"] == pytest.approx(0.95)
    assert row_30["archetype"] == "regression_to_xg"
    # Frame at minute 20 has no pick → NULL status
    row_20 = df.filter(pl.col("minute") == 20).row(0, named=True)
    assert row_20["status"] is None


def test_denials_aggregated_per_frame(tmp_path):
    """2 denials at the same frame → n_denials_this_frame=2,
    denial_rules_fired = sorted unique list."""
    frame_ts = _ts(minute=30)
    _write_gsv_log(tmp_path, day=12, rows=[_gsv_row(sv=1, ts=frame_ts, minute=30)])
    _write_denials(tmp_path, day=12, rows=[
        {"fixture_id": 12345, "timestamp_utc": frame_ts,
         "thesis_id": "t1", "archetype": "x", "family": "goals",
         "market_id": "m1", "rule_number": 5, "reason": "mes",
         "mes_score": 0.4, "direction": "over"},
        {"fixture_id": 12345, "timestamp_utc": frame_ts,
         "thesis_id": "t2", "archetype": "y", "family": "corners",
         "market_id": "m2", "rule_number": 7, "reason": "liq",
         "mes_score": 0.5, "direction": "over"},
    ])
    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=tmp_path / "out",
    )
    df = pl.read_parquet(report.output_path)
    row = df.row(0, named=True)
    assert row["n_denials_this_frame"] == 2
    assert sorted(row["denial_rules_fired"]) == [5, 7]


def test_no_gsv_json_drops_blob_column(tmp_path):
    _write_gsv_log(tmp_path, day=12, rows=[_gsv_row()])
    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=tmp_path / "out",
        include_gsv_json=False,
    )
    df = pl.read_parquet(report.output_path)
    assert "gsv_json" not in df.columns


def test_date_range_filters_partitions(tmp_path):
    _write_gsv_log(tmp_path, day=11, rows=[_gsv_row(fixture_id=1, ts=_ts(day=11))])
    _write_gsv_log(tmp_path, day=12, rows=[_gsv_row(fixture_id=2, ts=_ts(day=12))])
    _write_gsv_log(tmp_path, day=13, rows=[_gsv_row(fixture_id=3, ts=_ts(day=13))])

    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=tmp_path / "out",
    )
    assert report.n_frames == 1  # only day=12


def test_build_report_counts_match_output(tmp_path):
    """The BuildReport's counts must exactly match the materialised parquet."""
    _write_gsv_log(tmp_path, day=12, rows=[_gsv_row(sv=i, ts=_ts(minute=i)) for i in range(5)])
    _write_picks(tmp_path, day=12, rows=[
        _pick_row(ts=_ts(minute=1)),
        _pick_row(ts=_ts(minute=3), thesis_id="t2"),
    ])
    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=tmp_path / "out",
    )
    assert report.n_frames == 5
    assert report.n_picks == 2
    df = pl.read_parquet(report.output_path)
    # 5 frames, 2 with picks → 5 rows total (one row per frame, picks left-joined)
    assert df.height == 5


def test_flattened_features_in_dataset(tmp_path):
    """The dataset must include columns derived from gsv_json (not just the blob)."""
    _write_gsv_log(tmp_path, day=12, rows=[_gsv_row(
        gsv_json=_gsv_json_blob(dominant_losing=True, xg_diff=2.5, game_phase="open_attacking"),
    )])
    report = build_research_dataset(
        shadow_root=tmp_path,
        date_range=(_ts(day=12), _ts(day=12)),
        output_root=tmp_path / "out",
    )
    df = pl.read_parquet(report.output_path)
    row = df.row(0, named=True)
    assert row["score_dominant_losing"] is True
    assert row["xg_diff"] == pytest.approx(2.5)
    assert row["game_phase"] == "open_attacking"
