"""Research dataset builder — single canonical Polars table per day.

Joins gsv_log + picks + gate_denials (aggregated per frame) + outcomes
into ONE wide DataFrame where each row is either (a) an allowed pick
with its frame context + outcome, OR (b) a frame that produced no
allowed picks (pick fields NULL). The denial rules fired in each frame
appear as an aggregated column so frame-level analyses ("what's the
typical denial reason at minute 70 with score 1-1?") work without
re-joining the denials parquet.

# Schema (one row per pick OR per pickless frame)

Frame-level columns (from gsv_log.parquet):
    fixture_id, state_version, frame_timestamp_utc, minute, period,
    home_team_id, away_team_id, home_goals, away_goals,
    score_dominant_team_id, score_dominant_losing,
    xg_diff, xg_total, xg_vs_score_divergence,
    xg_per_min_home_last_15, xg_per_min_away_last_15,
    pressing_intensity_home, pressing_intensity_away,
    numerical_advantage, total_yellows, total_corners,
    game_phase, gsv_json (full Pydantic blob, preserved for novel features)

Per-frame denial summary:
    n_denials_this_frame, denial_rules_fired (list[int])

Pick-level columns (from picks.parquet, NULL for pickless frames):
    thesis_id, archetype, thesis_layer, rule_id,
    family, direction, magnitude_pp, horizon_minutes,
    market_id, line_value, bookmaker_odd, kelly_full_pct,
    fair_prob, base_edge, signal_clarity, book_slowness,
    liquidity_score, conditional_variance, mes_score,
    confidence_prior, activated_at_minute

Outcome columns (from picks_outcomes.parquet, NULL when not graded):
    status, profit_units, settled_at

# Why ONE table and not nested

Polars-native operations (filter, group_by, join) work best on flat
tables. Nested types complicate every downstream query. The size
trade-off (gsv_json repeated for each pick on a frame) is ~5KB per
duplicate; with ~100 frames per fixture and 1-3 picks per frame, the
amortized cost is fine.

# Why left-join (preserve frames without picks)

Workflow #1 (archetype discovery) needs pickless frames — those are
the "silent moments" where no archetype fired but something interesting
might have happened. Inner-joining would erase them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.evaluation.live.engine_v3.shadow_logger import DEFAULT_SHADOW_ROOT

log = logging.getLogger("v3.research_dataset")


DEFAULT_RESEARCH_ROOT = Path("reports/v3/research")


@dataclass(frozen=True)
class BuildReport:
    """Diagnostics for the dataset build run."""

    n_frames: int
    n_picks: int
    n_picks_with_outcome: int
    n_denials: int
    n_partitions_loaded: int
    output_path: Path


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────


def _load_partitioned(
    root: Path, filename: str, date_range: tuple[datetime, datetime] | None,
) -> Any:
    import polars as pl

    if not root.exists():
        return pl.DataFrame()
    frames = []
    for part in sorted(root.iterdir()):
        if not part.is_dir() or not part.name.startswith("dt="):
            continue
        try:
            part_date = datetime.strptime(part.name[3:], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if date_range is not None:
            if part_date.date() < date_range[0].date():
                continue
            if part_date.date() > date_range[1].date():
                continue
        path = part / filename
        if not path.exists():
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# ──────────────────────────────────────────────────────────────────────
# GSV JSON → flat feature columns
# ──────────────────────────────────────────────────────────────────────


_GSV_FLAT_FEATURES = (
    "score_dominant_team_id", "score_dominant_losing",
    "xg_diff", "xg_total", "xg_vs_score_divergence",
    "xg_per_min_home_last_15", "xg_per_min_away_last_15",
    "pressing_intensity_home", "pressing_intensity_away",
    "numerical_advantage", "total_yellows", "total_corners",
    "game_phase",
)


def _flatten_gsv_blob(blob: str) -> dict[str, Any]:
    """Pull frequently-used features out of the gsv_json blob so they
    are columnar (fast filter/group_by). Returns a dict keyed by the
    column names in _GSV_FLAT_FEATURES. Missing fields → None.

    Doesn't fully parse the Pydantic model — direct JSON access for speed.
    The blob itself stays available in the row for novel-feature work.
    """
    import json

    try:
        data = json.loads(blob)
    except Exception:  # noqa: BLE001
        return {k: None for k in _GSV_FLAT_FEATURES}

    score = data.get("score") or {}
    xg = data.get("xg") or {}
    tactical = data.get("tactical") or {}
    numerical = data.get("numerical") or {}
    cards = data.get("cards") or {}
    corners = data.get("corners") or {}

    yellows = cards.get("yellows") or [0, 0]
    if isinstance(yellows, (list, tuple)) and len(yellows) >= 2:
        total_yellows = (yellows[0] or 0) + (yellows[1] or 0)
    else:
        total_yellows = 0

    total_corners = (
        (corners.get("corners_home") or 0)
        + (corners.get("corners_away") or 0)
    )

    return {
        "score_dominant_team_id": score.get("dominant_team_id"),
        "score_dominant_losing": score.get("dominant_losing"),
        "xg_diff": xg.get("xg_diff"),
        "xg_total": xg.get("xg_total"),
        "xg_vs_score_divergence": xg.get("xg_vs_score_divergence"),
        "xg_per_min_home_last_15": xg.get("xg_per_min_home_last_15"),
        "xg_per_min_away_last_15": xg.get("xg_per_min_away_last_15"),
        "pressing_intensity_home": tactical.get("pressing_intensity_home"),
        "pressing_intensity_away": tactical.get("pressing_intensity_away"),
        "numerical_advantage": numerical.get("numerical_advantage"),
        "total_yellows": int(total_yellows),
        "total_corners": int(total_corners),
        "game_phase": tactical.get("game_phase"),
    }


# ──────────────────────────────────────────────────────────────────────
# Build pipeline
# ──────────────────────────────────────────────────────────────────────


def build_research_dataset(
    *,
    shadow_root: Path | str = DEFAULT_SHADOW_ROOT,
    date_range: tuple[datetime, datetime] | None = None,
    output_root: Path | str = DEFAULT_RESEARCH_ROOT,
    include_gsv_json: bool = True,
) -> BuildReport:
    """Materialise the canonical research dataset.

    Reads from ``shadow_root``, writes to ``output_root``. Returns a
    BuildReport with counts the operator can sanity-check.

    ``include_gsv_json=False`` drops the full JSON column — useful when
    materialising a smaller "fast features only" subset for repeated
    notebook reloads.
    """
    import polars as pl

    shadow_root = Path(shadow_root)
    output_root = Path(output_root)

    gsv_df = _load_partitioned(shadow_root, "gsv_log.parquet", date_range)
    picks_df = _load_partitioned(shadow_root, "picks.parquet", date_range)
    denials_df = _load_partitioned(shadow_root, "gate_denials.parquet", date_range)
    outcomes_df = _load_partitioned(shadow_root, "picks_outcomes.parquet", date_range)

    n_partitions = _count_partitions(shadow_root, date_range)

    if gsv_df.is_empty():
        log.warning("research_dataset_no_gsv_log")
        # Empty dataset still writes a 0-row parquet so downstream
        # consumers don't crash on missing file. Use a placeholder schema
        # since polars rejects empty-schema parquet writes.
        out_path = _resolve_output_path(output_root, date_range)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"fixture_id": pl.Series([], dtype=pl.Int64)}).write_parquet(
            out_path
        )
        return BuildReport(
            n_frames=0, n_picks=0, n_picks_with_outcome=0, n_denials=0,
            n_partitions_loaded=n_partitions, output_path=out_path,
        )

    # 1. Flatten gsv_json into columnar features.
    # We extract each feature as a properly-typed column rather than
    # using a single Object column (parquet rejects Object dtype).
    blobs = gsv_df["gsv_json"].to_list()
    flat_dicts = [_flatten_gsv_blob(b) for b in blobs]
    flat_columns = {
        "score_dominant_team_id": pl.Series(
            "score_dominant_team_id",
            [d.get("score_dominant_team_id") for d in flat_dicts],
            dtype=pl.Int64,
        ),
        "score_dominant_losing": pl.Series(
            "score_dominant_losing",
            [d.get("score_dominant_losing") for d in flat_dicts],
            dtype=pl.Boolean,
        ),
        "xg_diff": pl.Series(
            "xg_diff",
            [d.get("xg_diff") for d in flat_dicts],
            dtype=pl.Float64,
        ),
        "xg_total": pl.Series(
            "xg_total",
            [d.get("xg_total") for d in flat_dicts],
            dtype=pl.Float64,
        ),
        "xg_vs_score_divergence": pl.Series(
            "xg_vs_score_divergence",
            [d.get("xg_vs_score_divergence") for d in flat_dicts],
            dtype=pl.Float64,
        ),
        "xg_per_min_home_last_15": pl.Series(
            "xg_per_min_home_last_15",
            [d.get("xg_per_min_home_last_15") for d in flat_dicts],
            dtype=pl.Float64,
        ),
        "xg_per_min_away_last_15": pl.Series(
            "xg_per_min_away_last_15",
            [d.get("xg_per_min_away_last_15") for d in flat_dicts],
            dtype=pl.Float64,
        ),
        "pressing_intensity_home": pl.Series(
            "pressing_intensity_home",
            [d.get("pressing_intensity_home") for d in flat_dicts],
            dtype=pl.Utf8,
        ),
        "pressing_intensity_away": pl.Series(
            "pressing_intensity_away",
            [d.get("pressing_intensity_away") for d in flat_dicts],
            dtype=pl.Utf8,
        ),
        "numerical_advantage": pl.Series(
            "numerical_advantage",
            [d.get("numerical_advantage") for d in flat_dicts],
            dtype=pl.Int64,
        ),
        "total_yellows": pl.Series(
            "total_yellows",
            [d.get("total_yellows") for d in flat_dicts],
            dtype=pl.Int64,
        ),
        "total_corners": pl.Series(
            "total_corners",
            [d.get("total_corners") for d in flat_dicts],
            dtype=pl.Int64,
        ),
        "game_phase": pl.Series(
            "game_phase",
            [d.get("game_phase") for d in flat_dicts],
            dtype=pl.Utf8,
        ),
    }
    gsv_flat = gsv_df.with_columns([s for s in flat_columns.values()])

    # 2. Rename frame columns to avoid collision with pick columns
    gsv_renamed = gsv_flat.rename({"timestamp_utc": "frame_timestamp_utc"})

    if not include_gsv_json:
        gsv_renamed = gsv_renamed.drop("gsv_json")

    # 3. Aggregate denials per (fixture_id, frame_timestamp_utc)
    if denials_df.is_empty():
        denials_per_frame = pl.DataFrame({
            "fixture_id": pl.Series([], dtype=pl.Int64),
            "frame_timestamp_utc": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "n_denials_this_frame": pl.Series([], dtype=pl.UInt32),
            "denial_rules_fired": pl.Series([], dtype=pl.List(pl.Int64)),
        })
    else:
        denials_per_frame = (
            denials_df
            .rename({"timestamp_utc": "frame_timestamp_utc"})
            .group_by(["fixture_id", "frame_timestamp_utc"])
            .agg([
                pl.len().alias("n_denials_this_frame"),
                pl.col("rule_number").unique().sort().alias("denial_rules_fired"),
            ])
        )

    # 4. Join picks with outcomes (left)
    if outcomes_df.is_empty() or picks_df.is_empty():
        picks_with_outcomes = picks_df if not picks_df.is_empty() else pl.DataFrame()
        if not picks_with_outcomes.is_empty():
            picks_with_outcomes = picks_with_outcomes.with_columns([
                pl.lit(None).cast(pl.Utf8).alias("status"),
                pl.lit(None).cast(pl.Float64).alias("profit_units"),
                pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("settled_at"),
            ])
    else:
        outcomes_renamed = outcomes_df.rename({
            "pick_timestamp_utc": "timestamp_utc",
            "bookmaker_odd": "bookmaker_odd_outcome",  # disambiguate
        })
        picks_with_outcomes = picks_df.join(
            outcomes_renamed.select([
                "fixture_id", "thesis_id", "market_id", "timestamp_utc",
                "status", "profit_units", "settled_at",
            ]),
            on=["fixture_id", "thesis_id", "market_id", "timestamp_utc"],
            how="left",
        )

    # 5. Final left-join: every frame with its picks (one row per pick)
    if picks_with_outcomes.is_empty():
        # Pickless dataset: just frames + denial counts
        dataset = gsv_renamed.join(
            denials_per_frame,
            on=["fixture_id", "frame_timestamp_utc"],
            how="left",
        )
    else:
        picks_renamed = picks_with_outcomes.rename({
            "timestamp_utc": "frame_timestamp_utc",
        })
        # Join picks → gsv (left from picks side keeps all picks)
        dataset = gsv_renamed.join(
            picks_renamed,
            on=["fixture_id", "frame_timestamp_utc"],
            how="left",
        ).join(
            denials_per_frame,
            on=["fixture_id", "frame_timestamp_utc"],
            how="left",
        )

    # 6. Persist
    out_path = _resolve_output_path(output_root, date_range)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(out_path)

    n_picks_with_outcome = 0
    if not picks_with_outcomes.is_empty() and "status" in picks_with_outcomes.columns:
        n_picks_with_outcome = picks_with_outcomes.filter(
            pl.col("status").is_in(["won", "lost", "void"])
        ).height

    return BuildReport(
        n_frames=gsv_df.height,
        n_picks=picks_df.height if not picks_df.is_empty() else 0,
        n_picks_with_outcome=n_picks_with_outcome,
        n_denials=denials_df.height if not denials_df.is_empty() else 0,
        n_partitions_loaded=n_partitions,
        output_path=out_path,
    )


def _resolve_output_path(
    output_root: Path, date_range: tuple[datetime, datetime] | None,
) -> Path:
    """Compute the output filename based on date range. Single-day
    range → dataset_YYYY-MM-DD.parquet. Multi-day → dataset_YYYY-MM-DD_to_YYYY-MM-DD.parquet."""
    if date_range is None:
        tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return output_root / f"dataset_{tag}.parquet"
    start, end = date_range
    if start.date() == end.date():
        return output_root / f"dataset_{start.strftime('%Y-%m-%d')}.parquet"
    return output_root / (
        f"dataset_{start.strftime('%Y-%m-%d')}_to_"
        f"{end.strftime('%Y-%m-%d')}.parquet"
    )


def _count_partitions(
    root: Path, date_range: tuple[datetime, datetime] | None,
) -> int:
    if not root.exists():
        return 0
    n = 0
    for part in root.iterdir():
        if not part.is_dir() or not part.name.startswith("dt="):
            continue
        try:
            part_date = datetime.strptime(part.name[3:], "%Y-%m-%d").replace(
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        if date_range is not None:
            if part_date.date() < date_range[0].date():
                continue
            if part_date.date() > date_range[1].date():
                continue
        n += 1
    return n


__all__ = [
    "BuildReport",
    "DEFAULT_RESEARCH_ROOT",
    "build_research_dataset",
]
