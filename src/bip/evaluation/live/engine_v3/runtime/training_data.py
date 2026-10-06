"""GSV-thesis-outcome attribution: build real training pairs from shadow data.

The pattern layer's fit() takes ``list[tuple[GSV, Thesis]]``. Until now,
those pairs only came from synthetic data (50 anti-Napoli + 200 generic
realistic states). The ShadowLogger started persisting full GSVs in
T1.1, but no plumbing reconstructed real (GSV, fired thesis) pairs —
so weekly refit was effectively a no-op against operator's actual
shadow distribution.

This module closes the join. It reads:

- ``gsv_log.parquet``  : full GSVs (one per pipeline frame)
- ``picks.parquet``    : emitted picks (archetype, market_id, etc.)
- ``picks_outcomes.parquet`` (optional) : settlement (won/lost/void/pending)

And produces:

- ``RealPatternPair`` records — one per (GSV, fired-archetype) pair,
  with the original Thesis pydantic re-derived by RUNNING the rule
  layer on the recovered GSV.
- Optionally tagged with outcome status + profit_units, so the refit
  can filter to "winning archetypes only" (Phase 5 enhancement).

# Why we re-derive the Thesis instead of reading it from disk

The rule layer is deterministic: given a GSV, ``generate_theses(gsv)``
returns the same Thesis pydantic every time. We persist enough columnar
data on the picks parquet to confirm "this archetype fired here", but
the full Thesis (with premise predicates + invalidation triggers + causal
chain + horizon) is only reconstructable by running the rule layer.

Persisting the full Thesis JSON would be redundant — and worse, it
would lock us into a schema where evolving the rule layer breaks the
parquet. Re-deriving on load means rule-layer changes propagate
naturally through training.

# Cold-start safety

The pattern layer's ``min_samples_active`` floor (50) means refit only
triggers when we have ≥50 (GSV, thesis) pairs. With realistic shadow
volumes (~50 picks/jornada), that's about a week of operations.

If the join produces fewer than the floor, callers should fall back
to the existing synthetic generators in ``fit_pattern_layer.py``.
``build_real_pattern_pairs`` returns an empty list rather than failing
when there's no data.

# Mismatch handling

A GSV without a matching pick row (most pipeline frames are no-pick)
contributes nothing. A pick row without a matching GSV (rare —
happens if gsv_log.parquet was disabled mid-jornada) is dropped with
a structured warning. We never silently silently fabricate pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bip.evaluation.live.engine_v3.archetypes import generate_theses
from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.shadow_logger import (
    DEFAULT_SHADOW_ROOT,
    load_shadow_gsvs,
)
from bip.evaluation.live.engine_v3.thesis import Thesis, ThesisArchetype

log = logging.getLogger("v3.training_data")


@dataclass(frozen=True)
class RealPatternPair:
    """One (GSV, Thesis) pair derived from real shadow data, plus
    optional outcome attribution for filtering.
    """

    gsv: GameStateVector
    thesis: Thesis
    archetype: ThesisArchetype
    fixture_id: int
    pick_timestamp_utc: datetime
    market_id: str
    outcome_status: str | None = None  # "won" | "lost" | "void" | "pending" | None
    profit_units: float | None = None


@dataclass(frozen=True)
class JoinReport:
    """Diagnostics from a join run. Lets callers decide whether the
    sample is large enough to refit."""

    n_gsvs_loaded: int
    n_picks_loaded: int
    n_outcomes_loaded: int
    n_pairs_built: int
    n_picks_without_gsv: int  # picks whose (fixture, ts) didn't match a GSV row
    n_pairs_with_outcome: int
    n_pairs_won: int
    n_pairs_lost: int
    n_pairs_void: int
    n_pairs_pending: int
    archetype_distribution: dict[str, int] = field(default_factory=dict)


def build_real_pattern_pairs(
    shadow_root: Path | str = DEFAULT_SHADOW_ROOT,
    *,
    date_range: tuple[datetime, datetime] | None = None,
    require_outcome: bool = False,
    only_won: bool = False,
) -> tuple[list[RealPatternPair], JoinReport]:
    """Materialise (GSV, Thesis) pairs from shadow logs.

    ``require_outcome``  : drop pairs whose pick has no settled outcome
                           (status not in {won, lost, void}).
    ``only_won``         : keep only winning pairs (implies require_outcome).
                           Useful for "outcome-aware refit" — let the
                           kNN learn from theses that historically WORKED.

    Returns ``(pairs, report)``. ``report`` always populated even on
    empty data.
    """
    import polars as pl

    shadow_root = Path(shadow_root)

    gsvs, gsv_failures = load_shadow_gsvs(
        date_range=date_range, output_root=shadow_root
    )
    n_gsvs_loaded = len(gsvs)
    if gsv_failures:
        log.warning(
            "training_data_gsv_load_failures n=%d first_path=%s",
            len(gsv_failures),
            str(gsv_failures[0].path),
        )

    # Index GSVs by (fixture_id, timestamp_utc) for O(1) lookup.
    gsv_by_key: dict[tuple[int, datetime], GameStateVector] = {}
    for gsv in gsvs:
        key = (int(gsv.fixture_id), gsv.timestamp_utc)
        gsv_by_key[key] = gsv

    picks_df = _load_picks(shadow_root, date_range)
    n_picks_loaded = picks_df.height if not picks_df.is_empty() else 0

    outcomes_df = _load_outcomes(shadow_root, date_range)
    n_outcomes_loaded = outcomes_df.height if not outcomes_df.is_empty() else 0

    if picks_df.is_empty():
        return [], JoinReport(
            n_gsvs_loaded=n_gsvs_loaded,
            n_picks_loaded=0,
            n_outcomes_loaded=n_outcomes_loaded,
            n_pairs_built=0,
            n_picks_without_gsv=0,
            n_pairs_with_outcome=0,
            n_pairs_won=0,
            n_pairs_lost=0,
            n_pairs_void=0,
            n_pairs_pending=0,
        )

    # Index outcomes by canonical key for O(1) lookup during pair build.
    outcome_by_key: dict[tuple[int, str, str, datetime], dict[str, Any]] = {}
    if not outcomes_df.is_empty():
        for row in outcomes_df.iter_rows(named=True):
            key = (
                int(row["fixture_id"]),
                str(row["thesis_id"]),
                str(row["market_id"]),
                row["pick_timestamp_utc"],
            )
            outcome_by_key[key] = row

    pairs: list[RealPatternPair] = []
    n_picks_without_gsv = 0
    archetype_distribution: dict[str, int] = {}
    n_won = n_lost = n_void = n_pending = 0
    n_pairs_with_outcome = 0

    for pick_row in picks_df.iter_rows(named=True):
        fixture_id = int(pick_row["fixture_id"])
        pick_ts = pick_row["timestamp_utc"]
        gsv = gsv_by_key.get((fixture_id, pick_ts))
        if gsv is None:
            n_picks_without_gsv += 1
            continue

        # Re-derive theses for THIS gsv. The rule layer is deterministic;
        # the same archetype that fired in production will be present here.
        derived_theses = generate_theses(gsv)
        archetype_str = pick_row.get("archetype")
        thesis = _select_matching_thesis(derived_theses, archetype_str)
        if thesis is None:
            # The archetype that fired in production is no longer detected
            # by the current rule layer — likely a code change since the
            # pick was emitted. Drop with a structured warning.
            log.warning(
                "training_data_archetype_mismatch fixture=%d archetype=%s "
                "n_derived=%d",
                fixture_id,
                archetype_str,
                len(derived_theses),
            )
            continue

        # Outcome lookup (if available)
        outcome_status: str | None = None
        profit_units: float | None = None
        outcome_key = (
            fixture_id,
            str(pick_row["thesis_id"]),
            str(pick_row["market_id"]),
            pick_ts,
        )
        outcome_row = outcome_by_key.get(outcome_key)
        if outcome_row is not None:
            outcome_status = str(outcome_row["status"])
            profit_units = float(outcome_row["profit_units"])
            n_pairs_with_outcome += 1
            if outcome_status == "won":
                n_won += 1
            elif outcome_status == "lost":
                n_lost += 1
            elif outcome_status == "void":
                n_void += 1
            elif outcome_status == "pending":
                n_pending += 1

        # Apply caller-requested filters
        if require_outcome and outcome_status not in {"won", "lost", "void"}:
            continue
        if only_won and outcome_status != "won":
            continue

        archetype_distribution[archetype_str] = (
            archetype_distribution.get(archetype_str, 0) + 1
        )
        pairs.append(
            RealPatternPair(
                gsv=gsv,
                thesis=thesis,
                archetype=thesis.archetype,
                fixture_id=fixture_id,
                pick_timestamp_utc=pick_ts,
                market_id=str(pick_row["market_id"]),
                outcome_status=outcome_status,
                profit_units=profit_units,
            )
        )

    report = JoinReport(
        n_gsvs_loaded=n_gsvs_loaded,
        n_picks_loaded=n_picks_loaded,
        n_outcomes_loaded=n_outcomes_loaded,
        n_pairs_built=len(pairs),
        n_picks_without_gsv=n_picks_without_gsv,
        n_pairs_with_outcome=n_pairs_with_outcome,
        n_pairs_won=n_won,
        n_pairs_lost=n_lost,
        n_pairs_void=n_void,
        n_pairs_pending=n_pending,
        archetype_distribution=archetype_distribution,
    )
    return pairs, report


def _select_matching_thesis(
    derived: list[Thesis], archetype_str: str | None
) -> Thesis | None:
    """Pick the thesis whose archetype matches the production archetype.

    Returns None if no match — the rule layer changed since the pick
    was logged, OR the archetype string is unrecognized.
    """
    if not derived or not archetype_str:
        return None
    try:
        target = ThesisArchetype(archetype_str)
    except ValueError:
        return None
    for t in derived:
        if t.archetype == target:
            return t
    return None


def _load_picks(root: Path, date_range):
    import polars as pl

    if not root.exists():
        return pl.DataFrame()
    frames = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("dt="):
            continue
        try:
            part_date = datetime.strptime(part_dir.name[3:], "%Y-%m-%d")
        except ValueError:
            continue
        if date_range is not None:
            if part_date.date() < date_range[0].date():
                continue
            if part_date.date() > date_range[1].date():
                continue
        path = part_dir / "picks.parquet"
        if not path.exists():
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _load_outcomes(root: Path, date_range):
    import polars as pl

    if not root.exists():
        return pl.DataFrame()
    frames = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("dt="):
            continue
        try:
            part_date = datetime.strptime(part_dir.name[3:], "%Y-%m-%d")
        except ValueError:
            continue
        if date_range is not None:
            if part_date.date() < date_range[0].date():
                continue
            if part_date.date() > date_range[1].date():
                continue
        path = part_dir / "picks_outcomes.parquet"
        if not path.exists():
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def pairs_to_fit_input(
    pairs: list[RealPatternPair],
) -> list[tuple[GameStateVector, Thesis]]:
    """Convert ``RealPatternPair`` records to the tuple format that
    ``PatternLayer.fit`` expects.
    """
    return [(p.gsv, p.thesis) for p in pairs]


__all__ = [
    "JoinReport",
    "RealPatternPair",
    "build_real_pattern_pairs",
    "pairs_to_fit_input",
]
