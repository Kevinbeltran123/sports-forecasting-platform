"""Lightweight parquet cache for evaluation artifacts.

This module is intentionally simpler than `bip.core.storage.parquet_store`
(which uses Hive partitioning for the picks pipeline). Evaluation outputs
are research-scale: one parquet per (category, tournament, team-or-player).

Cache layout::

    data/cache/evaluation/
        team_qualifier_baselines/
            <tournament_slug>/
                <team_id>.parquet
        player_recent_form/
            <tournament_slug>/
                <player_canonical_id>.parquet
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_CACHE_ROOT = Path("data/cache/evaluation")


def cache_path(
    category: str,
    tournament_slug: str,
    entity_id: str,
    *,
    root: Path | None = None,
) -> Path:
    """Build the parquet path for a (category, tournament, entity) triple.

    `root` defaults to the module-level DEFAULT_CACHE_ROOT, resolved at
    call time so monkeypatching the module attribute works in tests.
    """
    base = root if root is not None else DEFAULT_CACHE_ROOT
    return base / category / tournament_slug / f"{entity_id}.parquet"


def write(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("evaluation_cache_write", path=str(path), rows=df.height)


def read(path: Path) -> pl.DataFrame | None:
    """Read a parquet from path, or return None if missing."""
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    logger.debug("evaluation_cache_hit", path=str(path), rows=df.height)
    return df
