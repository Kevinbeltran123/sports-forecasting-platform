"""Apply league-strength multipliers to player recent-form rates.

The club_form_loader produces RAW per-90 rates from a player's club season.
This module scales those rates into an EPL-equivalent baseline so they can
be summed across teammates from heterogeneous leagues without distortion.

Players from leagues missing in the multiplier table fall back to
FALLBACK_MULTIPLIER (see league_strength.py) and a warning is logged.
Unreliable players (minutes_total < 600) keep their raw rates but the
`reliable=False` flag propagates downstream so the blender can shrink
them toward position averages.
"""

from __future__ import annotations

import polars as pl
import structlog

from bip.evaluation.tournaments.data.league_strength import LeagueStrengthTable

logger = structlog.get_logger(__name__)

# Per-90 rate columns that should be scaled by the league multiplier.
# (Cards / fouls are NOT scaled — referee strictness is league-specific
# and dominates league-quality variance for those metrics.)
_SCALABLE_COLS = (
    "shots_per90",
    "sot_per90",
    "goals_per90",
    "assists_per90",
    "key_passes_per90",
)


def normalize_player_rates(
    raw_form: pl.DataFrame,
    strength_table: LeagueStrengthTable,
) -> pl.DataFrame:
    """Add `normalized_<metric>_per90` columns scaled by league strength.

    Args:
        raw_form: Output of club_form_loader.load_player_form (one row per
            player) with `club_league_id` populated.
        strength_table: Loaded league-strength multipliers.

    Returns:
        New DataFrame with normalized columns. Raw columns preserved for
        inspection.
    """
    if raw_form.is_empty():
        return raw_form

    # Build a lookup column: multiplier per row.
    if "club_league_id" not in raw_form.columns:
        logger.warning("normalize_player_rates_missing_league_id_column")
        return raw_form

    multipliers = [
        strength_table.get(lid)
        for lid in raw_form["club_league_id"].to_list()
    ]

    out = raw_form.with_columns(pl.Series(name="league_multiplier", values=multipliers))

    for col in _SCALABLE_COLS:
        if col in raw_form.columns:
            out = out.with_columns(
                (pl.col(col) * pl.col("league_multiplier")).alias(f"normalized_{col}")
            )

    return out
