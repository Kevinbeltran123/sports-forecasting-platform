"""Per-player match projections derived from blended rates.

For each starter in a predicted lineup, compute:
- λ shots, λ SoT (per-90 = per-match)
- P(anytime goalscorer) = 1 - exp(-λ_player_goals)
- λ fouls committed
- a reliability tag from minutes_total threshold (R-03 in SPIKE.md)

Reliability handling: players with minutes_total < 600 are flagged. Their
raw rates are kept but the consumer (output emitter, downstream picks) is
expected to honor the flag — typically by displaying with a (low minutes)
caveat or by not surfacing them in player-prop markets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

# Match length used to convert per-90 to per-match. National-team matches
# are exactly 90 + stoppage; we ignore stoppage for projection purposes.
MATCH_MINUTES = 90.0


@dataclass(frozen=True)
class PlayerProjection:
    """Per-player projection for a single match."""

    canonical_id: str
    name: str
    position: str

    lambda_shots: float
    lambda_shots_on_target: float
    lambda_goals: float
    p_anytime_scorer: float
    lambda_fouls_committed: float

    minutes_total: int
    reliable: bool


def project_starters(
    lineup_form: pl.DataFrame,
    *,
    name_overrides: dict[str, str] | None = None,
) -> list[PlayerProjection]:
    """Build a list of PlayerProjection entries for a starting XI.

    Args:
        lineup_form: 11-row frame with columns from normalize_player_rates:
            canonical_id, position, minutes_total, reliable, normalized_*_per90,
            fouls_committed_per90 (NOT normalized — referee-bound).
        name_overrides: optional mapping canonical_id -> display name. If a
            row's canonical_id is missing from this dict, the projection
            uses canonical_id as the name (caller can supply names from
            squad metadata).

    Returns:
        list of PlayerProjection (one per row, preserving input order).
    """
    if lineup_form.is_empty():
        return []

    overrides = name_overrides or {}
    out: list[PlayerProjection] = []
    for row in lineup_form.iter_rows(named=True):
        canonical = row["canonical_id"]
        lam_shots = _coalesce(row, "normalized_shots_per90", "shots_per90")
        lam_sot = _coalesce(row, "normalized_sot_per90", "sot_per90")
        lam_goals = _coalesce(row, "normalized_goals_per90", "goals_per90")
        lam_fouls = _coalesce(row, "fouls_committed_per90")

        # P(anytime scorer) = 1 - P(0 goals) where goals ~ Poisson(λ).
        p_scorer = 1.0 - math.exp(-lam_goals) if lam_goals > 0 else 0.0

        out.append(
            PlayerProjection(
                canonical_id=canonical,
                name=overrides.get(canonical, canonical),
                position=row.get("position") or "M",
                lambda_shots=lam_shots,
                lambda_shots_on_target=lam_sot,
                lambda_goals=lam_goals,
                p_anytime_scorer=p_scorer,
                lambda_fouls_committed=lam_fouls,
                minutes_total=int(row.get("minutes_total") or 0),
                reliable=bool(row.get("reliable") or False),
            )
        )
    return out


def _coalesce(row: dict, *cols: str) -> float:
    """Return the first non-null float value across `cols`."""
    for col in cols:
        v = row.get(col)
        if v is not None:
            return float(v)
    return 0.0
