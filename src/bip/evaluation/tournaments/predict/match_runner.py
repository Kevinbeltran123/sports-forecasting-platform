"""Orchestrator: combine team baselines + lineups + form into MatchPrediction.

The runner is intentionally thin — it does not load data from disk or hit
APIs. The caller (CLI script in scripts/, or future scheduler) is
responsible for fetching/caching and passing pre-loaded artifacts.

This separation keeps the orchestrator trivially testable (only Polars
DataFrames and dataclasses go in) and lets us swap data sources without
touching the prediction logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from bip.evaluation.tournaments.models import Player
from bip.evaluation.tournaments.players.lineup_predictor import (
    LineupConfidence,
    LineupPrediction,
)
from bip.evaluation.tournaments.predict.blender import (
    DEFAULT_ALPHA,
    blend_team_rates,
)
from bip.evaluation.tournaments.predict.output import (
    MatchPrediction,
    TeamSidePrediction,
)
from bip.evaluation.tournaments.predict.player_projections import (
    PlayerProjection,
    project_starters,
)
from bip.evaluation.tournaments.predictors.base import GoalsModel
from bip.evaluation.tournaments.predictors.corners_poisson import CornersPoissonModel


@dataclass(frozen=True)
class TeamMatchInputs:
    """Pre-loaded artifacts for one side of a fixture."""

    team_id: int
    team_name: str
    fifa_code: str
    baseline_row: pl.DataFrame  # 1 row, output of adjust_team_baselines
    lineup: LineupPrediction
    lineup_form: pl.DataFrame  # 11 rows, output of normalize_player_rates
    name_overrides: dict[str, str] | None = None  # canonical_id -> display name


class MatchPredictionRunner:
    """Build MatchPrediction from pre-loaded inputs and a chosen GoalsModel."""

    def __init__(
        self,
        goals_model: GoalsModel,
        *,
        alpha: float = DEFAULT_ALPHA,
        corners_model: CornersPoissonModel | None = None,
    ) -> None:
        self._goals_model = goals_model
        self._alpha = alpha
        self._corners_model = corners_model

    def predict(
        self,
        match_id: str,
        tournament_slug: str,
        kickoff: datetime,
        home: TeamMatchInputs,
        away: TeamMatchInputs,
        *,
        predicted_at: datetime | None = None,
    ) -> MatchPrediction:
        warnings: list[str] = []

        home_blend = blend_team_rates(
            home.baseline_row, home.lineup_form, alpha=self._alpha
        )
        away_blend = blend_team_rates(
            away.baseline_row, away.lineup_form, alpha=self._alpha
        )

        goals = self._goals_model.predict(home_blend, away_blend)
        corners = (
            self._corners_model.predict(home_blend, away_blend)
            if self._corners_model is not None
            else None
        )

        home_starters = _project_with_lineup_order(
            home.lineup_form, home.lineup, name_overrides=home.name_overrides
        )
        away_starters = _project_with_lineup_order(
            away.lineup_form, away.lineup, name_overrides=away.name_overrides
        )

        # Surface low-confidence lineups as explicit warnings.
        for label, lp in (("home", home.lineup), ("away", away.lineup)):
            if lp.confidence != LineupConfidence.CONFIRMED:
                warnings.append(
                    f"{label} lineup confidence is {lp.confidence.value}; "
                    "expect prediction shifts when XI is confirmed pre-kickoff"
                )

        # Surface player-form gaps (>1 unreliable starters per side).
        for label, blend in (("home", home_blend), ("away", away_blend)):
            unreliable = 11 - blend.n_starters_with_form
            if unreliable >= 2:
                warnings.append(
                    f"{label}: {unreliable} starters lack reliable club-form data "
                    "(<600 minutes); player-prop projections degraded"
                )

        rho = getattr(self._goals_model, "rho", None)

        return MatchPrediction(
            match_id=match_id,
            tournament_slug=tournament_slug,
            kickoff=kickoff,
            predicted_at=predicted_at or datetime.now(),
            home=TeamSidePrediction(
                team_id=home.team_id,
                team_name=home.team_name,
                fifa_code=home.fifa_code,
                lineup_confidence=home.lineup.confidence.value,
                starters=tuple(home_starters),
                blended=home_blend,
            ),
            away=TeamSidePrediction(
                team_id=away.team_id,
                team_name=away.team_name,
                fifa_code=away.fifa_code,
                lineup_confidence=away.lineup.confidence.value,
                starters=tuple(away_starters),
                blended=away_blend,
            ),
            goals=goals,
            corners=corners,
            alpha=self._alpha,
            rho=rho,
            warnings=tuple(warnings),
        )


def _project_with_lineup_order(
    lineup_form: pl.DataFrame,
    lineup: LineupPrediction,
    *,
    name_overrides: dict[str, str] | None,
) -> list[PlayerProjection]:
    """Project starters preserving the lineup order (for deterministic output).

    Re-orders the lineup_form rows so the resulting projection list matches
    `lineup.players` order. Falls back to whatever order the form is in
    if a player is missing from the form (and skips them — caller should
    have ensured 11-row form).
    """
    if lineup_form.is_empty():
        return []

    # Build a name override map from the lineup itself (canonical_id -> name).
    derived_names: dict[str, str] = {
        f"af-{p.api_football_id}": p.name for p in lineup.players
    }
    merged = (name_overrides or {}) | derived_names

    # Re-order the form by lineup order via a positional join.
    canonical_to_idx = {
        canonical_id: i
        for i, canonical_id in enumerate(lineup_form["canonical_id"].to_list())
    }
    ordered_indices = [
        canonical_to_idx[f"af-{p.api_football_id}"]
        for p in lineup.players
        if f"af-{p.api_football_id}" in canonical_to_idx
    ]
    if ordered_indices:
        ordered_form = lineup_form.with_row_index().filter(
            pl.col("index").is_in(ordered_indices)
        ).drop("index")
        # Build a list in the requested order.
        ordered_form = pl.concat(
            [lineup_form.slice(i, 1) for i in ordered_indices]
        )
    else:
        ordered_form = lineup_form

    return project_starters(ordered_form, name_overrides=merged)


def synthesize_lineup_form_for(
    lineup: LineupPrediction, raw_form: pl.DataFrame
) -> pl.DataFrame:
    """Filter raw_form to only the starters in `lineup`, preserving order.

    Raises if any starter is missing from raw_form — caller must ensure
    full coverage (typically by pulling form for the whole squad).
    """
    canonical_ids = [f"af-{p.api_football_id}" for p in lineup.players]
    if "canonical_id" not in raw_form.columns:
        raise ValueError("raw_form must contain a 'canonical_id' column")

    available = set(raw_form["canonical_id"].to_list())
    missing = [c for c in canonical_ids if c not in available]
    if missing:
        raise ValueError(
            f"Lineup references {len(missing)} players missing from raw_form: "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
        )

    # Order-preserving filter via slice-and-concat.
    by_canonical = {row["canonical_id"]: row for row in raw_form.iter_rows(named=True)}
    rows = [by_canonical[cid] for cid in canonical_ids]
    return pl.DataFrame(rows)


# Re-export for convenience: callers often want Player to build squads.
__all__ = [
    "MatchPredictionRunner",
    "TeamMatchInputs",
    "synthesize_lineup_form_for",
    "Player",
]
