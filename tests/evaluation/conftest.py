"""Shared test factories for evaluation module tests.

Plain helper functions (make_blended_rates) are importable directly.
Pytest fixtures (make_match_prediction) are auto-discovered by pytest.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from bip.evaluation.tournaments.players.lineup_predictor import (
    LineupConfidence,
    LineupPrediction,
)
from bip.evaluation.tournaments.models import Player, Position
from bip.evaluation.tournaments.predict.blender import BlendedRates
from bip.evaluation.tournaments.predict.output import MatchPrediction, TeamSidePrediction
from bip.evaluation.tournaments.predict.player_projections import PlayerProjection
from bip.evaluation.tournaments.predictors.base import GoalsDistribution
from bip.evaluation.tournaments.predictors.independent_poisson import IndependentPoissonModel


# ── Plain helpers (importable) ────────────────────────────────────────────────


def make_blended_rates(
    team_id: int = 1,
    goals: float = 1.5,
    corners: float = 5.0,
    shots: float = 12.0,
    sot: float = 4.0,
    fouls: float = 10.0,
    n_starters: int = 11,
) -> BlendedRates:
    return BlendedRates(
        team_id=team_id,
        expected_goals_per90=goals,
        expected_shots_per90=shots,
        expected_sot_per90=sot,
        expected_fouls_committed_per90=fouls,
        expected_corners_for_per90=corners,
        alpha=0.35,
        team_component_goals=goals,
        lineup_component_goals=goals,
        n_starters_with_form=n_starters,
    )


def make_player_projection(
    pid: int = 1,
    name: str | None = None,
    goals_per90: float = 0.18,
) -> PlayerProjection:
    lam_goals = goals_per90
    return PlayerProjection(
        canonical_id=f"af-{pid}",
        name=name or f"Player {pid}",
        position="F",
        lambda_shots=1.5,
        lambda_shots_on_target=0.6,
        lambda_goals=lam_goals,
        p_anytime_scorer=1.0 - math.exp(-lam_goals),
        lambda_fouls_committed=1.0,
        minutes_total=1500,
        reliable=True,
    )


def make_goals_distribution(
    lambda_home: float = 1.5,
    lambda_away: float = 1.2,
) -> GoalsDistribution:
    model = IndependentPoissonModel()
    return model.predict(
        make_blended_rates(team_id=1, goals=lambda_home),
        make_blended_rates(team_id=2, goals=lambda_away),
    )


def make_team_side(
    team_id: int = 1,
    team_name: str = "Home FC",
    n_starters: int = 4,
    lineup_confidence: str = "heuristic_qualifier",
) -> TeamSidePrediction:
    starters = tuple(
        make_player_projection(pid=team_id * 100 + i, name=f"Player {team_id}-{i}")
        for i in range(n_starters)
    )
    return TeamSidePrediction(
        team_id=team_id,
        team_name=team_name,
        fifa_code=team_name[:3].upper(),
        lineup_confidence=lineup_confidence,
        starters=starters,
        blended=make_blended_rates(team_id=team_id),
    )


# ── Pytest fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def make_match_prediction():
    """Factory fixture: call it to get a MatchPrediction with optional overrides."""

    def _factory(
        match_id: str = "test_match_001",
        home_name: str = "Home FC",
        away_name: str = "Away FC",
        lambda_home: float = 1.5,
        lambda_away: float = 1.2,
        warnings: tuple[str, ...] = (),
    ) -> MatchPrediction:
        goals = make_goals_distribution(lambda_home, lambda_away)
        return MatchPrediction(
            match_id=match_id,
            tournament_slug="world_cup_2026",
            kickoff=datetime(2026, 6, 14, 15, 0),
            predicted_at=datetime(2026, 6, 12, 10, 0),
            home=make_team_side(team_id=1, team_name=home_name),
            away=make_team_side(team_id=2, team_name=away_name),
            goals=goals,
            alpha=0.35,
            rho=0.04,
            warnings=warnings,
        )

    return _factory
