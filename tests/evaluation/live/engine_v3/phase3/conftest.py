"""Shared fixtures for Phase 3 tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bip.evaluation.live.engine_v3 import (
    GameStateVector,
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.evaluation.live.engine_v3.gsv import (
    CardsState,
    CornerState,
    CriticalEvent,
    FlowState,
    NumericalState,
    PlayerOnYellow,
    RosterState,
    ScoreState,
    TacticalState,
    TimeState,
    XGState,
)


@pytest.fixture
def base_gsv() -> GameStateVector:
    """Standard mid-game GSV — minute 60, 1-1, neutral phase."""
    now = datetime.now(timezone.utc)
    return GameStateVector(
        fixture_id=42,
        state_version=1,
        timestamp_utc=now,
        home_team_id=100,
        away_team_id=200,
        score=ScoreState(home_goals=1, away_goals=1, goal_diff=0,
                         dominant_team_id=100, dominant_losing=False),
        time=TimeState(minute=60, period="2H",
                       time_remaining_match=30.0),
        numerical=NumericalState(),
        xg=XGState(home_xg_total=1.3, away_xg_total=0.9, xg_diff=0.4,
                   xg_vs_score_divergence=0.4,
                   xg_per_min_home_last_15=0.018,
                   xg_per_min_away_last_15=0.012),
        flow=FlowState(possession_home_5min=55.0),
        corners=CornerState(corners_home=4, corners_away=3),
        cards=CardsState(yellows=(2, 3),
                         ref_card_rate_prior=4.5,
                         card_rate_last_15min=0.20),
        roster=RosterState(),
        tactical=TacticalState(),
        priors=PreMatchPriors(
            lambda_home_prematch=1.6, lambda_away_prematch=1.0,
            expected_cards_total=4.2,
        ),
        markets=MarketSnapshot(),
    )


@pytest.fixture
def napoli_gsv() -> GameStateVector:
    """Dominant-losing scenario (Napoli archetype)."""
    now = datetime.now(timezone.utc)
    return GameStateVector(
        fixture_id=43,
        state_version=1,
        timestamp_utc=now,
        home_team_id=100,
        away_team_id=200,
        score=ScoreState(home_goals=0, away_goals=1, goal_diff=-1,
                         dominant_team_id=100, dominant_losing=True,
                         last_goal_team_id=200, last_goal_minute=20),
        time=TimeState(minute=35, period="1H", time_remaining_match=55.0),
        numerical=NumericalState(),
        xg=XGState(home_xg_total=1.40, away_xg_total=0.35, xg_diff=1.05,
                   xg_vs_score_divergence=1.85,
                   xg_per_min_home_last_15=0.040,  # hot — well above prior
                   xg_per_min_away_last_15=0.005),
        flow=FlowState(possession_home_5min=65.0),
        corners=CornerState(corners_home=5, corners_away=1),
        cards=CardsState(yellows=(1, 2), ref_card_rate_prior=4.0),
        roster=RosterState(),
        tactical=TacticalState(home_phase="chasing", away_phase="parking_bus",
                                game_phase="desperate"),
        priors=PreMatchPriors(
            lambda_home_prematch=2.1, lambda_away_prematch=0.95,
        ),
        markets=MarketSnapshot(),
    )


@pytest.fixture
def cruise_gsv() -> GameStateVector:
    """Late lead 1-0 at minute 82, cruise mode."""
    now = datetime.now(timezone.utc)
    return GameStateVector(
        fixture_id=44,
        state_version=1,
        timestamp_utc=now,
        home_team_id=100,
        away_team_id=200,
        score=ScoreState(home_goals=1, away_goals=0, goal_diff=1,
                         dominant_team_id=100, dominant_losing=False),
        time=TimeState(minute=82, period="2H", time_remaining_match=8.0),
        numerical=NumericalState(),
        xg=XGState(home_xg_total=1.5, away_xg_total=0.9, xg_diff=0.6,
                   xg_vs_score_divergence=-0.2,
                   xg_per_min_home_last_15=0.005,
                   xg_per_min_away_last_15=0.005),
        flow=FlowState(possession_home_5min=68.0),
        corners=CornerState(corners_home=6, corners_away=2),
        cards=CardsState(yellows=(2, 4), ref_card_rate_prior=4.0),
        roster=RosterState(),
        tactical=TacticalState(home_phase="controlling", away_phase="chasing",
                                game_phase="cruise"),
        priors=PreMatchPriors(
            lambda_home_prematch=1.5, lambda_away_prematch=1.0,
        ),
        markets=MarketSnapshot(),
    )
