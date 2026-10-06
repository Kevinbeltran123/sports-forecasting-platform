"""Test fixtures for engine_v3.

We synthesize ``LiveMatchState`` instances directly rather than going
through the Sportmonks Fixture parser, because the v3 design treats
LiveMatchState as an abstract producer — what matters is that the
fields end up correctly populated, not the upstream wire format.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from bip.evaluation.live.engine_v3 import (
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.evaluation.live.match_state import LiveMatchState
from bip.sports.football.sportmonks.types import StatType


HOME_ID = 100
AWAY_ID = 200


def make_state(
    *,
    home_goals: int = 0,
    away_goals: int = 0,
    minute: int = 30,
    home_stats: dict[int, float] | None = None,
    away_stats: dict[int, float] | None = None,
    red_card_events: list[tuple[int, int]] | None = None,
    yellow_card_events: list[tuple[int, int, int]] | None = None,
    substitution_events: list[tuple[int, int, int]] | None = None,
    goal_events: list[tuple[int, int]] | None = None,
) -> LiveMatchState:
    """Build a minimal LiveMatchState for tests.

    Defaults: 30' into a 0-0 match with no events, plausible aggregate
    stats, no trends. Override individual fields as needed.
    """
    home_stats = home_stats or {
        StatType.SHOTS_TOTAL: 5,
        StatType.SHOTS_ON_TARGET: 2,
        StatType.SHOTS_INSIDEBOX: 3,
        StatType.BIG_CHANCES_CREATED: 1,
        StatType.CORNERS: 2,
        StatType.BALL_POSSESSION: 55.0,
        StatType.DANGEROUS_ATTACKS: 20,
        StatType.KEY_PASSES: 4,
    }
    away_stats = away_stats or {
        StatType.SHOTS_TOTAL: 3,
        StatType.SHOTS_ON_TARGET: 1,
        StatType.SHOTS_INSIDEBOX: 2,
        StatType.BIG_CHANCES_CREATED: 0,
        StatType.CORNERS: 1,
        StatType.BALL_POSSESSION: 45.0,
        StatType.DANGEROUS_ATTACKS: 12,
        StatType.KEY_PASSES: 2,
    }
    return LiveMatchState(
        fixture_id=1234,
        home_team_id=HOME_ID,
        away_team_id=AWAY_ID,
        home_team_name="Home FC",
        away_team_name="Away FC",
        home_goals=home_goals,
        away_goals=away_goals,
        minute=minute,
        period_id=2 if minute > 45 else 1,
        is_live=True,
        is_half_time=False,
        is_finished=False,
        home_stats=home_stats,
        away_stats=away_stats,
        red_card_events=red_card_events or [],
        yellow_card_events=yellow_card_events or [],
        substitution_events=substitution_events or [],
        goal_events=goal_events or [],
    )


@pytest.fixture
def priors() -> PreMatchPriors:
    """Standard priors: home favourite (Napoli scenario)."""
    return PreMatchPriors(
        lambda_home_prematch=2.10,  # home is dominant
        lambda_away_prematch=0.95,
        expected_corners_total=10.4,
        expected_cards_total=3.95,
        elo_diff=120.0,  # home favoured
    )


@pytest.fixture
def market_snapshot() -> MarketSnapshot:
    """A snapshot with representative lines: goals, corners, BTTS, next_goal."""
    now = datetime.now(timezone.utc)
    return MarketSnapshot(
        lines={
            "match_goals_over_2.5": MarketLine(
                market_id="match_goals_over_2.5",
                side_a_decimal=1.95, side_b_decimal=1.95,
                line_value=2.5, max_stake_cap=500.0,
                last_update_utc=now - timedelta(seconds=20),
            ),
            "match_goals_under_2.5": MarketLine(
                market_id="match_goals_under_2.5",
                side_a_decimal=1.95, side_b_decimal=1.95,
                line_value=2.5, max_stake_cap=500.0,
                last_update_utc=now - timedelta(seconds=20),
            ),
            "btts_yes": MarketLine(
                market_id="btts_yes",
                side_a_decimal=1.90, side_b_decimal=1.90,
                max_stake_cap=400.0,
                last_update_utc=now - timedelta(seconds=15),
            ),
            "btts_no": MarketLine(
                market_id="btts_no",
                side_a_decimal=1.85, side_b_decimal=1.85,
                max_stake_cap=400.0,
                last_update_utc=now - timedelta(seconds=15),
            ),
            "match_corners_over_10.5": MarketLine(
                market_id="match_corners_over_10.5",
                side_a_decimal=1.95, side_b_decimal=1.85,
                line_value=10.5, max_stake_cap=300.0,
                last_update_utc=now - timedelta(seconds=30),
            ),
            "second_half_corners_over_5.5": MarketLine(
                market_id="second_half_corners_over_5.5",
                side_a_decimal=2.10, side_b_decimal=1.72,
                line_value=5.5, max_stake_cap=200.0,
                last_update_utc=now - timedelta(seconds=25),
            ),
            "next_goal_home": MarketLine(
                market_id="next_goal_home",
                side_a_decimal=2.40, side_b_decimal=None,
                max_stake_cap=250.0,
                last_update_utc=now - timedelta(seconds=10),
            ),
            "next_goal_away": MarketLine(
                market_id="next_goal_away",
                side_a_decimal=3.10, side_b_decimal=None,
                max_stake_cap=250.0,
                last_update_utc=now - timedelta(seconds=10),
            ),
        }
    )


@pytest.fixture
def state_napoli_scenario() -> LiveMatchState:
    """The canonical anti-Napoli case.

    Home is the dominant team (priors set λ_home > λ_away). They are
    trailing 0-1 at minute 35. xG_diff strongly favours home (1.4 xG
    vs 0.4 xG → +1.0 divergence vs expected -0.8 for a -1 score state,
    so xg_vs_score_divergence ≈ +1.8).
    """
    stats_home = {
        StatType.SHOTS_TOTAL: 11,
        StatType.SHOTS_ON_TARGET: 4,
        StatType.SHOTS_INSIDEBOX: 6,
        StatType.BIG_CHANCES_CREATED: 3,
        StatType.CORNERS: 5,
        StatType.BALL_POSSESSION: 65.0,
        StatType.DANGEROUS_ATTACKS: 60,
        StatType.KEY_PASSES: 9,
    }
    stats_away = {
        StatType.SHOTS_TOTAL: 3,
        StatType.SHOTS_ON_TARGET: 2,
        StatType.SHOTS_INSIDEBOX: 1,
        StatType.BIG_CHANCES_CREATED: 0,
        StatType.CORNERS: 1,
        StatType.BALL_POSSESSION: 35.0,
        StatType.DANGEROUS_ATTACKS: 18,
        StatType.KEY_PASSES: 2,
    }
    return make_state(
        home_goals=0,
        away_goals=1,
        minute=35,
        home_stats=stats_home,
        away_stats=stats_away,
        goal_events=[(20, AWAY_ID)],
    )
