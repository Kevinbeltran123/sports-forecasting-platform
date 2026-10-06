"""Tests for altitude verdict (FIFA 2010 + Apunts 2022)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    ACCLIMATIZED_TEAMS,
    ALTITUDE_HOME_EDGE_GOALS_PER_1000M,
    ALTITUDE_THRESHOLD_M,
    altitude_verdict,
)


class TestConstants:
    def test_threshold_is_one_thousand_meters(self) -> None:
        assert ALTITUDE_THRESHOLD_M == 1000

    def test_home_edge_is_half_goal_per_1000m(self) -> None:
        # FIFA 2010 prospective study. Lock in.
        assert ALTITUDE_HOME_EDGE_GOALS_PER_1000M == 0.5

    def test_acclimatized_teams_include_canonical_set(self) -> None:
        assert "Mexico" in ACCLIMATIZED_TEAMS
        assert "Bolivia" in ACCLIMATIZED_TEAMS
        assert "Ecuador" in ACCLIMATIZED_TEAMS


class TestHighAltitudeAgainstSeaLevel:
    def test_mexico_city_vs_germany_triggers(self) -> None:
        v = altitude_verdict("mexico_city", "Mexico", "Germany")
        assert v.is_high_altitude is True
        assert v.away_team_acclimatized is False
        # 2240m * 0.5 / 1000 = 1.12 goals
        assert v.expected_home_edge_goals == pytest.approx(1.12, abs=0.01)
        assert v.tilt_direction == "home_edge + late_game_goals_75_90"
        assert "FIFA 2010" in v.rationale

    def test_guadalajara_vs_england_triggers(self) -> None:
        v = altitude_verdict("guadalajara", "Mexico", "England")
        assert v.is_high_altitude is True
        # 1566m * 0.5 / 1000 = 0.783 goals
        assert v.expected_home_edge_goals == pytest.approx(0.783, abs=0.01)

    def test_mexico_city_vs_brazil_triggers(self) -> None:
        v = altitude_verdict("mexico_city", "Mexico", "Brazil")
        assert v.is_high_altitude is True
        assert v.away_team_acclimatized is False


class TestAcclimatizedAwayTeam:
    @pytest.mark.parametrize(
        "away", ["Bolivia", "Ecuador", "Colombia", "Peru", "Mexico"]
    )
    def test_acclimatized_away_zeroes_edge(self, away: str) -> None:
        v = altitude_verdict("mexico_city", "Argentina", away)
        assert v.is_high_altitude is True
        assert v.away_team_acclimatized is True
        assert v.expected_home_edge_goals == 0.0
        assert v.tilt_direction == "no altitude tilt"

    def test_bolivia_in_guadalajara_no_tilt(self) -> None:
        v = altitude_verdict("guadalajara", "Mexico", "Bolivia")
        assert v.away_team_acclimatized is True
        assert v.expected_home_edge_goals == 0.0


class TestLowAltitudeVenuesNeverTrigger:
    @pytest.mark.parametrize(
        "slug",
        [
            "dallas",
            "houston",
            "monterrey",
            "atlanta",
            "los_angeles",
            "miami",
            "vancouver",
            "toronto",
            "new_york",
            "boston",
        ],
    )
    def test_below_threshold_no_tilt(self, slug: str) -> None:
        v = altitude_verdict(slug, "USA", "Germany")
        assert v.is_high_altitude is False
        assert v.expected_home_edge_goals == 0.0
        assert v.tilt_direction == "no altitude tilt"


class TestEdgeCases:
    def test_unknown_venue_returns_no_tilt(self) -> None:
        v = altitude_verdict("nonexistent", "Home", "Away")
        assert v.is_high_altitude is False
        assert v.tilt_direction == "no altitude tilt"
        assert "unknown" in v.rationale.lower()
