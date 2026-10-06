"""Tests for API-Football response parsers."""
from __future__ import annotations

from bip.evaluation.tournaments.team_style_profiler.api_football_parsers import (
    Event,
    Fixture,
    FixturesResponse,
    StatItem,
    StatsResponse,
    EventsResponse,
)


class TestFixtureParsing:
    def test_minimal_fixture(self) -> None:
        payload = {
            "response": [
                {
                    "fixture": {
                        "id": 12345,
                        "date": "2024-09-10T18:00:00+00:00",
                        "status": {"long": "Match Finished", "short": "FT", "elapsed": 90},
                    },
                    "league": {"id": 10, "name": "Friendlies", "country": "World", "season": 2024},
                    "teams": {
                        "home": {"id": 26, "name": "Argentina", "winner": True},
                        "away": {"id": 13, "name": "Japan", "winner": False},
                    },
                    "goals": {"home": 2, "away": 1},
                    "score": {
                        "halftime": {"home": 1, "away": 0},
                        "fulltime": {"home": 2, "away": 1},
                    },
                }
            ]
        }
        resp = FixturesResponse.model_validate(payload)
        assert len(resp.response) == 1
        f = resp.response[0]
        assert f.fixture_id == 12345
        assert f.is_finished is True
        assert f.ht_home_goals == 1
        assert f.ht_away_goals == 0
        assert f.teams.home.name == "Argentina"

    def test_unfinished_fixture(self) -> None:
        payload = {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "date": "2026-06-12T18:00:00+00:00",
                        "status": {"long": "Not Started", "short": "NS"},
                    },
                    "league": {"id": 1, "name": "WC", "season": 2026},
                    "teams": {
                        "home": {"id": 1, "name": "A"},
                        "away": {"id": 2, "name": "B"},
                    },
                    "goals": {"home": None, "away": None},
                    "score": {
                        "halftime": {"home": None, "away": None},
                        "fulltime": {"home": None, "away": None},
                    },
                }
            ]
        }
        f = FixturesResponse.model_validate(payload).response[0]
        assert f.is_finished is False
        assert f.ht_home_goals == 0  # None coerced to 0 via property


class TestStatsParsing:
    def test_stat_value_coercion(self) -> None:
        # Mixed types as API-Football returns them.
        payload = {
            "response": [
                {
                    "team": {"id": 26, "name": "Argentina"},
                    "statistics": [
                        {"type": "Total Shots", "value": 14},
                        {"type": "Shots on Goal", "value": 5},
                        {"type": "Ball Possession", "value": "62%"},
                        {"type": "Pass Accuracy", "value": "85.3%"},
                        {"type": "Corner Kicks", "value": 6},
                        {"type": "Offsides", "value": None},
                    ],
                }
            ]
        }
        resp = StatsResponse.model_validate(payload)
        block = resp.for_team(26)
        assert block is not None
        assert block.get("Total Shots") == 14
        assert block.get("Shots on Goal") == 5
        assert block.get("Ball Possession") == 62
        assert block.get("Pass Accuracy") == 85.3
        assert block.get("Corner Kicks") == 6
        assert block.get("Offsides") is None
        assert block.get("Nonexistent") is None

    def test_for_team_returns_none_when_missing(self) -> None:
        resp = StatsResponse.model_validate({"response": []})
        assert resp.for_team(99) is None


class TestEventsParsing:
    def test_goal_minute_extraction(self) -> None:
        payload = {
            "response": [
                {
                    "time": {"elapsed": 23, "extra": None},
                    "team": {"id": 26, "name": "Argentina"},
                    "type": "Goal",
                    "detail": "Normal Goal",
                },
                {
                    "time": {"elapsed": 45, "extra": 3},
                    "team": {"id": 13, "name": "Japan"},
                    "type": "Goal",
                    "detail": "Penalty",
                },
                {
                    "time": {"elapsed": 67, "extra": None},
                    "team": {"id": 26, "name": "Argentina"},
                    "type": "Card",
                    "detail": "Yellow Card",
                },
                {
                    "time": {"elapsed": 90, "extra": 5},
                    "team": {"id": 26, "name": "Argentina"},
                    "type": "Card",
                    "detail": "Red Card",
                },
                {
                    "time": {"elapsed": 88, "extra": None},
                    "team": {"id": 13, "name": "Japan"},
                    "type": "Goal",
                    "detail": "Missed Penalty",  # not counted as a goal
                },
            ]
        }
        resp = EventsResponse.model_validate(payload)
        evs = resp.response
        # Two goals scored (first 2), 1 missed penalty (last one)
        goals = [e for e in evs if e.is_goal]
        assert len(goals) == 2
        assert goals[0].time.total_minute == 23
        assert goals[1].time.total_minute == 48  # 45 + 3

        yellows = [e for e in evs if e.is_yellow_card]
        assert len(yellows) == 1
        assert yellows[0].team.id == 26

        reds = [e for e in evs if e.is_red_card]
        assert len(reds) == 1
        assert reds[0].time.total_minute == 95  # 90 + 5
