"""Tests for live.match_state — building MatchState from API responses."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.team_style_profiler.live.match_state import (
    MatchState,
    build_match_state,
    next_poll_seconds,
)


def _fixture_payload(
    fid: int = 1, status: str = "1H", elapsed: int = 30,
    hg: int = 1, ag: int = 0,
) -> dict:
    return {
        "response": [
            {
                "fixture": {
                    "id": fid,
                    "date": "2026-06-12T18:00:00+00:00",
                    "status": {"long": "test", "short": status, "elapsed": elapsed},
                },
                "league": {"id": 1, "name": "WC", "season": 2026},
                "teams": {
                    "home": {"id": 26, "name": "Argentina"},
                    "away": {"id": 13, "name": "Japan"},
                },
                "goals": {"home": hg, "away": ag},
                "score": {
                    "halftime": {"home": 0, "away": 0},
                    "fulltime": {"home": None, "away": None},
                },
            }
        ]
    }


def _events_payload(events: list[dict]) -> dict:
    return {"response": events}


def _yc_event(team_id: int, minute: int) -> dict:
    return {
        "time": {"elapsed": minute, "extra": None},
        "team": {"id": team_id, "name": f"T{team_id}"},
        "type": "Card",
        "detail": "Yellow Card",
    }


class TestBuildMatchState:
    def test_pre_match(self) -> None:
        ms = build_match_state(_fixture_payload(status="NS", elapsed=0, hg=0, ag=0), _events_payload([]))
        assert ms.is_pre_match is True
        assert ms.is_finished is False
        assert ms.minute == 0
        assert ms.minutes_remaining == 90

    def test_in_play_1h(self) -> None:
        ms = build_match_state(_fixture_payload(status="1H", elapsed=30, hg=1, ag=0), _events_payload([]))
        assert ms.is_in_play is True
        assert ms.minute == 30
        assert ms.home_goals == 1
        assert ms.away_goals == 0
        assert ms.total_goals == 1
        assert ms.minutes_remaining == 60

    def test_ht(self) -> None:
        ms = build_match_state(_fixture_payload(status="HT", elapsed=45), _events_payload([]))
        assert ms.status == "HT"
        assert ms.is_in_play is True

    def test_finished_ft(self) -> None:
        ms = build_match_state(_fixture_payload(status="FT", elapsed=90, hg=2, ag=1), _events_payload([]))
        assert ms.is_finished is True
        assert ms.minutes_remaining == 0
        assert ms.fraction_played == 1.0

    def test_cards_counted_per_team(self) -> None:
        events = [
            _yc_event(26, 12), _yc_event(26, 33),  # Argentina 2 yellow
            _yc_event(13, 45), _yc_event(13, 67), _yc_event(13, 88),  # Japan 3 yellow
        ]
        ms = build_match_state(_fixture_payload(elapsed=88), _events_payload(events))
        assert ms.home_yellow_cards == 2
        assert ms.away_yellow_cards == 3
        assert ms.total_yellow_cards == 5

    def test_red_card_counted(self) -> None:
        events = [
            {
                "time": {"elapsed": 60, "extra": None},
                "team": {"id": 26, "name": "Argentina"},
                "type": "Card", "detail": "Red Card",
            }
        ]
        ms = build_match_state(_fixture_payload(elapsed=60), _events_payload(events))
        assert ms.home_red_cards == 1
        assert ms.away_red_cards == 0

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ValueError):
            build_match_state({"response": []}, _events_payload([]))


class TestNextPollSeconds:
    def test_default_60s(self) -> None:
        ms = MatchState(
            fixture_id=1, status="1H", minute=25,
            home_team_id=1, away_team_id=2, home_team="A", away_team="B",
            home_goals=0, away_goals=0,
            home_yellow_cards=0, away_yellow_cards=0,
            home_red_cards=0, away_red_cards=0,
        )
        assert next_poll_seconds(ms) == 60

    def test_approaching_ht_15s(self) -> None:
        ms = MatchState(
            fixture_id=1, status="1H", minute=42,
            home_team_id=1, away_team_id=2, home_team="A", away_team="B",
            home_goals=0, away_goals=0,
            home_yellow_cards=0, away_yellow_cards=0,
            home_red_cards=0, away_red_cards=0,
        )
        assert next_poll_seconds(ms) == 15

    def test_approaching_ft_15s(self) -> None:
        ms = MatchState(
            fixture_id=1, status="2H", minute=85,
            home_team_id=1, away_team_id=2, home_team="A", away_team="B",
            home_goals=0, away_goals=0,
            home_yellow_cards=0, away_yellow_cards=0,
            home_red_cards=0, away_red_cards=0,
        )
        assert next_poll_seconds(ms) == 15

    def test_finished_60s(self) -> None:
        ms = MatchState(
            fixture_id=1, status="FT", minute=92,
            home_team_id=1, away_team_id=2, home_team="A", away_team="B",
            home_goals=1, away_goals=0,
            home_yellow_cards=0, away_yellow_cards=0,
            home_red_cards=0, away_red_cards=0,
        )
        assert next_poll_seconds(ms) == 60
