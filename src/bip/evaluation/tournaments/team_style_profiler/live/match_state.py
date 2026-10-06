"""Build a MatchState dataclass from API-Football live fixture + events.

MatchState is the input to ``residual_predictor`` — captures everything
the live update needs to recompute the FT probabilities conditional on
the current state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bip.evaluation.tournaments.team_style_profiler.api_football_parsers import (
    EventsResponse,
    Fixture,
    FixturesResponse,
)


LiveStatus = Literal[
    "NS",   # not started
    "1H",   # first half
    "HT",   # half time
    "2H",   # second half
    "ET",   # extra time
    "BT",   # break time
    "P",    # penalty shootout
    "FT",   # full time
    "AET",  # after extra time
    "PEN",  # finished after PEN
    "PST",  # postponed
    "ABD",  # abandoned
]


@dataclass(frozen=True)
class MatchState:
    """Snapshot of match state at one polling moment."""

    fixture_id: int
    status: LiveStatus
    minute: int
    """Elapsed minute. 0 pre-match, 45 at HT, 90 at FT, can go higher in ET."""

    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    # Events-derived counts (at this moment)
    home_yellow_cards: int
    away_yellow_cards: int
    home_red_cards: int
    away_red_cards: int

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def total_yellow_cards(self) -> int:
        return self.home_yellow_cards + self.away_yellow_cards

    @property
    def is_pre_match(self) -> bool:
        return self.status == "NS"

    @property
    def is_finished(self) -> bool:
        return self.status in {"FT", "AET", "PEN", "ABD"}

    @property
    def is_in_play(self) -> bool:
        return self.status in {"1H", "HT", "2H", "ET", "BT", "P"}

    @property
    def minutes_remaining(self) -> int:
        """Estimated regulation minutes remaining (0 once FT/AET).

        Pre-match = 90. During 1H/HT = 90 - minute. Caps at 0.
        """
        if self.is_finished:
            return 0
        if self.is_pre_match:
            return 90
        return max(0, 90 - self.minute)

    @property
    def fraction_played(self) -> float:
        """[0, 1] proxy of how much of regulation has elapsed."""
        if self.is_pre_match:
            return 0.0
        if self.is_finished:
            return 1.0
        return min(1.0, self.minute / 90.0)


def build_match_state(
    fixture_payload: dict, events_payload: dict
) -> MatchState:
    """Combine /fixtures?id={id} + /fixtures/events?fixture={id} into MatchState.

    Args:
        fixture_payload: JSON dict from get_fixture_by_id().
        events_payload: JSON dict from get_fixture_events().

    Returns:
        Fully-populated MatchState.

    Raises:
        ValueError: when fixture payload has no response or malformed.
    """
    fixtures_resp = FixturesResponse.model_validate(fixture_payload)
    if not fixtures_resp.response:
        raise ValueError("Fixture payload has empty response list")
    fix: Fixture = fixtures_resp.response[0]

    events_resp = EventsResponse.model_validate(events_payload)

    # Tally cards by team
    h_yellow = a_yellow = h_red = a_red = 0
    home_id = fix.teams.home.id
    away_id = fix.teams.away.id
    for e in events_resp.response:
        if e.is_yellow_card:
            if e.team.id == home_id:
                h_yellow += 1
            elif e.team.id == away_id:
                a_yellow += 1
        elif e.is_red_card:
            if e.team.id == home_id:
                h_red += 1
            elif e.team.id == away_id:
                a_red += 1

    status_short = fix.fixture.status.short
    minute = fix.fixture.status.elapsed or 0

    return MatchState(
        fixture_id=fix.fixture_id,
        status=status_short,  # type: ignore[arg-type]
        minute=minute,
        home_team_id=home_id,
        away_team_id=away_id,
        home_team=fix.teams.home.name,
        away_team=fix.teams.away.name,
        home_goals=fix.goals.home or 0,
        away_goals=fix.goals.away or 0,
        home_yellow_cards=h_yellow,
        away_yellow_cards=a_yellow,
        home_red_cards=h_red,
        away_red_cards=a_red,
    )


def next_poll_seconds(state: MatchState) -> int:
    """Adaptive polling cadence per operator decision.

    Returns seconds until the next poll. 60s default; 15s when within 10
    minutes of HT (35-45) or FT (80-95).
    """
    if state.is_finished or state.is_pre_match:
        return 60
    m = state.minute
    if 35 <= m <= 45:
        return 15  # approaching HT
    if 80 <= m <= 95:
        return 15  # approaching FT
    return 60
