"""Pydantic v2 parsers for API-Football v3 responses.

These models parse ONLY the fields TSP needs. API-Football returns much
richer payloads; we ignore everything we don't use to keep the schema
tight and reduce parse errors on optional-but-rarely-present fields.

Reference: https://www.api-football.com/documentation-v3

Three response types are parsed:
  1. /fixtures (FixturesResponse)         — list of matches with score + meta
  2. /fixtures/statistics (StatsResponse) — per-team stats per match
  3. /fixtures/events (EventsResponse)    — timestamped events (goals, cards)
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ─── /fixtures response ─────────────────────────────────────────────────


class FixtureStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")
    long: str
    short: str
    elapsed: int | None = None


class FixtureLeague(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    name: str
    country: str | None = None
    season: int


class FixtureTeam(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    name: str
    winner: bool | None = None
    """API-Football: True=winner, False=loser, None=draw or not finished."""


class FixtureTeams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    home: FixtureTeam
    away: FixtureTeam


class FixtureGoals(BaseModel):
    model_config = ConfigDict(extra="ignore")
    home: int | None = None
    away: int | None = None


class FixtureScore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    halftime: FixtureGoals
    fulltime: FixtureGoals
    extratime: FixtureGoals | None = None
    penalty: FixtureGoals | None = None


class FixtureCore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    date: datetime
    status: FixtureStatus


class Fixture(BaseModel):
    """A single fixture from API-Football /fixtures response."""

    model_config = ConfigDict(extra="ignore")
    fixture: FixtureCore
    league: FixtureLeague
    teams: FixtureTeams
    goals: FixtureGoals
    score: FixtureScore

    @property
    def fixture_id(self) -> int:
        return self.fixture.id

    @property
    def is_finished(self) -> bool:
        """True iff the match has a final result (FT or AET or PEN)."""
        return self.fixture.status.short in {"FT", "AET", "PEN", "AWD", "WO"}

    @property
    def ht_home_goals(self) -> int:
        return self.score.halftime.home or 0

    @property
    def ht_away_goals(self) -> int:
        return self.score.halftime.away or 0


class FixturesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: list[Fixture] = Field(default_factory=list)


# ─── /fixtures/statistics response ──────────────────────────────────────


class StatItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    """e.g. 'Shots on Goal', 'Total Shots', 'Fouls', 'Corner Kicks',
    'Yellow Cards', 'Red Cards', 'Ball Possession', 'Total passes',
    'Pass Accuracy', 'Offsides'."""
    value: int | float | str | None
    """API-Football returns mixed types — ints for counts, strings for
    percentages ('60%'), None when stat unavailable."""


class TeamStatsBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    team: FixtureTeam
    statistics: list[StatItem]

    def get(self, stat_type: str) -> int | float | None:
        """Return the numeric value of the named statistic.

        Strings like '60%' are parsed to float 60.0. None when missing.
        """
        for item in self.statistics:
            if item.type == stat_type:
                return _coerce_stat_value(item.value)
        return None


class StatsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: list[TeamStatsBlock] = Field(default_factory=list)

    def for_team(self, team_id: int) -> TeamStatsBlock | None:
        for block in self.response:
            if block.team.id == team_id:
                return block
        return None


def _coerce_stat_value(v: int | float | str | None) -> int | float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    # Strings: '60%', '60.0%', or numeric strings.
    s = v.strip()
    if not s:
        return None
    if s.endswith("%"):
        s = s[:-1]
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


# ─── /fixtures/events response ──────────────────────────────────────────


EventType = Literal["Goal", "Card", "subst", "Var"]
"""API-Football capitalization is inconsistent ('subst' lowercase, others
title case). We accept what they return."""


class EventTime(BaseModel):
    model_config = ConfigDict(extra="ignore")
    elapsed: int
    extra: int | None = None
    """Stoppage time added beyond elapsed (e.g., +3)."""

    @property
    def total_minute(self) -> int:
        """elapsed + extra clamped to 90 for first-half / 105 for second
        half boundaries (kept simple — extra rarely exceeds 10)."""
        return self.elapsed + (self.extra or 0)


class Event(BaseModel):
    """A timestamped event (goal, card, sub, VAR)."""

    model_config = ConfigDict(extra="ignore")
    time: EventTime
    team: FixtureTeam
    type: str
    """API-Football uses 'Goal', 'Card', 'subst' (lowercase!), 'Var'."""
    detail: str | None = None
    """For Card: 'Yellow Card' | 'Red Card'. For Goal: 'Normal Goal' |
    'Penalty' | 'Own Goal' | 'Missed Penalty'."""

    @property
    def is_goal(self) -> bool:
        return self.type == "Goal" and self.detail not in {"Missed Penalty"}

    @property
    def is_yellow_card(self) -> bool:
        return self.type == "Card" and self.detail == "Yellow Card"

    @property
    def is_red_card(self) -> bool:
        return self.type == "Card" and self.detail == "Red Card"


class EventsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: list[Event] = Field(default_factory=list)
