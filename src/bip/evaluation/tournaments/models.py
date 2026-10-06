"""Pydantic v2 models for tournament configuration and squad/player entities.

Loaded from YAML (configs/) and from API-Football payloads (via loaders).
The canonical_player_id is a string of form "af-{api_football_id}" so future
augmentation with FBref / StatsBomb IDs can be added without breaking
existing parquet schemas.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TournamentFormat(str, Enum):
    GROUPS_KNOCKOUT = "groups_knockout"
    KNOCKOUT_ONLY = "knockout_only"
    LEAGUE_FORMAT = "league_format"


class Confederation(str, Enum):
    UEFA = "uefa"
    CONMEBOL = "conmebol"
    AFC = "afc"
    CAF = "caf"
    CONCACAF = "concacaf"
    OFC = "ofc"


class Position(str, Enum):
    GOALKEEPER = "G"
    DEFENDER = "D"
    MIDFIELDER = "M"
    FORWARD = "F"


class Team(BaseModel):
    """National team."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_football_id: int
    name: str
    confederation: Confederation
    fifa_code: str = Field(min_length=3, max_length=3, description="3-letter FIFA code, e.g. BRA")


class Player(BaseModel):
    """Player record, canonical across data sources."""

    model_config = ConfigDict(extra="forbid")

    canonical_id: str = Field(pattern=r"^af-\d+$")
    api_football_id: int
    name: str
    dob: date | None = None
    position: Position
    national_team_id: int
    club_team_id: int | None = None
    club_league_id: int | None = None

    @classmethod
    def from_api_football(
        cls,
        af_player_id: int,
        name: str,
        position: Position,
        national_team_id: int,
        dob: date | None = None,
        club_team_id: int | None = None,
        club_league_id: int | None = None,
    ) -> Player:
        return cls(
            canonical_id=f"af-{af_player_id}",
            api_football_id=af_player_id,
            name=name,
            dob=dob,
            position=position,
            national_team_id=national_team_id,
            club_team_id=club_team_id,
            club_league_id=club_league_id,
        )


class Squad(BaseModel):
    """Roster called up for a tournament."""

    model_config = ConfigDict(extra="forbid")

    team_id: int
    tournament_slug: str
    players: list[Player]
    captured_at: datetime

    @field_validator("players")
    @classmethod
    def squad_size(cls, v: list[Player]) -> list[Player]:
        if not 18 <= len(v) <= 30:
            raise ValueError(f"Squad size {len(v)} outside expected range [18, 30]")
        return v


class Group(BaseModel):
    """Group-stage group."""

    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1, max_length=2, description="Group letter, e.g. 'A'")
    team_ids: list[int]

    @field_validator("team_ids")
    @classmethod
    def group_size(cls, v: list[int]) -> list[int]:
        if len(v) not in (3, 4):
            raise ValueError(f"Group must have 3 or 4 teams, got {len(v)}")
        if len(set(v)) != len(v):
            raise ValueError("Group cannot contain duplicate team_ids")
        return v


class TournamentMatch(BaseModel):
    """A single match within a tournament."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    home_team_id: int
    away_team_id: int
    kickoff: datetime
    stage: str = Field(description="'group:A', 'r32', 'r16', 'qf', 'sf', 'final', 'third_place'")
    api_football_fixture_id: int | None = None


class Tournament(BaseModel):
    """Full tournament config loaded from YAML.

    Groups and matches may be empty at config time and populated post-draw
    by the qualifying_loader. The format and participant_count are fixed
    by FIFA/UEFA before the draw.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    format: TournamentFormat
    season: int = Field(ge=2018, le=2030)
    start_date: date
    end_date: date
    participants_count: int = Field(ge=8, le=64)
    groups: list[Group] = Field(default_factory=list)
    matches: list[TournamentMatch] = Field(default_factory=list)
    teams_advancing_per_group: int | None = None
    best_third_placed_count: int = 0
    knockout_bracket_size: int | None = None

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:  # type: ignore[no-untyped-def]
        start = info.data.get("start_date")
        if start is not None and v < start:
            raise ValueError(f"end_date {v} before start_date {start}")
        return v


class QualifyingLeague(BaseModel):
    """API-Football league entry for one qualifier campaign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confederation: Confederation
    api_football_league_id: int
    season: int
    name: str
