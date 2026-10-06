"""Pydantic v2 models for Sportmonks v3 Football API responses.

Schemas are minimal and FORGIVING by default (extra='ignore') because
Sportmonks adds fields without notice. We pin only the fields our
predictor reads; the raw payload is also persisted to disk.

All models are frozen for safety — Sportmonks data is read-only here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


_MODEL_CONFIG = ConfigDict(frozen=True, extra="ignore")


def _ensure_utc(value: Any) -> Any:
    """Attach ``tzinfo=UTC`` to naive datetimes returned by Sportmonks.

    Sportmonks v3 emits timestamps as ISO-8601 WITHOUT offset (e.g.
    ``2026-05-12T19:30:00``). The API contract is that those are UTC,
    but Pydantic parses them as naive datetimes — which means any
    downstream ``.astimezone(...)`` call interprets the naive dt as
    SYSTEM-LOCAL time. On a Bogotá-localized machine that produces
    silent 5-hour offsets.

    This validator runs after parsing; it ONLY touches naive datetimes.
    Aware datetimes (which Sportmonks doesn't currently emit but might
    in the future) pass through untouched. Non-datetime inputs (e.g.
    epoch ints for ``Period.started``) also pass through.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ── Core fixture ─────────────────────────────────────────────────────────────


class ParticipantMeta(BaseModel):
    """Sportmonks per-participant metadata.

    ``location`` is the load-bearing field: it identifies which team is
    home and which is away, INDEPENDENT of array iteration order in the
    ``participants`` list. Sportmonks does NOT guarantee the array is
    home-first; ignoring this field silently mis-attributes scores,
    predictions, and any home/away-keyed downstream signal.

    Day-4 (2026-05-13) observed 9/31 fixtures where ``participants[0]``
    was the away team — most prominently Crystal Palace vs Man City,
    where the bug caused us to display the score "Palace 3-0 City" when
    the real result was the inverse.
    """

    model_config = _MODEL_CONFIG

    location: str | None = None  # 'home' | 'away'
    winner: bool | None = None
    position: int | None = None


class Participant(BaseModel):
    """One team in a fixture."""

    model_config = _MODEL_CONFIG

    id: int
    name: str
    short_code: str | None = None
    image_path: str | None = None
    meta: ParticipantMeta | None = None

    def is_home(self) -> bool | None:
        """Return True if Sportmonks marks this participant as home,
        False if marked away, None if meta absent."""
        if self.meta is None or self.meta.location is None:
            return None
        return self.meta.location == "home"


class FixtureState(BaseModel):
    """Current lifecycle state — INPLAY_*, FINISHED, NS, HT, etc."""

    model_config = _MODEL_CONFIG

    id: int
    state: str
    name: str
    short_name: str | None = None
    developer_name: str


class League(BaseModel):
    model_config = _MODEL_CONFIG

    id: int
    name: str
    country_id: int | None = None
    short_code: str | None = None


class Period(BaseModel):
    """Match period (1H / 2H / ET / Pen). Used to derive elapsed minute."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    type_id: int
    started: datetime | int | None = None  # epoch seconds when an int
    ended: datetime | int | None = None
    counts_from: int | None = None
    minutes: int | None = None

    _utc_started = field_validator("started", mode="after")(_ensure_utc)
    _utc_ended = field_validator("ended", mode="after")(_ensure_utc)
    seconds: int | None = None
    has_timer: bool | None = None
    ticking: bool | None = None
    description: str | None = None


class Score(BaseModel):
    """Per-period score per participant. Multiple types: 1H_SCORE, 2H_SCORE,
    CURRENT, FT_SCORE, etc."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    type_id: int
    participant_id: int | None = None
    score: dict[str, Any]  # {goals: int, participant: 'home'|'away'}
    description: str | None = None


# ── Statistics + trends ──────────────────────────────────────────────────────


class FixtureStatistic(BaseModel):
    """Aggregate stat for one team in one fixture (e.g. shots_total = 14)."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    type_id: int
    participant_id: int
    data: dict[str, Any]  # {value: number}
    location: str | None = None  # 'home' | 'away'

    @property
    def value(self) -> float | None:
        v = self.data.get("value") if self.data else None
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


class Trend(BaseModel):
    """Minute-by-minute statistic value. Same type_ids as FixtureStatistic
    but emitted per minute → time series."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    type_id: int
    participant_id: int
    period_id: int
    minute: int
    value: float | int | None = None


class PressureMinute(BaseModel):
    """Per-minute attack pressure index (Sportmonks proprietary, 0-100)."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    participant_id: int
    minute: int
    pressure: float


# ── Predictions + odds ───────────────────────────────────────────────────────


class Prediction(BaseModel):
    """One pre-built ML prediction for a market.

    The ``predictions`` body shape varies by ``type_id``:
    - 231 BTTS  → {yes, no}
    - 232 HT/FT → {home_home, home_draw, ..., away_away, draw_draw}
    - 233 First Half Winner → {home, draw, away}
    - 234 OU 1.5 → {yes, no}
    - 235 OU 2.5 → {yes, no}
    - 237 Fulltime Result → {home, draw, away}
    - 239 Double Chance → {draw_home, draw_away, home_away}
    - 240 Correct Score → {scores: {0-0, 0-1, ..., 5-5, other}}
    - 33 Value Bet → {bet, bookmaker, fair_odd, odd, stake, is_value}
    """

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    type_id: int
    predictions: dict[str, Any] = Field(default_factory=dict)


class Odd(BaseModel):
    """One bookmaker quote for one market selection."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    market_id: int
    bookmaker_id: int
    label: str
    value: str | None = None  # decimal odd string
    name: str | None = None
    sort_order: int | None = None
    market_description: str | None = None
    probability: str | None = None  # "12.5%" string
    handicap: str | None = None
    total: str | None = None
    participants: str | None = None
    suspended: bool = False
    stopped: bool = False
    winning: bool | None = None
    latest_bookmaker_update: datetime | None = None

    _utc_latest_bookmaker_update = field_validator(
        "latest_bookmaker_update", mode="after"
    )(_ensure_utc)

    @property
    def decimal_odd(self) -> float | None:
        if self.value is None:
            return None
        try:
            return float(self.value)
        except ValueError:
            return None

    @property
    def implied_prob(self) -> float | None:
        d = self.decimal_odd
        if d is None or d <= 1.0:
            return None
        return 1.0 / d


# ── Fixture (top-level) + LiveScore ──────────────────────────────────────────


class Event(BaseModel):
    """Match event (goal, yellow card, substitution, etc.)."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    # Sportmonks emits null period_id for some events (e.g. pre-match
    # cards, retroactive bookings, edge events without a clean period
    # attribution). Made optional in 2026-05-10 hotfix after validation
    # errors observed during the first live verification run.
    period_id: int | None = None
    participant_id: int | None = None
    type_id: int
    minute: int | None = None
    extra_minute: int | None = None
    player_id: int | None = None
    related_player_id: int | None = None
    player_name: str | None = None


class Lineup(BaseModel):
    """Team lineup entry (player + position)."""

    model_config = _MODEL_CONFIG

    id: int
    fixture_id: int
    player_id: int | None = None
    team_id: int
    type_id: int  # starting / bench / unavailable
    position_id: int | None = None
    jersey_number: int | None = None
    formation_position: int | None = None
    player_name: str | None = None


class Fixture(BaseModel):
    """A single fixture with optional includes attached.

    Includes that may be present (per request): participants, state,
    league, season, periods, scores, statistics, trends, pressure,
    predictions, odds, events, lineups, comments.
    """

    model_config = _MODEL_CONFIG

    id: int
    sport_id: int
    league_id: int
    season_id: int
    starting_at: datetime | None = None
    name: str | None = None
    state_id: int | None = None
    venue_id: int | None = None
    result_info: str | None = None
    leg: str | None = None
    has_odds: bool | None = None
    has_premium_odds: bool | None = None

    _utc_starting_at = field_validator("starting_at", mode="after")(_ensure_utc)

    # Includes (None when not requested)
    participants: list[Participant] | None = None
    state: FixtureState | None = None
    league: League | None = None
    periods: list[Period] | None = None
    scores: list[Score] | None = None
    events: list[Event] | None = None
    lineups: list[Lineup] | None = None
    statistics: list[FixtureStatistic] | None = None
    trends: list[Trend] | None = None
    pressure: list[PressureMinute] | None = None
    predictions: list[Prediction] | None = None
    odds: list[Odd] | None = None

    # Convenience accessors

    def _home_away_from_scores(self) -> tuple[int, int] | None:
        """Internal helper: derive (home_id, away_id) from ``scores``.

        Mirrors ``_home_away_from_scores`` in ``match_state.py``. Kept
        on Fixture so convenience accessors stay self-contained.
        """
        if not self.scores:
            return None
        home_id = away_id = None
        for s in self.scores:
            if s.participant_id is None:
                continue
            body = s.score or {}
            loc = body.get("participant")
            if loc == "home" and home_id is None:
                home_id = s.participant_id
            elif loc == "away" and away_id is None:
                away_id = s.participant_id
            if home_id is not None and away_id is not None:
                return home_id, away_id
        return None

    def _home_away_from_statistics(self) -> tuple[int, int] | None:
        """Internal helper: derive (home_id, away_id) from ``statistics[i].location``."""
        if not self.statistics:
            return None
        home_id = away_id = None
        for s in self.statistics:
            if s.participant_id is None or s.location is None:
                continue
            if s.location == "home" and home_id is None:
                home_id = s.participant_id
            elif s.location == "away" and away_id is None:
                away_id = s.participant_id
            if home_id is not None and away_id is not None:
                return home_id, away_id
        return None

    def home_team(self) -> Participant | None:
        """Return the home participant.

        Resolution order: ``participants[i].meta.location`` →
        ``scores[i].score.participant`` × ``participant_id`` →
        ``statistics[i].location`` × ``participant_id`` →
        ``participants[0]`` (last-resort fallback).

        Day-4 (2026-05-13) regression: ``participants[0]`` is NOT
        guaranteed to be the home team (Sportmonks may return either
        order). Crystal Palace vs Man City exposed this — Palace was
        ``participants[0]`` but City was the real home (Etihad).
        """
        if not self.participants:
            return None
        # Layer 1: meta.location
        for p in self.participants:
            if p.is_home() is True:
                return p
        any_meta = any(p.meta is not None for p in self.participants)
        if not any_meta:
            # Layer 2: scores
            pair = self._home_away_from_scores()
            if pair is not None:
                home_id, _ = pair
                for p in self.participants:
                    if p.id == home_id:
                        return p
            # Layer 3: statistics
            pair = self._home_away_from_statistics()
            if pair is not None:
                home_id, _ = pair
                for p in self.participants:
                    if p.id == home_id:
                        return p
            # Layer 4: array-order fallback
            return self.participants[0]
        return None

    def away_team(self) -> Participant | None:
        """Return the away participant. Mirrors ``home_team`` semantics."""
        if not self.participants or len(self.participants) < 2:
            return None
        # Layer 1: meta.location
        for p in self.participants:
            if p.is_home() is False:
                return p
        any_meta = any(p.meta is not None for p in self.participants)
        if not any_meta:
            pair = self._home_away_from_scores()
            if pair is not None:
                _, away_id = pair
                for p in self.participants:
                    if p.id == away_id:
                        return p
            pair = self._home_away_from_statistics()
            if pair is not None:
                _, away_id = pair
                for p in self.participants:
                    if p.id == away_id:
                        return p
            return self.participants[1]
        return None

    def is_live(self) -> bool:
        if self.state is None:
            return False
        return self.state.developer_name.startswith("INPLAY")

    def current_minute(self) -> int | None:
        """Best-effort current minute: max minute across active periods."""
        if not self.periods:
            return None
        active = [p for p in self.periods if p.ticking or (p.minutes and not p.ended)]
        if not active:
            # FT or HT — return the last started minute
            ms = [p.minutes for p in self.periods if p.minutes is not None]
            return max(ms) if ms else None
        return max((p.minutes or 0) for p in active)


class LiveScore(BaseModel):
    """A livescore is structurally a Fixture (with state/scores). We keep
    the alias for clarity in callers."""

    model_config = _MODEL_CONFIG

    id: int
    league_id: int
    starting_at: datetime | None = None
    state_id: int | None = None

    _utc_starting_at = field_validator("starting_at", mode="after")(_ensure_utc)


# ── API envelope ─────────────────────────────────────────────────────────────


class Pagination(BaseModel):
    model_config = ConfigDict(extra="ignore")

    count: int | None = None
    per_page: int | None = None
    current_page: int | None = None
    next_page: str | None = None
    has_more: bool | None = None


class RateLimitInfo(BaseModel):
    """Sportmonks per-entity rate limit reporting (per response meta)."""

    model_config = ConfigDict(extra="ignore")

    resets_in_seconds: int | None = None
    remaining: int | None = None
    requested_entity: str | None = None
