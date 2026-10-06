"""Game State Vector (GSV) — typed projection of a fixture-instant.

Section 4 of the v3 engine design notes. The GSV is NOT raw stats; it is a
structured projection over causally-relevant dimensions, with **derived
fields named explicitly** so downstream code can pattern-match on them
(``dominant_losing``, ``xg_vs_score_divergence``, ``game_phase``).

The architectural point: making these derivations explicit is the
difference between *reasoning* (an auditable rule that fires because
``score.dominant_losing=True``) and *averaging* (a black-box estimator
that maybe captures the same regime but cannot be explained or unit
tested).

This module defines ONLY the schema. Construction from
``LiveMatchState`` + ``MarketSnapshot`` + ``PreMatchPriors`` lives in
``gsv_builder.py``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")

# ── Enumerated literals ─────────────────────────────────────────────────

Period = Literal["NS", "1H", "HT", "2H", "ET1", "ET2", "PEN", "FT"]
PressingIntensity = Literal["low", "mid", "high"]
AttackZone = Literal["def_third", "middle", "att_third"]
TeamPhase = Literal[
    "parking_bus",
    "controlling",
    "pressing",
    "chasing",
    "collapsing",
]
GamePhase = Literal[
    "cagey_open",
    "open_attacking",
    "cagey_closed",
    "desperate",
    "cruise",
]
Tempo = Literal["low", "medium", "high"]
Direction = Literal["over", "under", "yes", "no", "home", "away", "draw"]


# ── Sub-state models ────────────────────────────────────────────────────


class RedCardEvent(BaseModel):
    model_config = _MODEL_CONFIG
    minute: int
    team_id: int
    player_id: int = 0


class SubstitutionEvent(BaseModel):
    model_config = _MODEL_CONFIG
    minute: int
    team_id: int
    player_in_id: int | None = None
    player_out_id: int | None = None
    role_signal: Literal["defensive", "offensive", "lateral", "unknown"] = "unknown"


class InjuryEvent(BaseModel):
    model_config = _MODEL_CONFIG
    minute: int
    team_id: int
    player_id: int = 0
    severity: Literal["minor", "major", "unknown"] = "unknown"


class FormationChange(BaseModel):
    model_config = _MODEL_CONFIG
    minute: int
    team_id: int
    from_formation: str
    to_formation: str


class PlayerOnYellow(BaseModel):
    model_config = _MODEL_CONFIG
    team_id: int
    player_id: int
    booked_at_minute: int


class ScoreState(BaseModel):
    """Score sub-state. ``dominant_losing`` is the load-bearing derived
    field — it is what gates the anti-Napoli rule (sec 6 rule #2)."""

    model_config = _MODEL_CONFIG
    home_goals: int
    away_goals: int
    goal_diff: int  # derived: home - away
    dominant_team_id: int | None = None  # pre-match favorite (ELO/market)
    dominant_losing: bool = False  # derived: favorite is trailing
    last_goal_minute: int | None = None
    last_goal_xg: float | None = None
    last_goal_team_id: int | None = None
    minutes_since_last_goal: float = 0.0


class TimeState(BaseModel):
    model_config = _MODEL_CONFIG
    minute: int
    period: Period
    added_time_estimate: float = 0.0
    time_remaining_half: float = 0.0  # derived
    time_remaining_match: float = 0.0  # derived


class NumericalState(BaseModel):
    """Player-count state. ``numerical_advantage`` codifies superiority
    explicitly so archetypes #1 and #11 can pattern-match directly."""

    model_config = _MODEL_CONFIG
    home_players: int = 11  # 11 - red cards home
    away_players: int = 11
    numerical_advantage: int = 0  # derived: home - away
    recent_red_card: RedCardEvent | None = None  # within last 5 min
    red_cards_home: int = 0
    red_cards_away: int = 0


class XGState(BaseModel):
    """Live xG state. ``xg_vs_score_divergence`` is the regression-to-xG
    signal — central to archetypes #2 and #5."""

    model_config = _MODEL_CONFIG
    home_xg_total: float = 0.0
    away_xg_total: float = 0.0
    xg_diff: float = 0.0  # derived
    xg_per_min_home_last_15: float = 0.0
    xg_per_min_away_last_15: float = 0.0
    xg_vs_score_divergence: float = 0.0  # derived: xg_diff - expected_xg_diff_given_goal_diff
    shots_total: tuple[int, int] = (0, 0)  # (home, away)
    shots_on_target: tuple[int, int] = (0, 0)
    shots_in_box: tuple[int, int] = (0, 0)
    big_chances: tuple[int, int] = (0, 0)


class FlowState(BaseModel):
    """Tempo & territorial state. ``pressing_intensity`` is a deliberately
    lossy enum — sec 4 note: enums beat dense vectors for auditability."""

    model_config = _MODEL_CONFIG
    possession_home_5min: float = 50.0  # rolling 5-min window (0-100)
    possession_home_match: float = 50.0
    attacks_last_10min: tuple[int, int] = (0, 0)
    dangerous_attacks_last_10min: tuple[int, int] = (0, 0)
    attack_zone_dominant: AttackZone | None = None
    pressing_intensity: PressingIntensity = "mid"  # derived


class CornerState(BaseModel):
    model_config = _MODEL_CONFIG
    corners_home: int = 0
    corners_away: int = 0
    corner_rate_last_15min: float = 0.0
    corners_pending: int = 0  # awarded but not taken (commentary parse)


class CardsState(BaseModel):
    model_config = _MODEL_CONFIG
    yellows: tuple[int, int] = (0, 0)
    reds: tuple[int, int] = (0, 0)
    card_rate_last_15min: float = 0.0
    players_on_yellow: list[PlayerOnYellow] = Field(default_factory=list)
    ref_card_rate_prior: float = 0.0  # ref's historical cards/game


class RosterState(BaseModel):
    """Roster & tactical-reset state. ``key_player_off`` enables
    archetype #9 (playmaker injured)."""

    model_config = _MODEL_CONFIG
    formation_home: str = "unknown"
    formation_away: str = "unknown"
    formation_changes: list[FormationChange] = Field(default_factory=list)
    subs_used: tuple[int, int] = (0, 0)
    subs_remaining: tuple[int, int] = (5, 5)
    recent_subs_5min: list[SubstitutionEvent] = Field(default_factory=list)
    injuries_live: list[InjuryEvent] = Field(default_factory=list)
    key_player_off: tuple[bool, bool] = (False, False)  # derived


class TacticalState(BaseModel):
    """Five-level phase enum per side + game-level + tempo. Sec 4 note:
    deliberately lossy. Auditability > information density for Tier A/B."""

    model_config = _MODEL_CONFIG
    home_phase: TeamPhase = "controlling"
    away_phase: TeamPhase = "controlling"
    game_phase: GamePhase = "open_attacking"
    tempo: Tempo = "medium"


class PreMatchPriors(BaseModel):
    model_config = _MODEL_CONFIG
    lambda_home_prematch: float = 1.35
    lambda_away_prematch: float = 1.15
    expected_corners_total: float = 10.4
    expected_cards_total: float = 3.95
    elo_diff: float = 0.0
    h2h_btts_rate: float = 0.5
    h2h_over25_rate: float = 0.5
    h2h_over_corners_95: float = 0.5
    bookmaker_consensus_close: dict[str, float] = Field(default_factory=dict)


class MarketLine(BaseModel):
    """One bookmaker line at one instant.

    ``line_history_5min`` carries the recent movement so the Mispricing
    Detector can distinguish *fresh* edge (line hasn't moved since the
    triggering event) from *stale* edge (line has had 5 min to update
    and didn't — probably our model is wrong, not the book)."""

    model_config = _MODEL_CONFIG
    market_id: str
    side_a_decimal: float
    side_b_decimal: float | None = None
    line_value: float | None = None
    max_stake_cap: float = 0.0
    last_update_utc: datetime | None = None
    # list of (timestamp, side_a_decimal, side_b_decimal)
    line_history_5min: list[tuple[datetime, float, float | None]] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    model_config = _MODEL_CONFIG
    lines: dict[str, MarketLine] = Field(default_factory=dict)


class CriticalEvent(BaseModel):
    """A pipeline-relevant event: goal, red card, key sub, key injury.

    The age field is what powers no-bet rule #3 (freshness): if the last
    critical event was <90s ago, the state hasn't stabilized yet."""

    model_config = _MODEL_CONFIG
    kind: Literal[
        "goal", "red_card", "yellow_red", "key_sub", "key_injury",
        "var_check", "penalty_awarded",
    ]
    minute: int
    team_id: int | None = None
    timestamp_utc: datetime | None = None


# ── Top-level GSV ───────────────────────────────────────────────────────


class GameStateVector(BaseModel):
    """Section 4 root model. One immutable frame of the match.

    Construction is via ``GSVBuilder`` (see ``gsv_builder.py``) — DO NOT
    instantiate by hand outside tests."""

    model_config = _MODEL_CONFIG

    fixture_id: int
    state_version: int  # monotonic
    timestamp_utc: datetime
    home_team_id: int
    away_team_id: int
    home_team_name: str = ""
    away_team_name: str = ""

    score: ScoreState
    time: TimeState
    numerical: NumericalState
    xg: XGState
    flow: FlowState
    corners: CornerState
    cards: CardsState
    roster: RosterState
    tactical: TacticalState
    priors: PreMatchPriors
    markets: MarketSnapshot

    last_critical_event: CriticalEvent | None = None
    last_critical_event_age_sec: float | None = None

    # Shadow dominant-team gap field (2026-05-14 forensic, task 5a).
    # Computed at _MARKET_DOM_MIN_PROB_GAP_SHADOW=0.025 by gsv_builder.build()
    # alongside the enforced dominant_team_id (which uses 0.04 gap).
    # Optional with default None so model_dump_json() stays backward-compatible
    # with existing gsv_log.parquet readers that don't know this field.
    # Enforced dominant_team_id (score.dominant_team_id) is UNCHANGED.
    shadow_dominant_team_id: int | None = None

    # ── tier-A derived helpers ──────────────────────────────────────────

    @property
    def is_dominant_losing(self) -> bool:
        """Alias for ``score.dominant_losing``. The Napoli predicate."""
        return self.score.dominant_losing

    @property
    def has_recent_critical_event(self) -> bool:
        """Rule #3 — last 90s window."""
        if self.last_critical_event_age_sec is None:
            return False
        return self.last_critical_event_age_sec < 90.0

    @property
    def is_open_game(self) -> bool:
        """Archetype #8 — 3+ goals before minute 60."""
        return (
            self.score.home_goals + self.score.away_goals >= 3
            and self.time.minute <= 60
        )

    @property
    def is_late_cagey_zero_zero(self) -> bool:
        """Archetype #3 — 0-0 at 75'+, low xG sum, both phases cagey.

        xG threshold lifted from 0.6 to 1.4 — internal research notes
        showed the empirical p25 of late-game 0-0 frames is 1.16 and
        median is 2.23; 0.6 was below p25 and captured ~0 real frames.
        """
        return (
            self.score.goal_diff == 0
            and self.score.home_goals == 0
            and self.time.minute >= 75
            and (self.xg.home_xg_total + self.xg.away_xg_total) < 1.4
            and self.tactical.game_phase == "cagey_closed"
        )


__all__ = [
    "AttackZone",
    "CardsState",
    "CornerState",
    "CriticalEvent",
    "Direction",
    "FlowState",
    "FormationChange",
    "GamePhase",
    "GameStateVector",
    "InjuryEvent",
    "MarketLine",
    "MarketSnapshot",
    "NumericalState",
    "Period",
    "PlayerOnYellow",
    "PreMatchPriors",
    "PressingIntensity",
    "RedCardEvent",
    "RosterState",
    "ScoreState",
    "SubstitutionEvent",
    "TacticalState",
    "TeamPhase",
    "Tempo",
    "TimeState",
    "XGState",
]
