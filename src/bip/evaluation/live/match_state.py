"""LiveMatchState — frozen snapshot of a fixture at one point in time.

Built from a Sportmonks ``Fixture`` populated with the standard live
includes (state, periods, scores, statistics, trends, pressure,
predictions, events, participants).

The state EXPOSES derived signals that the predictor consumes:
- Current score per side
- Elapsed minute (best-effort from periods)
- Per-team aggregate stats by name (`shots_on_target`, `dangerous_attacks`, etc.)
- Recent-window pressure averages (last 5/10 min)
- Goal-event timeline (for Dixon-Robinson scoring)
- Red card / yellow card / substitution event timelines
- Sportmonks predictions, indexed by type_id (for baseline lookup)
- Cached Sportmonks correct-score grid (for cross-market sanity)
- Informational-density score (gates live-adjusted picks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from bip.sports.football.sportmonks.schemas import (
    Fixture,
    FixtureStatistic,
    Prediction,
    PressureMinute,
    Trend,
)
from bip.sports.football.sportmonks.types import (
    PERIOD_FIRST_HALF,
    PERIOD_SECOND_HALF,
    PredictionType,
    StatType,
)

# Forward reference: ``TeamForm`` is imported lazily inside the dataclass to
# avoid a circular import (team_form.py depends on this module's schemas).
if False:  # pragma: no cover - typing only
    from bip.evaluation.live.team_form import TeamForm


# Stat IDs used as live "shot quality" signal — the proxy for live xG when
# Sportmonks does not populate xGFixture for the league.
SHOT_QUALITY_STAT_IDS: tuple[int, ...] = (
    StatType.SHOTS_ON_TARGET,        # 86
    StatType.SHOTS_INSIDEBOX,        # 49
    StatType.BIG_CHANCES_CREATED,    # 580
    StatType.DANGEROUS_ATTACKS,      # 44 (denominator-ish)
)

# Empirical xG weights per shot/event type (literature consensus + sanity):
# Big chance ≈ 0.30 xG, Shot inside box ≈ 0.10, Shot on target ≈ 0.07,
# Dangerous attack ≈ 0.005. These are conservative — calibrated for
# proxy purposes, not used as primary xG when Sportmonks emits real xG.
XG_PROXY_WEIGHTS: dict[int, float] = {
    StatType.SHOTS_ON_TARGET: 0.07,
    StatType.SHOTS_INSIDEBOX: 0.05,  # incremental over shots_on_target
    StatType.BIG_CHANCES_CREATED: 0.20,  # incremental over inside-box
    StatType.DANGEROUS_ATTACKS: 0.005,
}


# Event type_ids (verified 2026-05-09 against /core/types).
EVENT_TYPE_GOAL = 14
EVENT_TYPE_OWNGOAL = 15
EVENT_TYPE_PENALTY_SCORED = 16
EVENT_TYPE_MISSED_PENALTY = 17
EVENT_TYPE_SUBSTITUTION = 18
EVENT_TYPE_YELLOWCARD = 19
EVENT_TYPE_REDCARD = 20
EVENT_TYPE_YELLOWREDCARD = 21


# Score-grid dimension for the cached Sportmonks correct-score table.
_SCORE_GRID_MAX_N = 8


@dataclass(frozen=True)
class LiveMatchState:
    """Immutable snapshot — one frame of one match."""

    fixture_id: int
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    home_goals: int
    away_goals: int
    minute: int          # 0 if not yet started; clamps at 90/120
    period_id: int       # 0 if NS / FT, else 1 / 2 / 38 / 39
    is_live: bool
    is_half_time: bool
    is_finished: bool

    # Per-team aggregate stats (indexed by StatType enum value)
    home_stats: dict[int, float] = field(default_factory=dict)
    away_stats: dict[int, float] = field(default_factory=dict)

    # Pressure: per-minute samples, last N minutes window stored
    home_pressure_recent: list[float] = field(default_factory=list)
    away_pressure_recent: list[float] = field(default_factory=list)

    # Sportmonks pre-built predictions, indexed by type_id
    sportmonks_predictions: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Minute-by-minute trend timeline (cumulative stat values per type×team).
    # Sportmonks emits Trend records as (type_id, participant_id, minute, value).
    # We store the raw list and compute rolling-window deltas on demand —
    # avoids precomputing every (type, side, window) combination.
    trends: list[Trend] = field(default_factory=list)

    # Goal-event timeline (minute, scoring_team_id) — sorted ascending
    goal_events: list[tuple[int, int]] = field(default_factory=list)

    # Goal-event timeline WITH player_id preserved (minute, team_id, player_id).
    # Same goals as ``goal_events`` but with the scoring player attribution
    # retained for retrospective analysis. ``player_id`` may be 0 when
    # Sportmonks omits it (e.g., own goals attributed to whole team).
    goal_events_detailed: list[tuple[int, int, int]] = field(default_factory=list)

    # Red-card events (minute, team_affected_id)
    red_card_events: list[tuple[int, int]] = field(default_factory=list)

    # Yellow-card events (minute, team_id, player_id) — player_id is 0 when absent.
    # Used for booked-player tracking (yellow→red risk) and cards-market signal.
    yellow_card_events: list[tuple[int, int, int]] = field(default_factory=list)

    # Substitution events (minute, team_id, player_in_id-or-None).
    # Manager-intent signal: triple-sub by 70' = pushing for goal;
    # lone defensive sub at 80' = locking the result.
    substitution_events: list[tuple[int, int, int]] = field(default_factory=list)

    # Commentary-narrative events (VAR_CHECK, RED_CARD, INJURY_DELAY, etc.)
    # extracted from raw.comments via commentary.extract_events. Used by
    # ValueDetector's commentary cool-off gate to suppress picks during
    # market-disruption windows. Empty when raw_comments not provided to
    # from_fixture(), preserving backwards compatibility.
    commentary_events: list = field(default_factory=list)

    # Snapshot meta (best-effort; not always emitted by Sportmonks).
    snapshot_taken_at: datetime | None = None
    league_id: int | None = None
    season_id: int | None = None

    # Recent-form signals (resolved by the watcher via ``TeamFormCache``).
    # Either may be None when the team has too few completed fixtures or
    # when the fetch failed. The predictor reads these in the goal-timing
    # market emitters to tilt pre-match priors away from the league
    # baseline when a team's recent pattern is materially different.
    home_team_form: Any | None = None  # bip.evaluation.live.team_form.TeamForm
    away_team_form: Any | None = None

    # ── derived signals ─────────────────────────────────────────────────

    @property
    def score_diff_home(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def remaining_minutes(self) -> int:
        """Conservative remaining-minutes estimate (caps at full match length).

        Treats halftime as paused — remaining = 90 - minute. Extra time and
        injury time are not modelled here.
        """
        if self.is_finished:
            return 0
        if self.minute >= 90:
            return 0
        return max(0, 90 - self.minute)

    @property
    def home_shot_quality_xg_proxy(self) -> float:
        """Sum of weighted shot-quality stats for home side."""
        return _xg_proxy(self.home_stats)

    @property
    def away_shot_quality_xg_proxy(self) -> float:
        return _xg_proxy(self.away_stats)

    @property
    def home_pressure_avg(self) -> float:
        return (
            sum(self.home_pressure_recent) / len(self.home_pressure_recent)
            if self.home_pressure_recent else 0.0
        )

    @property
    def away_pressure_avg(self) -> float:
        return (
            sum(self.away_pressure_recent) / len(self.away_pressure_recent)
            if self.away_pressure_recent else 0.0
        )

    @property
    def home_pressure_trend(self) -> float:
        """Difference between recent-half and earlier-half of pressure window.

        Positive = pressure rising; negative = falling. Zero when window
        too small to derive a trend. Used to distinguish a team that's
        BUILDING attack from one that's tired and DROPPING off.
        """
        return _trend_split(self.home_pressure_recent)

    @property
    def away_pressure_trend(self) -> float:
        return _trend_split(self.away_pressure_recent)

    @property
    def has_red_card_home(self) -> bool:
        return any(t == self.home_team_id for _, t in self.red_card_events)

    @property
    def has_red_card_away(self) -> bool:
        return any(t == self.away_team_id for _, t in self.red_card_events)

    @property
    def red_card_minute_home(self) -> int | None:
        """Earliest red-card minute for home team (or None if no red)."""
        homes = [m for m, t in self.red_card_events if t == self.home_team_id]
        return min(homes) if homes else None

    @property
    def red_card_minute_away(self) -> int | None:
        aways = [m for m, t in self.red_card_events if t == self.away_team_id]
        return min(aways) if aways else None

    # ── Stat-derived signals (B-1, B-2, B-3, B-5, B-6, B-7, B-9, B-11) ──

    @property
    def home_corners(self) -> int:
        return int(self.home_stats.get(StatType.CORNERS, 0))

    @property
    def away_corners(self) -> int:
        return int(self.away_stats.get(StatType.CORNERS, 0))

    @property
    def home_possession(self) -> float:
        """Possession % (0-100). Defaults to 50 when stat missing."""
        return float(self.home_stats.get(StatType.BALL_POSSESSION, 50.0))

    @property
    def away_possession(self) -> float:
        return float(self.away_stats.get(StatType.BALL_POSSESSION, 50.0))

    @property
    def home_shots_total(self) -> int:
        return int(self.home_stats.get(StatType.SHOTS_TOTAL, 0))

    @property
    def away_shots_total(self) -> int:
        return int(self.away_stats.get(StatType.SHOTS_TOTAL, 0))

    @property
    def home_key_passes(self) -> int:
        return int(self.home_stats.get(StatType.KEY_PASSES, 0))

    @property
    def away_key_passes(self) -> int:
        return int(self.away_stats.get(StatType.KEY_PASSES, 0))

    @property
    def home_saves(self) -> int:
        return int(self.home_stats.get(StatType.SAVES, 0))

    @property
    def away_saves(self) -> int:
        return int(self.away_stats.get(StatType.SAVES, 0))

    @property
    def home_big_chances_missed(self) -> int:
        return int(self.home_stats.get(StatType.BIG_CHANCES_MISSED, 0))

    @property
    def away_big_chances_missed(self) -> int:
        return int(self.away_stats.get(StatType.BIG_CHANCES_MISSED, 0))

    @property
    def home_big_chances_created(self) -> int:
        return int(self.home_stats.get(StatType.BIG_CHANCES_CREATED, 0))

    @property
    def away_big_chances_created(self) -> int:
        return int(self.away_stats.get(StatType.BIG_CHANCES_CREATED, 0))

    @property
    def home_injuries(self) -> int:
        return int(self.home_stats.get(StatType.INJURIES, 0))

    @property
    def away_injuries(self) -> int:
        return int(self.away_stats.get(StatType.INJURIES, 0))

    @property
    def home_engagement(self) -> int:
        """Tackles + interceptions + duels-won — defensive engagement.

        High engagement signals a scrappy game where the underlying goal
        rate compresses (defenders winning the ball before xG accumulates).
        """
        return (
            int(self.home_stats.get(StatType.TACKLES, 0))
            + int(self.home_stats.get(StatType.INTERCEPTIONS, 0))
            + int(self.home_stats.get(StatType.DUELS_WON, 0))
        )

    @property
    def away_engagement(self) -> int:
        return (
            int(self.away_stats.get(StatType.TACKLES, 0))
            + int(self.away_stats.get(StatType.INTERCEPTIONS, 0))
            + int(self.away_stats.get(StatType.DUELS_WON, 0))
        )

    # ── Yellow-card / booked-player signals (B-29) ─────────────────────

    # ── Trend-derived rolling-window signals ───────────────────────────

    def _cumulative_at_minute(
        self, side: str, type_id: int, target_minute: int,
    ) -> float:
        """Cumulative stat value at or before ``target_minute`` for one side.

        Walks the ``trends`` list and returns the latest emitted value with
        ``minute ≤ target_minute``. Returns 0.0 when no record exists —
        which is also the correct semantic for "stat not yet accumulated."
        """
        pid = self.home_team_id if side == "home" else self.away_team_id
        best_minute = -1
        best_value = 0.0
        for t in self.trends:
            if t.type_id != type_id or t.participant_id != pid:
                continue
            if t.minute > target_minute:
                continue
            if t.minute > best_minute:
                best_minute = t.minute
                v = t.value
                if v is not None:
                    best_value = float(v)
        return best_value

    def stat_in_last_window(
        self, side: str, type_id: int, *, window: int = 5,
    ) -> int:
        """Generic rolling-window count from cumulative trend records.

        Computes ``cumulative(type_id, side, minute) - cumulative(type_id,
        side, minute-window)``. Returns 0 when no trends or when the stat
        wasn't emitted for this fixture. All per-stat helpers below are
        thin wrappers — keeps a single code path for the delta math.
        """
        if not self.trends:
            return 0
        current = self._cumulative_at_minute(side, type_id, self.minute)
        prior = self._cumulative_at_minute(
            side, type_id, max(0, self.minute - window),
        )
        return max(0, int(round(current - prior)))

    def shots_in_last_window(self, side: str, *, window: int = 5) -> int:
        """Total shots in last ``window`` match-minutes (graceful: 0 when
        no trends data — same as for the helpers below)."""
        return self.stat_in_last_window(side, StatType.SHOTS_TOTAL, window=window)

    def dangerous_attacks_in_last_window(
        self, side: str, *, window: int = 5,
    ) -> int:
        return self.stat_in_last_window(side, StatType.DANGEROUS_ATTACKS, window=window)

    def key_passes_in_last_window(self, side: str, *, window: int = 5) -> int:
        return self.stat_in_last_window(side, StatType.KEY_PASSES, window=window)

    def corners_in_last_window(self, side: str, *, window: int = 5) -> int:
        return self.stat_in_last_window(side, StatType.CORNERS, window=window)

    def fouls_in_last_window(self, side: str, *, window: int = 5) -> int:
        """Recent foul rate — drives cards-market λ when game tempo
        becomes aggressive (referee state, late-game tactical fouls)."""
        return self.stat_in_last_window(side, StatType.FOULS, window=window)

    def yellow_cards_in_last_window(self, side: str, *, window: int = 5) -> int:
        """Yellow-card cluster detection: 3+ yellows in 15 min often
        signals a card-happy referee, raises P(more cards before FT)."""
        return self.stat_in_last_window(side, StatType.YELLOW_CARDS, window=window)

    def crosses_in_last_window(self, side: str, *, window: int = 5) -> int:
        """Wing-attack pattern. Combined with corners, drives the
        set-piece intensity signal (set-pieces account for ~30% of goals
        in top-5 leagues)."""
        return self.stat_in_last_window(side, StatType.TOTAL_CROSSES, window=window)

    def shots_on_target_in_last_window(
        self, side: str, *, window: int = 5,
    ) -> int:
        return self.stat_in_last_window(side, StatType.SHOTS_ON_TARGET, window=window)

    def set_piece_intensity(self, side: str, *, window: int = 15) -> float:
        """Combined corner + cross frequency in the last ``window`` min,
        normalised against a league baseline (~5 set-pieces per side per
        90 min). Returns ratio: > 1.0 = team generating set-pieces
        above expected pace; < 1.0 below.

        Returns 1.0 (neutral) when no trends or minute too low.
        """
        if not self.trends or self.minute < 20:
            return 1.0
        recent = (
            self.corners_in_last_window(side, window=window)
            + self.crosses_in_last_window(side, window=window)
        )
        # League prior: ~5 corners + ~16 crosses = 21 set-pieces per
        # team per 90. Per-minute rate ≈ 0.23.
        expected_per_min = 0.23
        actual_window_minutes = min(window, self.minute)
        expected_recent = expected_per_min * actual_window_minutes
        if expected_recent <= 0.0:
            return 1.0
        return recent / expected_recent

    def shot_acceleration(self, side: str) -> float:
        """Second-derivative of shot rate: shots in last 3 min vs shots in
        the 5 minutes BEFORE that (min ``minute-8`` to ``minute-3``).

        Returns:
            > 1.0 → accelerating (rising shot rate)
            ~ 1.0 → steady pace
            < 1.0 → decelerating (cooling off)
            0.0   → no recent shots OR no trends data

        Used to:
        - Confirm/reject killing-clock detection (a team in clock-killing
          mode should be DECELERATING; if their accel is > 0.9 the
          detector is over-firing)
        - Scale the late-game trailing-team push proportionally to the
          actual measured push intensity
        - Distinguish "team peaked and stopped" from "team starting to push"
        """
        if not self.trends or self.minute < 8:
            return 0.0
        # Last 3 min: minute - 3 to minute
        recent = (
            self._cumulative_at_minute(side, StatType.SHOTS_TOTAL, self.minute)
            - self._cumulative_at_minute(side, StatType.SHOTS_TOTAL, max(0, self.minute - 3))
        )
        # Previous 5 min: minute - 8 to minute - 3
        prior = (
            self._cumulative_at_minute(side, StatType.SHOTS_TOTAL, max(0, self.minute - 3))
            - self._cumulative_at_minute(side, StatType.SHOTS_TOTAL, max(0, self.minute - 8))
        )
        recent_rate = max(0, recent) / 3.0
        prior_rate = max(0, prior) / 5.0
        if prior_rate <= 0.0 and recent_rate <= 0.0:
            return 0.0
        if prior_rate <= 0.0:
            # No prior shots, but recent ones → strong acceleration
            return 2.0
        return recent_rate / prior_rate

    def momentum_score(self, side: str, *, window: int = 5) -> float:
        """Composite attacking-momentum score combining shots, dangerous
        attacks, and key passes rolling-window rates against match-average
        rates.

        Returns a multiplier in roughly [0.5, 1.6]:
            < 0.85 → team has slowed below their match-average pace
            ~ 1.0  → team is at their typical match-pace
            > 1.15 → team is materially accelerating

        The composite is more robust than shots-only because:
        - Shots can spike from desperate long-range attempts (false positive)
        - Dangerous attacks and key passes confirm true attacking pressure
        - Three rising stats together = signal; one rising = noise

        Returns 1.0 (neutral) when no trends data exists or minute too low.
        """
        if not self.trends or self.minute < 15:
            return 1.0
        # Recent-window aggregate rate (per minute)
        recent_shots = self.shots_in_last_window(side, window=window)
        recent_da = self.dangerous_attacks_in_last_window(side, window=window)
        recent_kp = self.key_passes_in_last_window(side, window=window)

        # Match-average rate (per minute) — cumulative ÷ minute
        cum_shots = self._cumulative_at_minute(side, StatType.SHOTS_TOTAL, self.minute)
        cum_da = self._cumulative_at_minute(side, StatType.DANGEROUS_ATTACKS, self.minute)
        cum_kp = self._cumulative_at_minute(side, StatType.KEY_PASSES, self.minute)

        # Per-stat ratios, neutral=1.0 when stat-rate not estimable
        def _ratio(recent: float, cum: float) -> float:
            avg_per_min = cum / self.minute if self.minute > 0 else 0.0
            recent_per_min = recent / window if window > 0 else 0.0
            if avg_per_min <= 0.0:
                return 1.0
            return recent_per_min / avg_per_min

        # Weighted geometric mean — shots 50%, DA 30%, KP 20%.
        # Geometric mean keeps the composite bounded when one stat is 0
        # (clamped to 0.1) and avoids the arithmetic-mean pathology where
        # one extreme outlier dominates.
        import math as _m
        weights = (0.50, 0.30, 0.20)
        ratios = (
            max(0.1, _ratio(recent_shots, cum_shots)),
            max(0.1, _ratio(recent_da, cum_da)),
            max(0.1, _ratio(recent_kp, cum_kp)),
        )
        log_sum = sum(w * _m.log(r) for w, r in zip(weights, ratios))
        return _m.exp(log_sum)

    @property
    def yellow_card_count_home(self) -> int:
        return sum(1 for _m, t, _p in self.yellow_card_events if t == self.home_team_id)

    @property
    def yellow_card_count_away(self) -> int:
        return sum(1 for _m, t, _p in self.yellow_card_events if t == self.away_team_id)

    @property
    def players_booked_home(self) -> set[int]:
        return {p for _m, t, p in self.yellow_card_events
                if t == self.home_team_id and p}

    @property
    def players_booked_away(self) -> set[int]:
        return {p for _m, t, p in self.yellow_card_events
                if t == self.away_team_id and p}

    # ── Substitution signals (B-30) ────────────────────────────────────

    @property
    def substitutions_home(self) -> int:
        return sum(1 for _m, t, _p in self.substitution_events if t == self.home_team_id)

    @property
    def substitutions_away(self) -> int:
        return sum(1 for _m, t, _p in self.substitution_events if t == self.away_team_id)

    # ── Killing-the-clock detector (B-3) ───────────────────────────────

    def is_killing_clock(self, side: str) -> bool:
        """Detect "high possession but no penetration" pattern.

        Triggers when a team has ≥65% possession from min ≥60, but their
        attacking productivity per minute of ball is suspiciously low. In
        that pattern they are protecting a result, NOT going for a goal,
        so over-goals + draw-pushing nudges should be suppressed.

        Productivity = (shots_total + key_passes) / minutes_of_possession,
        where minutes_of_possession ≈ minute × possession_pct/100. League
        average for attacking teams is ~0.40-0.70 actions per poss-minute;
        clock-killing sits significantly below that band.

        Trend confirmation: if the instantaneous detector fires AND we
        have trends data, we additionally require that shot rate is
        actually DECELERATING (shot_acceleration < 0.9). This rejects the
        false-positive case where a team that was attacking hard 20 min
        ago has high cumulative possession but is currently still pushing
        — they aren't really killing clock yet.
        """
        if self.minute < 60:
            return False
        if side == "home":
            poss, st, kp = self.home_possession, self.home_shots_total, self.home_key_passes
        else:
            poss, st, kp = self.away_possession, self.away_shots_total, self.away_key_passes
        if poss < 65.0:
            return False
        poss_minutes = self.minute * (poss / 100.0)
        if poss_minutes < 5.0:
            return False  # too small a sample for a stable rate
        productivity = (st + kp) / poss_minutes
        if productivity >= 0.20:
            return False
        # Instantaneous gate fired — confirm via trend if available.
        # No trends data → trust instantaneous detector (preserves prior
        # behavior for leagues without minute-by-minute coverage).
        if self.trends:
            accel = self.shot_acceleration(side)
            # accel > 0.9 means shots NOT decelerating → team still pushing,
            # not actually killing clock. Reject.
            if accel > 0.9:
                return False
        return True

    # ── Information-content gate (B-7 / B-G3) ──────────────────────────

    @property
    def informational_density(self) -> float:
        """Score in [0, 1] — how much real live signal has accumulated.

        Used by the predictor to decide whether to live-adjust at all.
        At low density we return Sportmonks pre-match priors verbatim
        rather than rebuild from per-team OU + pressure.

        Formula::

            minute_term = min(1, minute / 25)
            data_term   = max(pressure_term, stat_term, event_term)
            density     = minute_term × (0.4 + 0.6 × data_term)

        Match events (goals, red cards) feed ``event_term`` — a strong
        signal that accelerates density to full credit ON THE DATA AXIS.
        But they DO NOT bypass the minute floor: a red card at minute 5
        still leaves minute_term=0.20, so density caps at 0.20 — below
        the gate floor — until enough match time has elapsed for stats
        and pressure to complement the event.

        The previous override to 1.0 on any event was a bug: it allowed
        live-adjusted picks to emit at minute 5-15 with no statistical
        accumulation, recreating the cluster-C "min-8 burst" via early
        events.
        """
        minute_term = min(1.0, max(0, self.minute) / 25.0)
        n_pressure = len(self.home_pressure_recent) + len(self.away_pressure_recent)
        pressure_term = min(1.0, n_pressure / 8.0)  # 4 samples per side = full
        stat_events = sum(
            self.home_stats.get(t, 0) + self.away_stats.get(t, 0)
            for t in (
                StatType.SHOTS_TOTAL, StatType.KEY_PASSES,
                StatType.CORNERS, StatType.SHOTS_ON_TARGET,
            )
        )
        stat_term = min(1.0, stat_events / 12.0)
        event_term = 1.0 if (
            self.red_card_events
            or (self.home_goals + self.away_goals) >= 1
        ) else 0.0
        data_term = max(pressure_term, stat_term, event_term)
        return minute_term * (0.4 + 0.6 * data_term)

    # ── Sportmonks correct-score grid (B-16) ───────────────────────────

    @property
    def sportmonks_score_grid(self) -> np.ndarray | None:
        """Build a (max_n+1, max_n+1) probability grid from Sportmonks
        ``CORRECT_SCORE_PROBABILITY`` (type_id 240).

        Body shape (verified): ``{scores: {"0-0": 8.4, "1-0": 11.2, ..., "other": 4.1}}``.
        Values are percentages 0-100. The "other" bucket is ignored for
        marginal calculations — its mass is redistributed via final
        normalisation. None when Sportmonks didn't emit the prediction.
        """
        sm = self.sportmonks_predictions.get(PredictionType.CORRECT_SCORE_PROBABILITY)
        if not sm:
            return None
        scores = sm.get("scores") if isinstance(sm, dict) else None
        if not isinstance(scores, dict):
            return None
        n = _SCORE_GRID_MAX_N + 1
        grid = np.zeros((n, n))
        for k, v in scores.items():
            if not isinstance(k, str) or k == "other":
                continue
            try:
                h_str, a_str = k.split("-")
                h, a = int(h_str), int(a_str)
            except (ValueError, IndexError):
                continue
            if 0 <= h < n and 0 <= a < n:
                try:
                    grid[h, a] = float(v) / 100.0
                except (TypeError, ValueError):
                    continue
        total = grid.sum()
        if total <= 0:
            return None
        return grid / total

    # ── Live xG signal (computed vs expected) ───────────────────────────

    def home_live_xg_signal(self) -> float:
        """Live xG performance vs minute-prorated expectation.

        Positive value means home team has out-performed their
        pre-match expected goal-creation rate up to current minute;
        negative means they've under-performed. Range typically
        ±1.5 in active matches.

        Uses the operator-tuned XG_PROXY_WEIGHTS to estimate live
        creation, and compares against (Sportmonks pre-match λ_home) ×
        (minute / 90) as the expected level for this point in time.
        """
        return self._live_xg_signal(self.home_stats, self.home_team_id)

    def away_live_xg_signal(self) -> float:
        return self._live_xg_signal(self.away_stats, self.away_team_id)

    def _live_xg_signal(self, stats: dict[int, float], team_id: int) -> float:
        live_proxy = _xg_proxy(stats)
        # We can't compute the expected here without external context;
        # the predictor has access to Sportmonks predictions and will
        # subtract the expected level itself. This method exposes raw
        # live proxy.
        return live_proxy

    def sportmonks_prediction(self, type_id: int) -> dict[str, Any] | None:
        return self.sportmonks_predictions.get(type_id)

    # ── construction from a Sportmonks Fixture ──────────────────────────

    @classmethod
    def from_fixture(
        cls,
        fixture: Fixture,
        *,
        pressure_window_minutes: int = 10,
        snapshot_taken_at: datetime | None = None,
        raw_comments: list[dict] | None = None,
    ) -> LiveMatchState:
        """Build a state from an enriched Fixture.

        Required includes on the fixture:
        - participants, state, periods, scores, statistics, predictions
        Optional:
        - trends, pressure, events
        """
        if not fixture.participants or len(fixture.participants) < 2:
            raise ValueError("Fixture missing participants")
        home, away = _identify_home_away(fixture)

        # Score (latest current score per side)
        home_goals, away_goals = _extract_current_score(fixture, home.id, away.id)

        # Minute + period
        minute, period_id = _extract_minute_and_period(fixture)
        is_live = fixture.is_live()
        state_dev = fixture.state.developer_name if fixture.state else ""
        is_half_time = state_dev == "HT"
        is_finished = state_dev in ("FT", "AET", "FT_PEN", "FINISHED")

        # Aggregate statistics by type_id × team
        home_stats, away_stats = _aggregate_statistics(
            fixture.statistics or [], home.id, away.id
        )

        # Pressure window (last N minutes)
        home_pressure, away_pressure = _pressure_window(
            fixture.pressure or [], home.id, away.id, minute, pressure_window_minutes
        )

        # Sportmonks predictions indexed by type_id
        predictions_by_type = _index_predictions(fixture.predictions or [])

        # Events: extract goals + red cards + yellow cards + subs
        # The detailed goal stream preserves scorer player_id for retro
        # analysis (which the predictor doesn't read but the snapshot does).
        (
            goal_events, red_card_events, yellow_events, sub_events,
            goal_events_detailed,
        ) = _extract_events(
            fixture.events or [], home.id, away.id
        )

        # Trends — minute-by-minute stat values per (type_id, side).
        # Filter to the two participating teams to keep memory bounded.
        team_ids = {home.id, away.id}
        trends = [
            t for t in (fixture.trends or [])
            if t.participant_id in team_ids
        ]

        # Commentary-narrative events (optional — extracted from raw payload
        # when caller passes raw_comments). Lazy import to avoid coupling.
        commentary_events: list = []
        if raw_comments:
            from bip.evaluation.live.commentary import extract_events
            commentary_events = extract_events(raw_comments)

        return cls(
            fixture_id=fixture.id,
            home_team_id=home.id,
            away_team_id=away.id,
            home_team_name=home.name,
            away_team_name=away.name,
            home_goals=home_goals,
            away_goals=away_goals,
            minute=minute,
            period_id=period_id,
            is_live=is_live,
            is_half_time=is_half_time,
            is_finished=is_finished,
            home_stats=home_stats,
            away_stats=away_stats,
            home_pressure_recent=home_pressure,
            away_pressure_recent=away_pressure,
            sportmonks_predictions=predictions_by_type,
            trends=trends,
            goal_events=goal_events,
            goal_events_detailed=goal_events_detailed,
            red_card_events=red_card_events,
            yellow_card_events=yellow_events,
            substitution_events=sub_events,
            commentary_events=commentary_events,
            snapshot_taken_at=snapshot_taken_at,
            league_id=fixture.league_id,
            season_id=fixture.season_id,
        )


# ── helpers ─────────────────────────────────────────────────────────────────


def _trend_split(samples: list[float]) -> float:
    """Compare recent-half vs earlier-half of a sample window.

    Positive return = trend is rising. Zero when fewer than 4 samples.
    """
    if len(samples) < 4:
        return 0.0
    half = len(samples) // 2
    earlier = samples[:half]
    later = samples[half:]
    return (sum(later) / len(later)) - (sum(earlier) / len(earlier))


def _xg_proxy(stats: dict[int, float]) -> float:
    """Compute a weighted xG proxy from shot-quality stats.

    NOTE: this is a backstop when Sportmonks does not emit real xG. The
    weights are tuned for monotonicity (more shots → higher xG) but are
    NOT calibrated against ground truth. Use Sportmonks predictions
    directly when available.
    """
    return sum(
        XG_PROXY_WEIGHTS[tid] * stats.get(tid, 0.0)
        for tid in XG_PROXY_WEIGHTS
        if tid in stats
    )


def _home_away_from_scores(fixture: Fixture) -> tuple[int, int] | None:
    """Derive (home_participant_id, away_participant_id) from ``scores``.

    Sportmonks emits each score record with both ``participant_id`` (the
    team that scored) AND ``score.participant`` ('home' or 'away'). The
    cross-reference is the ground truth for which team is home — the
    score body says "this team is the home side" regardless of array
    order in ``participants``.

    Day-4 (2026-05-13) Palace-City raw payload confirmed:
        participant_id=9  score={participant: "home"}  → City is home
        participant_id=51 score={participant: "away"}  → Palace is away

    Even though ``participants[0]`` in the API response was Palace.
    """
    if not fixture.scores:
        return None
    home_id: int | None = None
    away_id: int | None = None
    for s in fixture.scores:
        if s.participant_id is None:
            continue
        body = s.score or {}
        loc = body.get("participant")
        if loc == "home" and home_id is None:
            home_id = s.participant_id
        elif loc == "away" and away_id is None:
            away_id = s.participant_id
        if home_id is not None and away_id is not None:
            break
    if home_id is None or away_id is None or home_id == away_id:
        return None
    return home_id, away_id


def _home_away_from_statistics(fixture: Fixture) -> tuple[int, int] | None:
    """Derive home/away from ``statistics[i].location``.

    Secondary fallback when ``scores`` is empty (pre-kickoff / NS state).
    Statistics are also keyed by ``participant_id`` × ``location``.
    """
    if not fixture.statistics:
        return None
    home_id: int | None = None
    away_id: int | None = None
    for s in fixture.statistics:
        if s.participant_id is None or s.location is None:
            continue
        if s.location == "home" and home_id is None:
            home_id = s.participant_id
        elif s.location == "away" and away_id is None:
            away_id = s.participant_id
        if home_id is not None and away_id is not None:
            break
    if home_id is None or away_id is None or home_id == away_id:
        return None
    return home_id, away_id


def _identify_home_away(fixture: Fixture):
    """Identify the home and away participants.

    Resolution order (most authoritative first):

    1. ``participants[i].meta.location`` — Sportmonks' designated field
       for home/away. Forward-compatible: included if/when the API
       request asks for the ``meta`` sub-resource.

    2. ``scores[i].score.participant`` cross-referenced with
       ``participant_id`` — ground truth for live and finished fixtures.
       Sportmonks tags each score with both team_id and home/away.

    3. ``statistics[i].location`` cross-referenced with
       ``participant_id`` — fallback for fixtures with stats but no
       scores yet.

    4. ``participants[0] = home`` — last-resort fallback. **Unreliable**:
       Day-4 (2026-05-13) observed 9/31 fixtures where ``participants[0]``
       was actually the away team (Crystal Palace vs Man City being the
       canonical case — we displayed "Palace 3-0 City" when in reality
       City won 3-0 at the Etihad).
    """
    if not fixture.participants or len(fixture.participants) < 2:
        raise ValueError("need 2 participants")
    participants_by_id = {p.id: p for p in fixture.participants}

    # Layer 1: meta.location (if API returned it).
    home = next((p for p in fixture.participants if p.is_home() is True), None)
    away = next((p for p in fixture.participants if p.is_home() is False), None)
    if home is not None and away is not None:
        return home, away
    any_meta = any(p.meta is not None for p in fixture.participants)
    if any_meta and (home is None) != (away is None):
        raise ValueError(
            f"fixture {fixture.id}: incomplete participant meta.location "
            f"(home={home is not None}, away={away is not None})"
        )

    # Layer 2: scores cross-reference.
    score_pair = _home_away_from_scores(fixture)
    if score_pair is not None:
        h_id, a_id = score_pair
        if h_id in participants_by_id and a_id in participants_by_id:
            return participants_by_id[h_id], participants_by_id[a_id]

    # Layer 3: statistics cross-reference.
    stats_pair = _home_away_from_statistics(fixture)
    if stats_pair is not None:
        h_id, a_id = stats_pair
        if h_id in participants_by_id and a_id in participants_by_id:
            return participants_by_id[h_id], participants_by_id[a_id]

    # Layer 4: last-resort array order. Unreliable; emits when scores
    # AND statistics are both absent (pre-kickoff state).
    return fixture.participants[0], fixture.participants[1]


def _extract_current_score(
    fixture: Fixture, home_id: int, away_id: int,
) -> tuple[int, int]:
    """Pull the CURRENT score per team from scores list.

    Sportmonks emits multiple score entries per fixture with different
    type_id values:
      - type_id=1     (1ST_HALF)       — first half only
      - type_id=2     (2ND_HALF)       — cumulative at end of 2H (== FT)
      - type_id=1525  (CURRENT)        — live current / FT score
      - type_id=48996 (2ND_HALF_ONLY)  — 2H goals only (NOT cumulative)

    Bug 2026-05-11: prior implementation walked all entries and
    overwrote home/away based on iteration order, but the API does
    NOT guarantee chronological order. 1ST_HALF could overwrite
    CURRENT and return half-time score even when the fixture was FT.
    Observed in 66% of finished fixtures (Randers FC vs Odense BK
    stored 0-1 vs real 2-2 final).

    Fix: filter by type_id, preferring CURRENT (1525) which is the
    live score during in-play AND the final score at FT.
    """
    if not fixture.scores:
        return 0, 0

    # Priority order:
    #   1. CURRENT (1525) — single source of truth in-play and at FT
    #   2. 2ND_HALF cumulative (2) — fallback if CURRENT absent at FT
    #   3. 1ST_HALF (1) — fallback for fixtures captured at half-time
    for preferred_type_id in (1525, 2, 1):
        h, a = 0, 0
        found_home = found_away = False
        for s in fixture.scores:
            if s.type_id != preferred_type_id:
                continue
            body = s.score or {}
            loc = body.get("participant")
            goals = body.get("goals")
            if not isinstance(goals, (int, float)):
                continue
            if loc == "home":
                h = int(goals)
                found_home = True
            elif loc == "away":
                a = int(goals)
                found_away = True
        if found_home and found_away:
            return h, a
    return 0, 0


def _extract_minute_and_period(fixture: Fixture) -> tuple[int, int]:
    """Return (minute, period_id) for the active period.

    Falls back to 0 when periods are absent, FT, or NS.
    """
    if not fixture.periods:
        return 0, 0
    # Active period: ticking, or the latest started one
    active = [p for p in fixture.periods if p.ticking]
    if active:
        p = active[0]
        return (p.minutes or 0, p.type_id)
    # Finished match: pick the period with the largest minutes
    finished = sorted(
        (p for p in fixture.periods if p.minutes is not None),
        key=lambda p: p.minutes or 0,
        reverse=True,
    )
    if finished:
        return finished[0].minutes or 0, finished[0].type_id
    return 0, 0


def _aggregate_statistics(
    stats: list[FixtureStatistic], home_id: int, away_id: int,
) -> tuple[dict[int, float], dict[int, float]]:
    home: dict[int, float] = {}
    away: dict[int, float] = {}
    for s in stats:
        v = s.value
        if v is None:
            continue
        if s.participant_id == home_id:
            home[s.type_id] = v
        elif s.participant_id == away_id:
            away[s.type_id] = v
    return home, away


def _pressure_window(
    pressure: list[PressureMinute],
    home_id: int,
    away_id: int,
    current_minute: int,
    window: int,
) -> tuple[list[float], list[float]]:
    """Return last ``window`` minutes of pressure values per side."""
    if not pressure:
        return [], []
    threshold = max(0, current_minute - window)
    home: list[float] = []
    away: list[float] = []
    for p in pressure:
        if p.minute < threshold:
            continue
        if p.minute > current_minute:
            continue
        if p.participant_id == home_id:
            home.append(p.pressure)
        elif p.participant_id == away_id:
            away.append(p.pressure)
    return home, away


def _index_predictions(preds: list[Prediction]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for p in preds:
        if p.predictions:
            out[p.type_id] = p.predictions
    return out


def _extract_events(
    events, home_id: int, away_id: int,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int | None]],
]:
    """Pull goals + red cards + yellow cards + substitutions from event list.

    Sportmonks v3 event type_ids (verified 2026-05-09 against /core/types):
    - 14 = GOAL
    - 15 = OWNGOAL (we record minute regardless of attribution side)
    - 16 = PENALTY (scored from the spot)
    - 17 = MISSED_PENALTY (NOT a goal)
    - 18 = SUBSTITUTION
    - 19 = YELLOWCARD
    - 20 = REDCARD (straight red)
    - 21 = YELLOWREDCARD (second yellow → red, treated as red card)

    Returns
    -------
    goal_events
        ``[(minute, scoring_team_id), ...]`` sorted by minute.
    red_card_events
        ``[(minute, team_affected_id), ...]`` sorted by minute. Combines
        straight-red and yellow→red into one stream.
    yellow_card_events
        ``[(minute, team_id, player_id_or_0), ...]`` sorted.
    substitution_events
        ``[(minute, team_id, related_player_id_or_None), ...]`` sorted.
    goal_events_detailed
        ``[(minute, team_id, player_id_or_0), ...]`` — same goals as
        ``goal_events`` but with scorer attribution preserved for
        retrospective analysis.
    """
    goal_events: list[tuple[int, int]] = []
    goal_events_detailed: list[tuple[int, int, int]] = []
    red_card_events: list[tuple[int, int]] = []
    yellow_card_events: list[tuple[int, int, int]] = []
    substitution_events: list[tuple[int, int, int]] = []
    GOAL_TYPE_IDS = {EVENT_TYPE_GOAL, EVENT_TYPE_OWNGOAL, EVENT_TYPE_PENALTY_SCORED}
    RED_CARD_TYPE_IDS = {EVENT_TYPE_REDCARD, EVENT_TYPE_YELLOWREDCARD}
    for e in events:
        minute = e.minute
        team = e.participant_id
        if minute is None or team is None:
            continue
        if team not in (home_id, away_id):
            continue
        if e.type_id in GOAL_TYPE_IDS:
            goal_events.append((minute, team))
            goal_events_detailed.append((minute, team, e.player_id or 0))
        elif e.type_id in RED_CARD_TYPE_IDS:
            red_card_events.append((minute, team))
        elif e.type_id == EVENT_TYPE_YELLOWCARD:
            yellow_card_events.append((minute, team, e.player_id or 0))
        elif e.type_id == EVENT_TYPE_SUBSTITUTION:
            # ``or 0`` defends against the sort below: when two subs happen
            # at the same minute on the same team (common in double-subs),
            # tuple comparison falls through to the third element. If one
            # is None and the other is int, Python raises TypeError. The
            # 0 sentinel preserves "no related player" semantics safely.
            substitution_events.append((minute, team, e.related_player_id or 0))
    goal_events.sort()
    goal_events_detailed.sort()
    red_card_events.sort()
    yellow_card_events.sort()
    substitution_events.sort()
    return (
        goal_events, red_card_events, yellow_card_events,
        substitution_events, goal_events_detailed,
    )
