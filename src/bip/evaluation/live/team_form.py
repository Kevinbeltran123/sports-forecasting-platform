"""Team form cache — recent-pattern signals derived from past fixtures.

A team's "1H goal rate over last 10 matches" is a slow-moving prior that
Sportmonks's pre-match prediction already implicitly incorporates. Where
team form adds value: detecting DIVERGENCE between the team's recent
pattern and what Sportmonks's ML model is pricing live — that's our
signal of model drift.

Architecture
------------

Storage: SQLite at ``data/cache/sportmonks/team_form.db``.
Key: ``(team_id, season_id)``. One row per team-in-season.
Refresh: 18-hour TTL (covers a typical match cycle). On miss/expired,
the watcher fetches the team's last N fixtures and computes metrics.

Metrics derived from ``list[Fixture]`` with goal_events:
    - first_half_goal_rate / second_half_goal_rate
    - early_goals_rate (≤ min 15) / late_goals_rate (≥ min 75)
    - btts_rate, clean_sheet_rate, failed_to_score_rate
    - goals_scored_avg, goals_conceded_avg
    - wins/draws/losses_last_5 (recency-weighted form)

Hooks into ``LiveMatchState`` via ``home_team_form`` / ``away_team_form``
optional fields. The predictor reads these in ``_btts_first_half``,
``_btts_second_half``, and ``_team_score_in_remaining_uncond`` to tilt
pre-match priors when the team's recent pattern diverges from the
league baseline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bip.sports.football.sportmonks.schemas import Fixture

logger = logging.getLogger(__name__)


DEFAULT_FORM_DB_PATH = Path("data/cache/sportmonks/team_form.db")
DEFAULT_FORM_TTL_HOURS = 18
DEFAULT_FORM_DAYS_BACK = 90
MIN_FIXTURES_FOR_FORM = 3   # below this, form is unreliable — return None

# League-baseline priors for divergence ratios. Top-5 European average,
# 2024-2025 season. Used as the denominator when computing how far a
# team's pattern diverges from "typical." Per-league calibration TODO.
LEAGUE_FIRST_HALF_GOAL_RATE_PRIOR = 0.55   # P(team scores ≥1 in 1H)
LEAGUE_SECOND_HALF_GOAL_RATE_PRIOR = 0.65
LEAGUE_LATE_GOAL_RATE_PRIOR = 0.42         # P(team scores ≥1 after 75')
LEAGUE_EARLY_GOAL_RATE_PRIOR = 0.20        # P(team scores ≥1 before 15')
LEAGUE_BTTS_RATE_PRIOR = 0.52
LEAGUE_CLEAN_SHEET_RATE_PRIOR = 0.30


_SCHEMA = """
CREATE TABLE IF NOT EXISTS team_form (
    team_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    last_updated TEXT NOT NULL,            -- ISO 8601 UTC
    n_matches INTEGER NOT NULL,
    -- Goal timing rates [0, 1]
    first_half_goal_rate REAL,
    second_half_goal_rate REAL,
    early_goals_rate REAL,
    late_goals_rate REAL,
    -- Match outcomes [0, 1] / averages
    btts_rate REAL,
    clean_sheet_rate REAL,
    failed_to_score_rate REAL,
    goals_scored_avg REAL,
    goals_conceded_avg REAL,
    -- Form (last 5 matches)
    wins_last_5 INTEGER,
    draws_last_5 INTEGER,
    losses_last_5 INTEGER,
    -- Audit
    payload_json TEXT,
    PRIMARY KEY (team_id, season_id)
);

CREATE INDEX IF NOT EXISTS idx_team_form_updated ON team_form (last_updated);
"""


@dataclass(frozen=True)
class TeamForm:
    """Aggregated recent-pattern signals for one team in one season."""

    team_id: int
    season_id: int
    n_matches: int
    last_updated: datetime

    first_half_goal_rate: float
    second_half_goal_rate: float
    early_goals_rate: float
    late_goals_rate: float

    btts_rate: float
    clean_sheet_rate: float
    failed_to_score_rate: float
    goals_scored_avg: float
    goals_conceded_avg: float

    wins_last_5: int
    draws_last_5: int
    losses_last_5: int

    @property
    def first_half_divergence(self) -> float:
        """Ratio vs league baseline. >1 = scores 1H more than typical;
        <1 = scores 1H less. Used as a multiplicative tilt in predictor."""
        return self.first_half_goal_rate / max(LEAGUE_FIRST_HALF_GOAL_RATE_PRIOR, 0.05)

    @property
    def late_goal_divergence(self) -> float:
        """Ratio vs league baseline for late-game (≥75') goals."""
        return self.late_goals_rate / max(LEAGUE_LATE_GOAL_RATE_PRIOR, 0.05)


# ── Form computation from past fixtures ─────────────────────────────────────


def compute_team_form(
    team_id: int,
    season_id: int,
    fixtures: Iterable[Fixture],
    *,
    now: datetime | None = None,
) -> TeamForm | None:
    """Derive form metrics from a list of past fixtures.

    Each fixture must have ``events`` populated and a final score (excluded
    from aggregation if the match never finished). Fixtures are filtered
    to those involving ``team_id``; goals are attributed via
    ``event.participant_id``.

    Returns None when fewer than MIN_FIXTURES_FOR_FORM completed matches
    are available — form is too noisy below that.
    """
    completed: list[tuple[Fixture, list[tuple[int, int]]]] = []
    for f in fixtures:
        if f.events is None or f.participants is None:
            continue
        # Only completed matches
        state_dev = f.state.developer_name if f.state else ""
        if state_dev not in ("FT", "AET", "FT_PEN", "FINISHED"):
            continue
        # Identify home/away on this fixture (api convention: [0]=home, [1]=away)
        if len(f.participants) < 2:
            continue
        # Extract goal events (type_id 14, 15, 16) for this fixture
        goal_events: list[tuple[int, int]] = []
        for e in f.events:
            if e.type_id not in (14, 15, 16):  # goal, owngoal, penalty scored
                continue
            if e.minute is None or e.participant_id is None:
                continue
            goal_events.append((e.minute, e.participant_id))
        completed.append((f, goal_events))

    if len(completed) < MIN_FIXTURES_FOR_FORM:
        return None

    n = len(completed)
    n_with_1h_goal = 0
    n_with_2h_goal = 0
    n_with_early_goal = 0
    n_with_late_goal = 0
    n_btts = 0
    n_clean_sheet = 0
    n_failed_to_score = 0
    total_scored = 0
    total_conceded = 0

    # Form: last 5 chronologically (sort fixtures by starting_at desc).
    # Sportmonks may emit starting_at as tz-naive — normalise so the sort
    # key never mixes naive/aware datetimes (Python raises TypeError on
    # comparison between mixed types).
    def _sort_key(pair):
        sa = pair[0].starting_at
        if sa is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        if sa.tzinfo is None:
            return sa.replace(tzinfo=timezone.utc)
        return sa

    completed_sorted = sorted(completed, key=_sort_key, reverse=True)
    last_5 = completed_sorted[:5]
    wins = draws = losses = 0

    for fixture, goal_events in completed:
        team_goals_1h = 0
        team_goals_2h = 0
        team_goals_total = 0
        opp_goals_total = 0
        team_scored_early = False
        team_scored_late = False
        any_team_goal = False
        any_opp_goal = False

        for minute, scoring_team in goal_events:
            if scoring_team == team_id:
                team_goals_total += 1
                any_team_goal = True
                if minute <= 45:
                    team_goals_1h += 1
                else:
                    team_goals_2h += 1
                if minute <= 15:
                    team_scored_early = True
                if minute >= 75:
                    team_scored_late = True
            else:
                opp_goals_total += 1
                any_opp_goal = True

        if team_goals_1h > 0:
            n_with_1h_goal += 1
        if team_goals_2h > 0:
            n_with_2h_goal += 1
        if team_scored_early:
            n_with_early_goal += 1
        if team_scored_late:
            n_with_late_goal += 1
        if any_team_goal and any_opp_goal:
            n_btts += 1
        if not any_opp_goal:
            n_clean_sheet += 1
        if not any_team_goal:
            n_failed_to_score += 1
        total_scored += team_goals_total
        total_conceded += opp_goals_total

    # Form W/D/L on last 5
    for fixture, goal_events in last_5:
        team_goals = sum(1 for _m, t in goal_events if t == team_id)
        opp_goals = sum(1 for _m, t in goal_events if t != team_id)
        if team_goals > opp_goals:
            wins += 1
        elif team_goals == opp_goals:
            draws += 1
        else:
            losses += 1

    return TeamForm(
        team_id=team_id,
        season_id=season_id,
        n_matches=n,
        last_updated=now or datetime.now(timezone.utc),
        first_half_goal_rate=n_with_1h_goal / n,
        second_half_goal_rate=n_with_2h_goal / n,
        early_goals_rate=n_with_early_goal / n,
        late_goals_rate=n_with_late_goal / n,
        btts_rate=n_btts / n,
        clean_sheet_rate=n_clean_sheet / n,
        failed_to_score_rate=n_failed_to_score / n,
        goals_scored_avg=total_scored / n,
        goals_conceded_avg=total_conceded / n,
        wins_last_5=wins,
        draws_last_5=draws,
        losses_last_5=losses,
    )


# ── SQLite-backed cache ─────────────────────────────────────────────────────


class TeamFormCache:
    """Get-or-fetch cache for team form. SQLite-persistent.

    Usage::

        cache = TeamFormCache()
        # In watcher loop:
        form = await cache.get_or_fetch(
            client, team_id=10, season_id=22, days_back=90,
        )
        # form is None if no API data; otherwise a TeamForm instance.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_FORM_DB_PATH,
        *,
        ttl_hours: int = DEFAULT_FORM_TTL_HOURS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(
        self, team_id: int, season_id: int,
        *, now: datetime | None = None,
    ) -> TeamForm | None:
        """Return cached form if fresh; None if expired or missing."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM team_form WHERE team_id = ? AND season_id = ?",
                (team_id, season_id),
            ).fetchone()
        if row is None:
            return None
        last_updated = datetime.fromisoformat(row["last_updated"])
        # Defensive tz normalisation: legacy cache rows may lack timezone
        # info even though current writes always include +00:00. Avoid
        # the tz-aware-vs-naive subtraction crash.
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        ref = now or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        if ref - last_updated > self.ttl:
            return None
        return TeamForm(
            team_id=row["team_id"],
            season_id=row["season_id"],
            n_matches=row["n_matches"],
            last_updated=last_updated,
            first_half_goal_rate=row["first_half_goal_rate"],
            second_half_goal_rate=row["second_half_goal_rate"],
            early_goals_rate=row["early_goals_rate"],
            late_goals_rate=row["late_goals_rate"],
            btts_rate=row["btts_rate"],
            clean_sheet_rate=row["clean_sheet_rate"],
            failed_to_score_rate=row["failed_to_score_rate"],
            goals_scored_avg=row["goals_scored_avg"],
            goals_conceded_avg=row["goals_conceded_avg"],
            wins_last_5=row["wins_last_5"],
            draws_last_5=row["draws_last_5"],
            losses_last_5=row["losses_last_5"],
        )

    def put(self, form: TeamForm) -> None:
        """Insert or replace cache entry."""
        payload = json.dumps(asdict(form), default=str, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO team_form (
                    team_id, season_id, last_updated, n_matches,
                    first_half_goal_rate, second_half_goal_rate,
                    early_goals_rate, late_goals_rate,
                    btts_rate, clean_sheet_rate, failed_to_score_rate,
                    goals_scored_avg, goals_conceded_avg,
                    wins_last_5, draws_last_5, losses_last_5,
                    payload_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    form.team_id, form.season_id,
                    form.last_updated.isoformat(),
                    form.n_matches,
                    form.first_half_goal_rate, form.second_half_goal_rate,
                    form.early_goals_rate, form.late_goals_rate,
                    form.btts_rate, form.clean_sheet_rate,
                    form.failed_to_score_rate,
                    form.goals_scored_avg, form.goals_conceded_avg,
                    form.wins_last_5, form.draws_last_5, form.losses_last_5,
                    payload,
                ),
            )

    async def get_or_fetch(
        self,
        client: Any,
        team_id: int,
        season_id: int,
        *,
        days_back: int = DEFAULT_FORM_DAYS_BACK,
        now: datetime | None = None,
    ) -> TeamForm | None:
        """Cache hit: return immediately. Miss/expired: fetch + compute + store.

        ``client`` must expose ``get_team_recent_fixtures(team_id, start_iso,
        end_iso, includes=...)``. We always include ``state`` (to filter
        completed matches) and ``events`` (for goal extraction).

        Returns None when the team has too few completed fixtures.
        """
        cached = self.get(team_id, season_id, now=now)
        if cached is not None:
            return cached
        ref = now or datetime.now(timezone.utc)
        start = (ref - timedelta(days=days_back)).date().isoformat()
        end = ref.date().isoformat()
        try:
            fixtures = await client.get_team_recent_fixtures(
                team_id, start_iso=start, end_iso=end,
                includes=["events", "state", "participants"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "team_form_fetch_failed team_id=%d err=%s", team_id, exc,
            )
            return None
        form = compute_team_form(team_id, season_id, fixtures, now=ref)
        if form is None:
            return None
        try:
            self.put(form)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "team_form_cache_put_failed team_id=%d err=%s", team_id, exc,
            )
        return form
