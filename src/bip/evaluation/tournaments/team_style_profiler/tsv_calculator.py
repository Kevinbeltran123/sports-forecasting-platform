"""TSV calculator — compute Team Style Vector from raw API-Football data.

Given a team and a list of (Fixture, TeamStatsBlock, list[Event]) tuples
for that team's current-coach matches since 2023-01-01, this module emits
a fully populated ``TeamStyleVector``.

All distributions are computed with 2000-resample bootstrap (seed=42 —
matches the convention from patterns_v2).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.api_football_parsers import (
    Event,
    Fixture,
    TeamStatsBlock,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    CoachInfo,
    DistributionStat,
    GoalsPer15Min,
    SubProfile,
    TeamStyleVector,
)


BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


# ─── Bootstrap helper ───────────────────────────────────────────────────


def _bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> DistributionStat:
    """Bootstrap mean + 95% CI for a 1-D array of values.

    Returns DistributionStat(mean, ci_low, ci_high, n).
    Handles empty input by returning zero-CI with n=0.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    rng = np.random.default_rng(seed)
    point = float(stat_fn(arr))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(arr[idx])
    ci_low = float(np.quantile(boots, 0.025))
    ci_high = float(np.quantile(boots, 0.975))
    return DistributionStat(mean=point, ci_low=ci_low, ci_high=ci_high, n=n)


# ─── Per-match feature extraction ───────────────────────────────────────


@dataclass(frozen=True)
class MatchFeatures:
    """Extracted features for ONE match from the team's perspective.

    Used as the per-match unit fed into bootstrap aggregation.
    """

    fixture_id: int
    opponent_team_id: int
    opponent_confederation: str
    goals_for: int
    goals_against: int
    goals_for_minutes: tuple[int, ...]
    goals_against_minutes: tuple[int, ...]
    shots: int | None
    shots_on_target: int | None
    corners_for: int | None
    corners_against: int | None
    possession: float | None
    fouls: int | None
    yellow_cards: int
    red_cards: int
    offsides_against: int | None
    """Offsides committed by the OPPONENT — proxy for our defensive line."""

    @property
    def btts(self) -> bool:
        return self.goals_for > 0 and self.goals_against > 0

    @property
    def over_25(self) -> bool:
        return (self.goals_for + self.goals_against) > 2

    @property
    def over_35(self) -> bool:
        return (self.goals_for + self.goals_against) > 3

    @property
    def clean_sheet(self) -> bool:
        return self.goals_against == 0

    @property
    def total_goals(self) -> int:
        return self.goals_for + self.goals_against

    @property
    def shots_on_target_ratio(self) -> float | None:
        if self.shots is None or self.shots == 0 or self.shots_on_target is None:
            return None
        return self.shots_on_target / self.shots


def extract_match_features(
    fixture: Fixture,
    team_id: int,
    team_stats: TeamStatsBlock | None,
    opponent_stats: TeamStatsBlock | None,
    events: list[Event],
    opponent_confederation_lookup: Callable[[int], str],
) -> MatchFeatures:
    """Turn raw API-Football data for one match into MatchFeatures.

    ``team_id`` is the SUBJECT team (the one we're profiling). Opponent
    is whichever of fixture.teams.home / fixture.teams.away is NOT
    ``team_id``.

    Statistics are read from team_stats / opponent_stats (each is a
    TeamStatsBlock for the respective team in the match). Either can be
    None when API-Football doesn't have stats — missing features become
    None and are excluded from downstream aggregation.

    Goal minutes come from events.
    """
    home_id = fixture.teams.home.id
    away_id = fixture.teams.away.id
    is_home = team_id == home_id
    if not is_home and team_id != away_id:
        raise ValueError(
            f"team_id={team_id} not in fixture {fixture.fixture_id} "
            f"({home_id} vs {away_id})"
        )

    opponent_id = away_id if is_home else home_id
    opponent_conf = opponent_confederation_lookup(opponent_id)

    if is_home:
        gf = fixture.goals.home or 0
        ga = fixture.goals.away or 0
    else:
        gf = fixture.goals.away or 0
        ga = fixture.goals.home or 0

    # Per-team goal minutes from events (only Goal events for OUR team).
    goals_for_min: list[int] = []
    goals_against_min: list[int] = []
    yellow = 0
    red = 0
    for ev in events:
        m = ev.time.total_minute
        if ev.is_goal:
            if ev.team.id == team_id:
                goals_for_min.append(m)
            else:
                goals_against_min.append(m)
        elif ev.is_yellow_card and ev.team.id == team_id:
            yellow += 1
        elif ev.is_red_card and ev.team.id == team_id:
            red += 1

    # Per-team stats from /fixtures/statistics
    def _stat(block: TeamStatsBlock | None, key: str) -> float | None:
        if block is None:
            return None
        v = block.get(key)
        return float(v) if v is not None else None

    return MatchFeatures(
        fixture_id=fixture.fixture_id,
        opponent_team_id=opponent_id,
        opponent_confederation=opponent_conf,
        goals_for=int(gf),
        goals_against=int(ga),
        goals_for_minutes=tuple(goals_for_min),
        goals_against_minutes=tuple(goals_against_min),
        shots=int(_stat(team_stats, "Total Shots") or 0) if _stat(team_stats, "Total Shots") is not None else None,
        shots_on_target=int(_stat(team_stats, "Shots on Goal") or 0) if _stat(team_stats, "Shots on Goal") is not None else None,
        corners_for=int(_stat(team_stats, "Corner Kicks") or 0) if _stat(team_stats, "Corner Kicks") is not None else None,
        corners_against=int(_stat(opponent_stats, "Corner Kicks") or 0) if _stat(opponent_stats, "Corner Kicks") is not None else None,
        possession=_stat(team_stats, "Ball Possession"),
        fouls=int(_stat(team_stats, "Fouls") or 0) if _stat(team_stats, "Fouls") is not None else None,
        yellow_cards=yellow,
        red_cards=red,
        offsides_against=int(_stat(opponent_stats, "Offsides") or 0) if _stat(opponent_stats, "Offsides") is not None else None,
    )


# ─── 15-min goal distribution ───────────────────────────────────────────


_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 200)]
"""Last bucket extends to 200 to absorb stoppage time."""


def _per15_from_minutes(
    matches: list[MatchFeatures], side: str
) -> GoalsPer15Min:
    """Compute GoalsPer15Min from minute-level events.

    side: 'for' (our goals) or 'against' (opponent goals).
    """
    if side not in ("for", "against"):
        raise ValueError("side must be 'for' or 'against'")
    # Per match: count goals in each bucket. Bootstrap MEAN per match.
    per_match_counts: list[list[int]] = [
        [0] * len(_BUCKETS) for _ in matches
    ]
    for i, m in enumerate(matches):
        mins = m.goals_for_minutes if side == "for" else m.goals_against_minutes
        for minute in mins:
            for b_idx, (lo, hi) in enumerate(_BUCKETS):
                if lo <= minute < hi:
                    per_match_counts[i][b_idx] += 1
                    break

    return GoalsPer15Min(
        bucket_0_14=_bootstrap_ci([row[0] for row in per_match_counts]),
        bucket_15_29=_bootstrap_ci([row[1] for row in per_match_counts]),
        bucket_30_44=_bootstrap_ci([row[2] for row in per_match_counts]),
        bucket_45_59=_bootstrap_ci([row[3] for row in per_match_counts]),
        bucket_60_74=_bootstrap_ci([row[4] for row in per_match_counts]),
        bucket_75_90=_bootstrap_ci([row[5] for row in per_match_counts]),
    )


# ─── TSV computation ────────────────────────────────────────────────────


def _non_null(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def compute_tsv(
    team_name: str,
    api_football_team_id: int,
    confederation: str,
    coach: CoachInfo,
    matches: list[MatchFeatures],
    last_updated: datetime | None = None,
    sub_profile_min_n: int = 3,
) -> TeamStyleVector:
    """Compute TeamStyleVector from a list of MatchFeatures.

    ``matches`` MUST already be filtered to:
      - the team's current-coach spell
      - era window (e.g., since 2023-01-01)

    Flag assignment:
      - red:    n < 5
      - yellow: 5 <= n < 10
      - green:  n >= 10

    Sub-profiles created for any confederation with n_matches_vs_conf >= sub_profile_min_n.
    """
    n_matches = len(matches)
    if n_matches < 5:
        flag = "red"
    elif n_matches < 10:
        flag = "yellow"
    else:
        flag = "green"

    if last_updated is None:
        last_updated = datetime.now()

    # Per-match scalar features
    goals_for = [m.goals_for for m in matches]
    goals_against = [m.goals_against for m in matches]
    total_goals = [m.total_goals for m in matches]
    btts = [int(m.btts) for m in matches]
    over_25 = [int(m.over_25) for m in matches]
    over_35 = [int(m.over_35) for m in matches]
    clean_sheet = [int(m.clean_sheet) for m in matches]

    shots = _non_null([m.shots for m in matches])
    shots_on = _non_null([m.shots_on_target for m in matches])
    shots_against = []  # we don't track opponent's shots cleanly; placeholder
    sot_ratio = _non_null([m.shots_on_target_ratio for m in matches])
    corners_for = _non_null([m.corners_for for m in matches])
    corners_against = _non_null([m.corners_against for m in matches])
    possession = _non_null([m.possession for m in matches])
    fouls = _non_null([m.fouls for m in matches])
    yellow = [m.yellow_cards for m in matches]
    red = [m.red_cards for m in matches]
    offsides_against = _non_null([m.offsides_against for m in matches])

    # Sub-profiles by opposing confederation
    sub_profiles: dict[str, SubProfile] = {}
    confs_seen: dict[str, list[MatchFeatures]] = {}
    for m in matches:
        confs_seen.setdefault(m.opponent_confederation, []).append(m)
    for conf, sub_matches in confs_seen.items():
        if conf in {"UNK", ""}:
            continue
        if len(sub_matches) < sub_profile_min_n:
            continue
        sub_profiles[conf] = SubProfile(
            vs_confederation=conf,  # type: ignore[arg-type]
            n_matches_vs_conf=len(sub_matches),
            goals_for_per_match=_bootstrap_ci([m.goals_for for m in sub_matches]),
            goals_against_per_match=_bootstrap_ci([m.goals_against for m in sub_matches]),
            btts_rate=_bootstrap_ci([int(m.btts) for m in sub_matches]),
            over_25_rate=_bootstrap_ci([int(m.over_25) for m in sub_matches]),
            corners_for_per_match=_bootstrap_ci(
                _non_null([m.corners_for for m in sub_matches]) or [0.0]
            ),
            yellow_cards_per_match=_bootstrap_ci(
                [m.yellow_cards for m in sub_matches]
            ),
        )

    return TeamStyleVector(
        team_name=team_name,
        api_football_team_id=api_football_team_id,
        confederation=confederation,  # type: ignore[arg-type]
        coach=coach,
        flag=flag,  # type: ignore[arg-type]
        n_matches=n_matches,
        last_updated=last_updated,
        goals_for_per_match=_bootstrap_ci(goals_for),
        goals_for_per_15min=_per15_from_minutes(matches, "for"),
        shots_per_match=_bootstrap_ci(shots) if shots else _bootstrap_ci([]),
        shots_on_target_per_match=(
            _bootstrap_ci(shots_on) if shots_on else _bootstrap_ci([])
        ),
        shots_on_target_ratio=(
            _bootstrap_ci(sot_ratio) if sot_ratio else _bootstrap_ci([])
        ),
        corners_for_per_match=(
            _bootstrap_ci(corners_for) if corners_for else _bootstrap_ci([])
        ),
        possession_avg=(
            _bootstrap_ci(possession) if possession else _bootstrap_ci([])
        ),
        goals_against_per_match=_bootstrap_ci(goals_against),
        goals_against_per_15min=_per15_from_minutes(matches, "against"),
        clean_sheet_rate=_bootstrap_ci(clean_sheet),
        shots_against_per_match=(
            _bootstrap_ci(shots_against) if shots_against else _bootstrap_ci([])
        ),
        corners_against_per_match=(
            _bootstrap_ci(corners_against) if corners_against else _bootstrap_ci([])
        ),
        offsides_against_per_match=(
            _bootstrap_ci(offsides_against) if offsides_against else _bootstrap_ci([])
        ),
        fouls_per_match=_bootstrap_ci(fouls) if fouls else _bootstrap_ci([]),
        yellow_cards_per_match=_bootstrap_ci(yellow),
        red_cards_rate=_bootstrap_ci(red),
        btts_rate=_bootstrap_ci(btts),
        over_25_rate=_bootstrap_ci(over_25),
        over_35_rate=_bootstrap_ci(over_35),
        mean_total_goals=_bootstrap_ci(total_goals),
        sub_profiles=sub_profiles,
    )
