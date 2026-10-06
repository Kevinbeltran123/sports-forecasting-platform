"""TSV validator — gating decisions before a pick is allowed.

Combines:
- flag check (red/yellow/green from tsv_calculator)
- freshness check (was the TSV last_updated recently enough?)
- coach-change detection (did the DT change since the TSV was built?)
- bettability composite (all of the above must pass)

Used by the pipeline orchestrator to decide whether to:
  - use the team's own TSV (green + fresh + no DT change)
  - fall back to confederation cohort (red flag)
  - abstain entirely (cohort also fails)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from bip.evaluation.tournaments.team_style_profiler.coach_history import (
    get_current_coach,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    TeamStyleVector,
)


# Refresh budget per operator decision: 7 days during FIFA windows,
# forced refresh 24-48h before WC matches.
DEFAULT_MAX_STALENESS_DAYS = 7


GateDecision = Literal[
    "use_own_tsv",
    "use_confederation_cohort",
    "use_cross_cohort_subprofile",
    "abstain",
]


@dataclass(frozen=True)
class ValidationVerdict:
    team_name: str
    decision: GateDecision
    reason: str
    is_stale: bool
    staleness_days: int | None
    coach_changed_since_build: bool
    flag: str
    n_matches: int


def _staleness_days(tsv: TeamStyleVector, now: datetime | None = None) -> int:
    now = now or datetime.now()
    return (now - tsv.last_updated).days


def validate_tsv(
    tsv: TeamStyleVector,
    now: datetime | None = None,
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
) -> ValidationVerdict:
    """Decide whether ``tsv`` is usable, or what fallback to take.

    Decision tree:
      1. coach_changed since TSV build -> abstain (operator decision: invalidate)
      2. flag is red -> use confederation_cohort
      3. flag is yellow -> use confederation_cohort (low confidence)
      4. is_stale -> use_own_tsv but flagged stale (caller may force refresh)
      5. green + fresh + no coach change -> use_own_tsv
    """
    now = now or datetime.now()
    staleness = _staleness_days(tsv, now)
    is_stale = staleness > max_staleness_days

    # Check coach change
    current = get_current_coach(tsv.team_name)
    coach_changed = False
    if current is not None and current.coach_name != tsv.coach.coach_name:
        coach_changed = True
    elif current is not None and current.coach_start_date > tsv.coach.start_date:
        # Same name but spell changed (re-hire) — treat as coach change.
        coach_changed = True

    if coach_changed:
        return ValidationVerdict(
            team_name=tsv.team_name,
            decision="abstain",
            reason=(
                f"coach changed since TSV built. TSV coach: "
                f"{tsv.coach.coach_name} (start "
                f"{tsv.coach.start_date.isoformat()}); current: "
                f"{current.coach_name if current else 'unknown'} (start "
                f"{current.coach_start_date.isoformat() if current else '?'}). "
                f"TSV invalidated per operator decision; need >=3 matches "
                f"with new DT before re-profiling."
            ),
            is_stale=is_stale,
            staleness_days=staleness,
            coach_changed_since_build=True,
            flag=tsv.flag,
            n_matches=tsv.n_matches,
        )

    if tsv.flag == "red":
        return ValidationVerdict(
            team_name=tsv.team_name,
            decision="use_confederation_cohort",
            reason=(
                f"red flag (n_matches={tsv.n_matches} < 10 with current "
                f"coach). Fallback: average of green TSVs in "
                f"{tsv.confederation}."
            ),
            is_stale=is_stale,
            staleness_days=staleness,
            coach_changed_since_build=False,
            flag="red",
            n_matches=tsv.n_matches,
        )

    if tsv.flag == "yellow":
        return ValidationVerdict(
            team_name=tsv.team_name,
            decision="use_confederation_cohort",
            reason=(
                f"yellow flag (5 <= n_matches < 10, low confidence). "
                f"Fallback to confederation cohort. Operator may override "
                f"if they have additional context."
            ),
            is_stale=is_stale,
            staleness_days=staleness,
            coach_changed_since_build=False,
            flag="yellow",
            n_matches=tsv.n_matches,
        )

    # Green flag
    return ValidationVerdict(
        team_name=tsv.team_name,
        decision="use_own_tsv",
        reason=(
            f"green flag (n_matches={tsv.n_matches}). "
            f"Staleness: {staleness}d (max {max_staleness_days}d). "
            f"{'STALE — recommend refresh' if is_stale else 'fresh'}."
        ),
        is_stale=is_stale,
        staleness_days=staleness,
        coach_changed_since_build=False,
        flag="green",
        n_matches=tsv.n_matches,
    )


def should_force_refresh(
    tsv: TeamStyleVector,
    next_fixture_date: date,
    hours_before_match: int = 48,
    now: datetime | None = None,
) -> bool:
    """True iff TSV was built more than `hours_before_match` before the
    next fixture date — meaning we should refresh before betting.

    Per operator policy: refresh forced 24-48h before each WC match.
    """
    now = now or datetime.now()
    fixture_dt = datetime.combine(next_fixture_date, datetime.min.time())
    hours_to_match = (fixture_dt - now).total_seconds() / 3600
    if hours_to_match < 0:
        return False  # match already in the past
    if hours_to_match > hours_before_match:
        return False  # too early
    # Within the refresh window: force refresh unless TSV is also within window
    tsv_age_hours = (now - tsv.last_updated).total_seconds() / 3600
    return tsv_age_hours > hours_before_match
