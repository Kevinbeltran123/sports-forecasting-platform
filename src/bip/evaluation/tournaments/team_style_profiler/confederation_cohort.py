"""Confederation cohort fallback — used when a team is in red/yellow flag.

OPERATOR DECISIONS (locked 2026-05-24):

1. **Primary cohort** (red flag): use the SIMPLE AVERAGE of all GREEN TSVs
   in the same confederation. E.g. Cape Verde with no usable profile gets
   replaced by mean(Morocco, Senegal, Côte d'Ivoire, Egypt, South Africa)
   for predictive purposes.

2. **Sub-profile fallback** (green TSV but no sub-profile vs the opponent's
   confederation): build a CROSS-CONFEDERATION cohort by averaging the
   "vs opponent_conf" sub-profiles across all green TSVs in the team's
   confederation. E.g. Argentina has no sub-profile vs AFC -> use the
   average of {Brazil.vs_AFC, Uruguay.vs_AFC, Colombia.vs_AFC, ...} where
   each contributing TSV has its own AFC sub-profile.

Both cohorts respect the n>=3 threshold for SubProfile and n>=10 for the
underlying TSVs (only "green" contribute).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    Confederation,
    DistributionStat,
    SubProfile,
    TeamStyleVector,
)


MIN_COHORT_SIZE = 3
"""Minimum number of green TSVs required to form a cohort. Below this,
the cohort is unreliable -> caller must abstain from picks."""


@dataclass(frozen=True)
class CohortProfile:
    """A simplified TSV-like object built from averaging green TSVs.

    Only includes the fields needed by the cross-team predictor (Phase 4).
    Not a full TeamStyleVector — explicitly typed as a cohort to surface
    its usage in audit logs.
    """

    confederation: Confederation
    n_teams_in_cohort: int
    """How many green TSVs were averaged."""

    team_names_in_cohort: tuple[str, ...]
    """For audit — exact teams that contributed."""

    goals_for_per_match: DistributionStat
    goals_against_per_match: DistributionStat
    btts_rate: DistributionStat
    over_25_rate: DistributionStat
    over_35_rate: DistributionStat
    corners_for_per_match: DistributionStat
    corners_against_per_match: DistributionStat
    yellow_cards_per_match: DistributionStat
    fouls_per_match: DistributionStat
    mean_total_goals: DistributionStat
    possession_avg: DistributionStat
    last_built: datetime


@dataclass(frozen=True)
class CrossCohortProfile:
    """Cohort of 'how confederation A plays vs confederation B' built from
    green TSVs that HAVE a sub-profile against B.

    Used when an individual team has a green general profile but no
    sub-profile vs the opponent's confederation.
    """

    our_confederation: Confederation
    vs_confederation: Confederation
    n_teams_in_cohort: int
    team_names_in_cohort: tuple[str, ...]

    goals_for_per_match: DistributionStat
    goals_against_per_match: DistributionStat
    btts_rate: DistributionStat
    over_25_rate: DistributionStat
    corners_for_per_match: DistributionStat
    yellow_cards_per_match: DistributionStat
    last_built: datetime


# ─── Helpers ────────────────────────────────────────────────────────────


def _avg_distribution(stats: list[DistributionStat]) -> DistributionStat:
    """Average mean across distributions; widen CI proportionally.

    The output CI is the union of input CIs (conservative): ci_low is the
    minimum of all ci_low, ci_high the maximum. This explicitly inflates
    uncertainty when the cohort is heterogeneous.

    n is the SUM of source n's — represents the total underlying match
    count contributing to the cohort.
    """
    if not stats:
        return DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    means = [s.mean for s in stats]
    ci_lows = [s.ci_low for s in stats]
    ci_highs = [s.ci_high for s in stats]
    return DistributionStat(
        mean=float(np.mean(means)),
        ci_low=float(np.min(ci_lows)),
        ci_high=float(np.max(ci_highs)),
        n=sum(s.n for s in stats),
    )


def _green(tsvs: list[TeamStyleVector]) -> list[TeamStyleVector]:
    return [t for t in tsvs if t.flag == "green"]


# ─── Primary cohort ─────────────────────────────────────────────────────


def build_confederation_cohort(
    confederation: Confederation,
    all_tsvs: list[TeamStyleVector],
) -> CohortProfile | None:
    """Build the average cohort for ``confederation`` using all green TSVs.

    Returns None when fewer than MIN_COHORT_SIZE green TSVs exist for the
    confederation — caller MUST treat this as "no fallback available".

    >>> # (example would require constructing TSVs; covered in tests)
    """
    green = [
        t for t in _green(all_tsvs) if t.confederation == confederation
    ]
    if len(green) < MIN_COHORT_SIZE:
        return None

    return CohortProfile(
        confederation=confederation,
        n_teams_in_cohort=len(green),
        team_names_in_cohort=tuple(sorted(t.team_name for t in green)),
        goals_for_per_match=_avg_distribution([t.goals_for_per_match for t in green]),
        goals_against_per_match=_avg_distribution([t.goals_against_per_match for t in green]),
        btts_rate=_avg_distribution([t.btts_rate for t in green]),
        over_25_rate=_avg_distribution([t.over_25_rate for t in green]),
        over_35_rate=_avg_distribution([t.over_35_rate for t in green]),
        corners_for_per_match=_avg_distribution([t.corners_for_per_match for t in green]),
        corners_against_per_match=_avg_distribution([t.corners_against_per_match for t in green]),
        yellow_cards_per_match=_avg_distribution([t.yellow_cards_per_match for t in green]),
        fouls_per_match=_avg_distribution([t.fouls_per_match for t in green]),
        mean_total_goals=_avg_distribution([t.mean_total_goals for t in green]),
        possession_avg=_avg_distribution([t.possession_avg for t in green]),
        last_built=datetime.now(),
    )


# ─── Cross-confederation cohort (sub-profile fallback) ──────────────────


def build_cross_confederation_cohort(
    our_confederation: Confederation,
    vs_confederation: Confederation,
    all_tsvs: list[TeamStyleVector],
) -> CrossCohortProfile | None:
    """Build the 'how our_conf plays vs vs_conf' cohort.

    Looks at all green TSVs in ``our_confederation`` and pulls their
    sub-profile against ``vs_confederation``. Averages across those
    sub-profiles.

    Returns None when fewer than MIN_COHORT_SIZE green TSVs have a
    sub-profile vs ``vs_confederation``.
    """
    contributors: list[tuple[TeamStyleVector, SubProfile]] = []
    for t in _green(all_tsvs):
        if t.confederation != our_confederation:
            continue
        sub = t.sub_profiles.get(vs_confederation)
        if sub is not None:
            contributors.append((t, sub))

    if len(contributors) < MIN_COHORT_SIZE:
        return None

    subs = [c[1] for c in contributors]
    return CrossCohortProfile(
        our_confederation=our_confederation,
        vs_confederation=vs_confederation,
        n_teams_in_cohort=len(contributors),
        team_names_in_cohort=tuple(sorted(t.team_name for t, _ in contributors)),
        goals_for_per_match=_avg_distribution([s.goals_for_per_match for s in subs]),
        goals_against_per_match=_avg_distribution([s.goals_against_per_match for s in subs]),
        btts_rate=_avg_distribution([s.btts_rate for s in subs]),
        over_25_rate=_avg_distribution([s.over_25_rate for s in subs]),
        corners_for_per_match=_avg_distribution([s.corners_for_per_match for s in subs]),
        yellow_cards_per_match=_avg_distribution([s.yellow_cards_per_match for s in subs]),
        last_built=datetime.now(),
    )
