"""BettableProfile — unified interface over TSV / CohortProfile / CrossCohortProfile.

The cross_team_predictor doesn't care whether the source is the team's
own TSV or a cohort fallback. It needs a fixed set of fields. This
module provides the conversion from any of the three source types into
a canonical BettableProfile.

It also carries metadata about the source (for audit/explainability)
and the optional sub-profile that may be merged in via weighted_combine
inside the predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bip.evaluation.tournaments.team_style_profiler.confederation_cohort import (
    CohortProfile,
    CrossCohortProfile,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    Confederation,
    DistributionStat,
    SubProfile,
    TeamStyleVector,
)


SourceType = Literal["own_tsv", "confederation_cohort", "cross_cohort", "sub_profile"]


@dataclass(frozen=True)
class BettableProfile:
    """Canonical input for the cross-team predictor.

    Carries the minimum needed for the supported markets:
      - BTTS / O/U goals / AH:  goals_for, goals_against, btts, over_2.5, over_3.5
      - O/U corners:            corners_for, corners_against
      - O/U cards:              yellow_cards, fouls (proxy)

    Plus identity + source provenance for audit.
    """

    team_name: str
    confederation: Confederation
    source: SourceType

    # Core attacking
    goals_for_per_match: DistributionStat
    # Core defending
    goals_against_per_match: DistributionStat
    # Direct empirical market rates
    btts_rate: DistributionStat
    over_25_rate: DistributionStat
    over_35_rate: DistributionStat
    # Other markets
    corners_for_per_match: DistributionStat
    corners_against_per_match: DistributionStat
    yellow_cards_per_match: DistributionStat
    fouls_per_match: DistributionStat
    # Context
    mean_total_goals: DistributionStat
    possession_avg: DistributionStat

    # Optional sub-profile against the OPPONENT's confederation
    # (only set when the source is own_tsv AND a matching sub-profile exists).
    sub_profile_vs_opponent: SubProfile | None = None


def from_tsv(
    tsv: TeamStyleVector,
    opponent_confederation: Confederation | None = None,
) -> BettableProfile:
    """Convert a TeamStyleVector to a BettableProfile.

    If ``opponent_confederation`` is provided AND the TSV has a matching
    sub-profile, the sub-profile is attached for the predictor to merge in
    via weighted_combine.
    """
    sub = None
    if opponent_confederation is not None:
        sub = tsv.sub_profiles.get(opponent_confederation)

    return BettableProfile(
        team_name=tsv.team_name,
        confederation=tsv.confederation,
        source="own_tsv",
        goals_for_per_match=tsv.goals_for_per_match,
        goals_against_per_match=tsv.goals_against_per_match,
        btts_rate=tsv.btts_rate,
        over_25_rate=tsv.over_25_rate,
        over_35_rate=tsv.over_35_rate,
        corners_for_per_match=tsv.corners_for_per_match,
        corners_against_per_match=tsv.corners_against_per_match,
        yellow_cards_per_match=tsv.yellow_cards_per_match,
        fouls_per_match=tsv.fouls_per_match,
        mean_total_goals=tsv.mean_total_goals,
        possession_avg=tsv.possession_avg,
        sub_profile_vs_opponent=sub,
    )


def from_confederation_cohort(cohort: CohortProfile) -> BettableProfile:
    """Convert a CohortProfile (red/yellow flag fallback) to BettableProfile."""
    return BettableProfile(
        team_name=f"<cohort:{cohort.confederation}>",
        confederation=cohort.confederation,
        source="confederation_cohort",
        goals_for_per_match=cohort.goals_for_per_match,
        goals_against_per_match=cohort.goals_against_per_match,
        btts_rate=cohort.btts_rate,
        over_25_rate=cohort.over_25_rate,
        over_35_rate=cohort.over_35_rate,
        corners_for_per_match=cohort.corners_for_per_match,
        corners_against_per_match=cohort.corners_against_per_match,
        yellow_cards_per_match=cohort.yellow_cards_per_match,
        fouls_per_match=cohort.fouls_per_match,
        mean_total_goals=cohort.mean_total_goals,
        possession_avg=cohort.possession_avg,
        sub_profile_vs_opponent=None,
    )


def from_cross_cohort(
    our_team_name: str,
    cross: CrossCohortProfile,
) -> BettableProfile:
    """Build a BettableProfile from a cross-confederation cohort.

    Used when a team has a green general profile but no sub-profile vs
    the opponent's confederation. The cross cohort holds the "average
    how our_conf plays vs opponent_conf" stats; we use it as a profile.

    NOTE: cross cohorts lack defensive/possession/fouls fields. Those
    are zero-n DistributionStats, signalling to the predictor that the
    market relying on them (e.g. cards) should NOT be emitted from this
    cross cohort. The predictor checks ``is_usable`` per market.
    """
    zero = DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    return BettableProfile(
        team_name=our_team_name,
        confederation=cross.our_confederation,
        source="cross_cohort",
        goals_for_per_match=cross.goals_for_per_match,
        goals_against_per_match=cross.goals_against_per_match,
        btts_rate=cross.btts_rate,
        over_25_rate=cross.over_25_rate,
        over_35_rate=zero,  # not aggregated in CrossCohortProfile
        corners_for_per_match=cross.corners_for_per_match,
        corners_against_per_match=zero,  # not aggregated
        yellow_cards_per_match=cross.yellow_cards_per_match,
        fouls_per_match=zero,
        mean_total_goals=zero,
        possession_avg=zero,
        sub_profile_vs_opponent=None,
    )
