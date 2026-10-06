"""Team Style Vector (TSV) schema — Pydantic v2.

Single source of truth for what a "team's style" means in this system.
Every dimension is calibrated as ``DistributionStat`` (mean + bootstrap
95% CI + n) so downstream consumers can reason about uncertainty.

Versioning: bump ``TSV_SCHEMA_VERSION`` on any field change so cached
profiles can be invalidated. Field-level changes that affect cross-team
predictions are MAJOR (1.0 -> 2.0); additive changes that don't affect
existing predictions are MINOR (1.0 -> 1.1).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

TSV_SCHEMA_VERSION = "1.0.0"

Confederation = Literal["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC", "UNK"]
ProfileFlag = Literal["green", "red", "yellow"]
"""``green`` = ≥10 matches with current coach, profile usable.
``yellow`` = 5-9 matches, exploratory only (low confidence).
``red`` = <5 matches, DO NOT bet on this team."""


class DistributionStat(BaseModel):
    """Mean + bootstrap 95% CI + sample size for a single feature.

    Use ``mean`` for point estimates, ``ci_low``/``ci_high`` for
    uncertainty propagation, and ``n`` to filter out under-powered cells.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    ci_low: float
    ci_high: float
    n: int = Field(ge=0, description="Sample size used to compute the stat.")

    @computed_field
    @property
    def ci_width(self) -> float:
        """CI width — wider = less confident."""
        return self.ci_high - self.ci_low

    @computed_field
    @property
    def is_usable(self) -> bool:
        """Usable for predictions when n >= 10."""
        return self.n >= 10


class GoalsPer15Min(BaseModel):
    """Goal distribution by 15-minute bucket across the regulation window.

    Buckets are inclusive-lower, exclusive-upper:
        0-14, 15-29, 30-44, 45-59, 60-74, 75-90 (stoppage counted in 75-90).
    """

    model_config = ConfigDict(frozen=True)

    bucket_0_14: DistributionStat
    bucket_15_29: DistributionStat
    bucket_30_44: DistributionStat
    bucket_45_59: DistributionStat
    bucket_60_74: DistributionStat
    bucket_75_90: DistributionStat


class CoachInfo(BaseModel):
    """Current head coach with start date of current spell.

    ``start_date`` is the cutoff for TSP's ``current-coach-only`` filter.
    Any fixture before this date is excluded from the team's profile.
    """

    model_config = ConfigDict(frozen=True)

    coach_name: str
    """Canonical name (e.g. 'Lionel Scaloni', 'Hugo Broos')."""

    start_date: date
    """First match in the current spell (inclusive)."""

    transfermarkt_id: int | None = None
    """Optional Transfermarkt coach ID for cross-reference."""


class SubProfile(BaseModel):
    """Per-confederation sub-profile.

    Only reported when ``n_matches_vs_conf >= 3``. Below that, falls back
    to the team's general profile.
    """

    model_config = ConfigDict(frozen=True)

    vs_confederation: Confederation
    n_matches_vs_conf: int

    goals_for_per_match: DistributionStat
    goals_against_per_match: DistributionStat
    btts_rate: DistributionStat
    over_25_rate: DistributionStat
    corners_for_per_match: DistributionStat
    yellow_cards_per_match: DistributionStat


class TeamStyleVector(BaseModel):
    """Full team style profile (~20 dimensions).

    Used by cross_team_predictor to derive market-specific probabilities
    for a fixture by combining home_tsv + away_tsv.

    All metrics are computed over matches:
      1. Of the team with its CURRENT head coach (per ``coach``).
      2. Since 2023-01-01 (post-WC22 cycle start).
      3. Excluding fixtures with missing statistics on the relevant
         feature (per-feature missing-data handling).
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    schema_version: str = TSV_SCHEMA_VERSION
    team_name: str
    api_football_team_id: int
    confederation: Confederation
    coach: CoachInfo

    # --- Profile metadata ---
    flag: ProfileFlag
    """green (n>=10), yellow (5-9), red (<5). Operator must respect."""

    n_matches: int = Field(ge=0)
    """Total matches with current coach since 2023-01-01."""

    last_updated: datetime
    """When this profile was last computed."""

    # --- Offensive dimensions ---
    goals_for_per_match: DistributionStat
    goals_for_per_15min: GoalsPer15Min
    shots_per_match: DistributionStat
    shots_on_target_per_match: DistributionStat
    shots_on_target_ratio: DistributionStat
    """SoT / shots — finishing/positioning quality proxy."""
    corners_for_per_match: DistributionStat
    possession_avg: DistributionStat
    """Average possession percentage [0, 100]."""
    xg_for_per_match: DistributionStat | None = None
    """xG generated — only available when StatsBomb data covers the match."""

    # --- Defensive dimensions ---
    goals_against_per_match: DistributionStat
    goals_against_per_15min: GoalsPer15Min
    clean_sheet_rate: DistributionStat
    shots_against_per_match: DistributionStat
    corners_against_per_match: DistributionStat
    offsides_against_per_match: DistributionStat
    """Offsides committed by opponent — proxy for high defensive line."""

    # --- "Character of match" dimensions ---
    fouls_per_match: DistributionStat
    yellow_cards_per_match: DistributionStat
    red_cards_rate: DistributionStat
    """Red cards per match — typically <0.1, high variance."""
    btts_rate: DistributionStat
    over_25_rate: DistributionStat
    over_35_rate: DistributionStat
    mean_total_goals: DistributionStat
    """Goals scored by both teams per match."""

    # --- Optional sub-profiles per opposing confederation ---
    sub_profiles: dict[str, SubProfile] = Field(default_factory=dict)
    """Keyed by Confederation literal. Only populated when
    ``n_matches_vs_conf >= 3``."""

    @computed_field
    @property
    def is_bettable(self) -> bool:
        """True when flag is green and core distributions are usable."""
        return (
            self.flag == "green"
            and self.goals_for_per_match.is_usable
            and self.goals_against_per_match.is_usable
        )

    @computed_field
    @property
    def has_sub_profile_for(self) -> list[str]:
        """Confederations for which a sub-profile exists."""
        return sorted(self.sub_profiles.keys())
