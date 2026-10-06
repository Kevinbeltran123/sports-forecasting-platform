"""Tests for TSV schema (Pydantic v2 models)."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    TSV_SCHEMA_VERSION,
    CoachInfo,
    DistributionStat,
    GoalsPer15Min,
    SubProfile,
    TeamStyleVector,
)


def _ds(mean: float = 1.5, ci_low: float = 1.2, ci_high: float = 1.8, n: int = 20) -> DistributionStat:
    return DistributionStat(mean=mean, ci_low=ci_low, ci_high=ci_high, n=n)


def _goals_15(n: int = 20) -> GoalsPer15Min:
    return GoalsPer15Min(
        bucket_0_14=_ds(0.1, 0.05, 0.15, n),
        bucket_15_29=_ds(0.2, 0.12, 0.28, n),
        bucket_30_44=_ds(0.25, 0.18, 0.32, n),
        bucket_45_59=_ds(0.2, 0.13, 0.27, n),
        bucket_60_74=_ds(0.3, 0.22, 0.38, n),
        bucket_75_90=_ds(0.4, 0.30, 0.50, n),
    )


def _coach() -> CoachInfo:
    return CoachInfo(coach_name="Lionel Scaloni", start_date=date(2018, 8, 23))


class TestDistributionStat:
    def test_ci_width(self) -> None:
        d = DistributionStat(mean=1.5, ci_low=1.2, ci_high=1.8, n=20)
        assert d.ci_width == pytest.approx(0.6)

    def test_is_usable_at_threshold(self) -> None:
        assert DistributionStat(mean=0, ci_low=0, ci_high=0, n=10).is_usable is True
        assert DistributionStat(mean=0, ci_low=0, ci_high=0, n=9).is_usable is False

    def test_frozen(self) -> None:
        d = _ds()
        with pytest.raises(Exception):
            d.mean = 99  # type: ignore[misc]

    def test_n_non_negative(self) -> None:
        with pytest.raises(Exception):
            DistributionStat(mean=0, ci_low=0, ci_high=0, n=-1)


class TestCoachInfo:
    def test_canonical_construction(self) -> None:
        c = CoachInfo(coach_name="Hugo Broos", start_date=date(2021, 5, 5))
        assert c.coach_name == "Hugo Broos"
        assert c.start_date.year == 2021

    def test_optional_transfermarkt_id(self) -> None:
        c = CoachInfo(coach_name="X", start_date=date(2024, 1, 1), transfermarkt_id=42)
        assert c.transfermarkt_id == 42


class TestTeamStyleVector:
    def _minimal_tsv(self, **overrides) -> TeamStyleVector:
        kwargs = dict(
            team_name="Argentina",
            api_football_team_id=26,
            confederation="CONMEBOL",
            coach=_coach(),
            flag="green",
            n_matches=20,
            last_updated=datetime(2026, 5, 24, 12, 0, 0),
            goals_for_per_match=_ds(2.1, 1.7, 2.5, 20),
            goals_for_per_15min=_goals_15(20),
            shots_per_match=_ds(14.0, 12.0, 16.0, 20),
            shots_on_target_per_match=_ds(5.5, 4.5, 6.5, 20),
            shots_on_target_ratio=_ds(0.39, 0.32, 0.46, 20),
            corners_for_per_match=_ds(5.8, 4.5, 7.1, 20),
            possession_avg=_ds(58.0, 53.0, 63.0, 20),
            goals_against_per_match=_ds(0.85, 0.55, 1.15, 20),
            goals_against_per_15min=_goals_15(20),
            clean_sheet_rate=_ds(0.40, 0.27, 0.53, 20),
            shots_against_per_match=_ds(8.0, 6.5, 9.5, 20),
            corners_against_per_match=_ds(3.5, 2.5, 4.5, 20),
            offsides_against_per_match=_ds(2.0, 1.3, 2.7, 20),
            fouls_per_match=_ds(13.5, 11.0, 16.0, 20),
            yellow_cards_per_match=_ds(2.2, 1.6, 2.8, 20),
            red_cards_rate=_ds(0.05, 0.0, 0.15, 20),
            btts_rate=_ds(0.50, 0.36, 0.64, 20),
            over_25_rate=_ds(0.55, 0.41, 0.69, 20),
            over_35_rate=_ds(0.30, 0.18, 0.42, 20),
            mean_total_goals=_ds(2.95, 2.5, 3.4, 20),
        )
        kwargs.update(overrides)
        return TeamStyleVector(**kwargs)

    def test_minimal_tsv_constructs(self) -> None:
        tsv = self._minimal_tsv()
        assert tsv.team_name == "Argentina"
        assert tsv.coach.coach_name == "Lionel Scaloni"
        assert tsv.schema_version == TSV_SCHEMA_VERSION
        assert tsv.is_bettable is True

    def test_red_flag_blocks_bettable(self) -> None:
        tsv = self._minimal_tsv(flag="red", n_matches=4)
        assert tsv.is_bettable is False

    def test_low_n_blocks_bettable(self) -> None:
        # Even with green flag (operator override), insufficient n on
        # core distribution should block bettability.
        tsv = self._minimal_tsv(
            goals_for_per_match=_ds(2.0, 1.5, 2.5, 5),
        )
        assert tsv.is_bettable is False

    def test_sub_profiles_optional(self) -> None:
        tsv = self._minimal_tsv()
        assert tsv.sub_profiles == {}
        assert tsv.has_sub_profile_for == []

    def test_sub_profiles_attach(self) -> None:
        sub = SubProfile(
            vs_confederation="UEFA",
            n_matches_vs_conf=4,
            goals_for_per_match=_ds(2.0, 1.5, 2.5, 4),
            goals_against_per_match=_ds(0.5, 0.0, 1.0, 4),
            btts_rate=_ds(0.5, 0.25, 0.75, 4),
            over_25_rate=_ds(0.5, 0.25, 0.75, 4),
            corners_for_per_match=_ds(6.0, 4.5, 7.5, 4),
            yellow_cards_per_match=_ds(2.0, 1.0, 3.0, 4),
        )
        tsv = self._minimal_tsv(sub_profiles={"UEFA": sub})
        assert "UEFA" in tsv.has_sub_profile_for
        assert tsv.sub_profiles["UEFA"].n_matches_vs_conf == 4

    def test_frozen(self) -> None:
        tsv = self._minimal_tsv()
        with pytest.raises(Exception):
            tsv.team_name = "Brazil"  # type: ignore[misc]
