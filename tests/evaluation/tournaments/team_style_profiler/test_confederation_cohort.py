"""Tests for confederation_cohort + cross_confederation_cohort."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.confederation_cohort import (
    MIN_COHORT_SIZE,
    _avg_distribution,
    build_confederation_cohort,
    build_cross_confederation_cohort,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    CoachInfo,
    DistributionStat,
    GoalsPer15Min,
    SubProfile,
    TeamStyleVector,
)


def _ds(mean: float, n: int = 20, half_width: float = 0.3) -> DistributionStat:
    return DistributionStat(
        mean=mean, ci_low=mean - half_width, ci_high=mean + half_width, n=n
    )


def _g15(value: float = 0.3, n: int = 20) -> GoalsPer15Min:
    return GoalsPer15Min(
        bucket_0_14=_ds(value, n), bucket_15_29=_ds(value, n),
        bucket_30_44=_ds(value, n), bucket_45_59=_ds(value, n),
        bucket_60_74=_ds(value, n), bucket_75_90=_ds(value, n),
    )


def _make_tsv(
    name: str, conf: str, flag: str = "green", n_matches: int = 20,
    goals_for: float = 1.8, btts: float = 0.55,
    coach_name: str = "C", coach_start: date = date(2022, 1, 1),
    sub_profile_vs: dict[str, SubProfile] | None = None,
) -> TeamStyleVector:
    return TeamStyleVector(
        team_name=name, api_football_team_id=hash(name) % 9999,
        confederation=conf,  # type: ignore[arg-type]
        coach=CoachInfo(coach_name=coach_name, start_date=coach_start),
        flag=flag,  # type: ignore[arg-type]
        n_matches=n_matches,
        last_updated=datetime(2026, 5, 20),
        goals_for_per_match=_ds(goals_for, n_matches),
        goals_for_per_15min=_g15(goals_for / 6, n_matches),
        shots_per_match=_ds(13.0, n_matches),
        shots_on_target_per_match=_ds(5.0, n_matches),
        shots_on_target_ratio=_ds(0.4, n_matches),
        corners_for_per_match=_ds(6.0, n_matches),
        possession_avg=_ds(55.0, n_matches),
        goals_against_per_match=_ds(1.0, n_matches),
        goals_against_per_15min=_g15(0.17, n_matches),
        clean_sheet_rate=_ds(0.35, n_matches),
        shots_against_per_match=_ds(0.0, 0),
        corners_against_per_match=_ds(4.0, n_matches),
        offsides_against_per_match=_ds(2.0, n_matches),
        fouls_per_match=_ds(12.0, n_matches),
        yellow_cards_per_match=_ds(2.0, n_matches),
        red_cards_rate=_ds(0.05, n_matches),
        btts_rate=_ds(btts, n_matches),
        over_25_rate=_ds(0.55, n_matches),
        over_35_rate=_ds(0.3, n_matches),
        mean_total_goals=_ds(2.8, n_matches),
        sub_profiles=sub_profile_vs or {},
    )


def _sub(conf: str, n: int = 5, goals_for: float = 1.5) -> SubProfile:
    return SubProfile(
        vs_confederation=conf,  # type: ignore[arg-type]
        n_matches_vs_conf=n,
        goals_for_per_match=_ds(goals_for, n),
        goals_against_per_match=_ds(1.0, n),
        btts_rate=_ds(0.5, n),
        over_25_rate=_ds(0.55, n),
        corners_for_per_match=_ds(5.5, n),
        yellow_cards_per_match=_ds(2.0, n),
    )


class TestAvgDistribution:
    def test_average_means(self) -> None:
        d = _avg_distribution([_ds(1.0), _ds(2.0), _ds(3.0)])
        assert d.mean == pytest.approx(2.0)

    def test_widen_ci(self) -> None:
        # ci_low = min, ci_high = max — conservative
        a = DistributionStat(mean=1.0, ci_low=0.5, ci_high=1.5, n=10)
        b = DistributionStat(mean=2.0, ci_low=1.7, ci_high=2.3, n=10)
        d = _avg_distribution([a, b])
        assert d.ci_low == 0.5
        assert d.ci_high == 2.3

    def test_n_is_sum(self) -> None:
        d = _avg_distribution([_ds(1.0, n=10), _ds(2.0, n=15), _ds(3.0, n=20)])
        assert d.n == 45

    def test_empty(self) -> None:
        d = _avg_distribution([])
        assert d.n == 0
        assert d.mean == 0


class TestBuildConfederationCohort:
    def test_basic_cohort(self) -> None:
        tsvs = [
            _make_tsv("Morocco", "CAF", goals_for=1.8),
            _make_tsv("Senegal", "CAF", goals_for=2.0),
            _make_tsv("Egypt", "CAF", goals_for=1.5),
            _make_tsv("South Africa", "CAF", goals_for=1.3),
            _make_tsv("France", "UEFA", goals_for=2.5),  # not CAF
        ]
        cohort = build_confederation_cohort("CAF", tsvs)
        assert cohort is not None
        assert cohort.n_teams_in_cohort == 4
        assert cohort.team_names_in_cohort == ("Egypt", "Morocco", "Senegal", "South Africa")
        # Mean of (1.8, 2.0, 1.5, 1.3) = 1.65
        assert cohort.goals_for_per_match.mean == pytest.approx(1.65)

    def test_excludes_non_green(self) -> None:
        tsvs = [
            _make_tsv("A", "CAF", flag="green", goals_for=2.0),
            _make_tsv("B", "CAF", flag="green", goals_for=2.0),
            _make_tsv("C", "CAF", flag="green", goals_for=2.0),
            _make_tsv("D", "CAF", flag="yellow", goals_for=999.0),  # excluded
            _make_tsv("E", "CAF", flag="red", goals_for=999.0),  # excluded
        ]
        cohort = build_confederation_cohort("CAF", tsvs)
        assert cohort is not None
        assert cohort.n_teams_in_cohort == 3
        # 999 values not in mean
        assert cohort.goals_for_per_match.mean == pytest.approx(2.0)

    def test_below_min_cohort_returns_none(self) -> None:
        tsvs = [
            _make_tsv("A", "OFC", goals_for=1.0),
            _make_tsv("B", "OFC", goals_for=1.0),
        ]
        cohort = build_confederation_cohort("OFC", tsvs)
        assert cohort is None  # only 2 < MIN_COHORT_SIZE=3

    def test_exactly_min_cohort_size(self) -> None:
        tsvs = [_make_tsv(f"T{i}", "CAF", goals_for=1.5) for i in range(MIN_COHORT_SIZE)]
        cohort = build_confederation_cohort("CAF", tsvs)
        assert cohort is not None
        assert cohort.n_teams_in_cohort == MIN_COHORT_SIZE


class TestBuildCrossConfederationCohort:
    def test_basic_cross_cohort(self) -> None:
        # 3 CONMEBOL teams, each with vs-AFC sub-profile
        tsvs = [
            _make_tsv(
                "Brazil", "CONMEBOL",
                sub_profile_vs={"AFC": _sub("AFC", n=4, goals_for=2.5)},
            ),
            _make_tsv(
                "Argentina", "CONMEBOL",
                sub_profile_vs={"AFC": _sub("AFC", n=3, goals_for=2.0)},
            ),
            _make_tsv(
                "Uruguay", "CONMEBOL",
                sub_profile_vs={"AFC": _sub("AFC", n=3, goals_for=1.5)},
            ),
            # No AFC sub-profile, should be excluded
            _make_tsv("Colombia", "CONMEBOL"),
        ]
        cohort = build_cross_confederation_cohort("CONMEBOL", "AFC", tsvs)
        assert cohort is not None
        assert cohort.n_teams_in_cohort == 3
        assert cohort.our_confederation == "CONMEBOL"
        assert cohort.vs_confederation == "AFC"
        # mean(2.5, 2.0, 1.5) = 2.0
        assert cohort.goals_for_per_match.mean == pytest.approx(2.0)

    def test_below_min_returns_none(self) -> None:
        tsvs = [
            _make_tsv(
                "Brazil", "CONMEBOL",
                sub_profile_vs={"AFC": _sub("AFC")},
            ),
            _make_tsv(
                "Argentina", "CONMEBOL",
                sub_profile_vs={"AFC": _sub("AFC")},
            ),
        ]
        cohort = build_cross_confederation_cohort("CONMEBOL", "AFC", tsvs)
        assert cohort is None

    def test_only_green_tsvs_contribute(self) -> None:
        tsvs = [
            _make_tsv(
                "A", "CONMEBOL", flag="green",
                sub_profile_vs={"AFC": _sub("AFC", n=4, goals_for=2.0)},
            ),
            _make_tsv(
                "B", "CONMEBOL", flag="green",
                sub_profile_vs={"AFC": _sub("AFC", n=4, goals_for=2.0)},
            ),
            _make_tsv(
                "C", "CONMEBOL", flag="green",
                sub_profile_vs={"AFC": _sub("AFC", n=4, goals_for=2.0)},
            ),
            _make_tsv(
                "Yellow", "CONMEBOL", flag="yellow",
                sub_profile_vs={"AFC": _sub("AFC", n=4, goals_for=999.0)},
            ),
        ]
        cohort = build_cross_confederation_cohort("CONMEBOL", "AFC", tsvs)
        assert cohort is not None
        assert "Yellow" not in cohort.team_names_in_cohort
        assert cohort.goals_for_per_match.mean == pytest.approx(2.0)
