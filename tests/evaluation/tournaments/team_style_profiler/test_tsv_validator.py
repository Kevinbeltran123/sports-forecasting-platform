"""Tests for tsv_validator — decision tree gating."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    CoachInfo,
    DistributionStat,
    GoalsPer15Min,
    TeamStyleVector,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_validator import (
    DEFAULT_MAX_STALENESS_DAYS,
    should_force_refresh,
    validate_tsv,
)


def _ds(mean: float = 1.0, n: int = 20) -> DistributionStat:
    return DistributionStat(mean=mean, ci_low=mean - 0.3, ci_high=mean + 0.3, n=n)


def _g15(n: int = 20) -> GoalsPer15Min:
    return GoalsPer15Min(
        bucket_0_14=_ds(0.2, n), bucket_15_29=_ds(0.2, n),
        bucket_30_44=_ds(0.2, n), bucket_45_59=_ds(0.2, n),
        bucket_60_74=_ds(0.3, n), bucket_75_90=_ds(0.3, n),
    )


def _tsv(
    team: str = "Argentina", flag: str = "green", n_matches: int = 20,
    coach_name: str = "Lionel Scaloni", coach_start: date = date(2018, 8, 23),
    last_updated: datetime = datetime(2026, 5, 20),
    confederation: str = "CONMEBOL",
) -> TeamStyleVector:
    return TeamStyleVector(
        team_name=team, api_football_team_id=26,
        confederation=confederation,  # type: ignore[arg-type]
        coach=CoachInfo(coach_name=coach_name, start_date=coach_start),
        flag=flag,  # type: ignore[arg-type]
        n_matches=n_matches,
        last_updated=last_updated,
        goals_for_per_match=_ds(2.0, n_matches),
        goals_for_per_15min=_g15(n_matches),
        shots_per_match=_ds(13.0, n_matches),
        shots_on_target_per_match=_ds(5.0, n_matches),
        shots_on_target_ratio=_ds(0.4, n_matches),
        corners_for_per_match=_ds(6.0, n_matches),
        possession_avg=_ds(55.0, n_matches),
        goals_against_per_match=_ds(1.0, n_matches),
        goals_against_per_15min=_g15(n_matches),
        clean_sheet_rate=_ds(0.35, n_matches),
        shots_against_per_match=_ds(0.0, 0),
        corners_against_per_match=_ds(4.0, n_matches),
        offsides_against_per_match=_ds(2.0, n_matches),
        fouls_per_match=_ds(12.0, n_matches),
        yellow_cards_per_match=_ds(2.0, n_matches),
        red_cards_rate=_ds(0.05, n_matches),
        btts_rate=_ds(0.55, n_matches),
        over_25_rate=_ds(0.55, n_matches),
        over_35_rate=_ds(0.3, n_matches),
        mean_total_goals=_ds(2.8, n_matches),
    )


class TestValidateTsv:
    def test_green_fresh_no_coach_change(self) -> None:
        # Argentina with Scaloni (matches registry) — should be use_own_tsv
        now = datetime(2026, 5, 24)
        v = validate_tsv(_tsv(), now=now)
        assert v.decision == "use_own_tsv"
        assert v.is_stale is False
        assert v.coach_changed_since_build is False
        assert "green" in v.reason

    def test_red_flag_uses_cohort(self) -> None:
        v = validate_tsv(
            _tsv(team="Cabo Verde", confederation="CAF", flag="red", n_matches=2,
                  coach_name="Bubista", coach_start=date(2020, 3, 1)),
            now=datetime(2026, 5, 24),
        )
        # Cabo Verde isn't in coach registry — so coach_changed is False
        # (current returns None, no coach mismatch detected)
        assert v.decision == "use_confederation_cohort"
        assert "red flag" in v.reason

    def test_yellow_flag_uses_cohort(self) -> None:
        v = validate_tsv(
            _tsv(team="Cabo Verde", confederation="CAF", flag="yellow", n_matches=7,
                  coach_name="Bubista", coach_start=date(2020, 3, 1)),
            now=datetime(2026, 5, 24),
        )
        assert v.decision == "use_confederation_cohort"
        assert "yellow" in v.reason

    def test_coach_change_invalidates(self) -> None:
        # Brazil with FAKE coach (Tite) — current registry says Ancelotti
        v = validate_tsv(
            _tsv(team="Brazil", confederation="CONMEBOL",
                  coach_name="Tite", coach_start=date(2016, 6, 1)),
            now=datetime(2026, 5, 24),
        )
        assert v.decision == "abstain"
        assert "coach changed" in v.reason.lower()
        assert v.coach_changed_since_build is True

    def test_stale_flag_set_but_decision_stays_use_own(self) -> None:
        # Argentina built 30 days ago (stale)
        stale_built = datetime(2026, 4, 20)
        v = validate_tsv(
            _tsv(last_updated=stale_built), now=datetime(2026, 5, 24)
        )
        assert v.decision == "use_own_tsv"
        assert v.is_stale is True
        assert v.staleness_days >= 30

    def test_unknown_team_has_no_coach_change(self) -> None:
        # Custom team not in registry -> get_current_coach returns None
        # -> coach_changed=False
        v = validate_tsv(
            _tsv(team="Atlantis", coach_name="Plato"),
            now=datetime(2026, 5, 24),
        )
        # Defaults to green=use_own_tsv (no coach change detected)
        assert v.decision == "use_own_tsv"
        assert v.coach_changed_since_build is False


class TestShouldForceRefresh:
    def test_within_window_old_tsv_triggers_refresh(self) -> None:
        # Match in 30h, TSV built 5 days ago -> should refresh
        now = datetime(2026, 6, 11, 12, 0)
        tsv = _tsv(last_updated=datetime(2026, 6, 6, 12, 0))
        match_date = date(2026, 6, 12)
        assert should_force_refresh(tsv, match_date, hours_before_match=48, now=now) is True

    def test_too_early_no_refresh(self) -> None:
        # Match in 10 days, TSV built yesterday -> not refresh window
        now = datetime(2026, 6, 1, 12, 0)
        tsv = _tsv(last_updated=datetime(2026, 5, 31, 12, 0))
        match_date = date(2026, 6, 11)
        assert should_force_refresh(tsv, match_date, hours_before_match=48, now=now) is False

    def test_fresh_tsv_in_window_no_refresh(self) -> None:
        # Match in 24h, TSV built 6h ago -> fresh enough
        now = datetime(2026, 6, 11, 12, 0)
        tsv = _tsv(last_updated=datetime(2026, 6, 11, 6, 0))
        match_date = date(2026, 6, 12)
        assert should_force_refresh(tsv, match_date, hours_before_match=48, now=now) is False

    def test_match_in_past_no_refresh(self) -> None:
        now = datetime(2026, 6, 15, 12, 0)
        tsv = _tsv(last_updated=datetime(2026, 5, 1, 12, 0))
        match_date = date(2026, 6, 10)  # past
        assert should_force_refresh(tsv, match_date, hours_before_match=48, now=now) is False
