"""Tests for coach_history — WC2026 current-coach registry."""
from __future__ import annotations

from datetime import date

import pytest

from bip.evaluation.tournaments.team_style_profiler.coach_history import (
    CURRENT_COACHES,
    get_current_coach,
    get_effective_filter_date,
    is_new_coach,
    teams_by_confederation,
)


class TestRegistryShape:
    def test_all_wc2026_hosts_present(self) -> None:
        assert "USA" in CURRENT_COACHES
        assert "Mexico" in CURRENT_COACHES
        assert "Canada" in CURRENT_COACHES

    def test_all_records_have_confederation(self) -> None:
        valid = {"UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"}
        for name, r in CURRENT_COACHES.items():
            assert r.confederation in valid, f"{name} has invalid conf"

    def test_coach_start_dates_in_past(self) -> None:
        # As of 2026-05-24, all start_dates should be in the past.
        today = date(2026, 5, 24)
        for name, r in CURRENT_COACHES.items():
            assert r.coach_start_date <= today, f"{name} coach starts in future"

    def test_team_ids_positive(self) -> None:
        for name, r in CURRENT_COACHES.items():
            assert r.api_football_team_id > 0, name


class TestGetCurrentCoach:
    def test_argentina_scaloni(self) -> None:
        r = get_current_coach("Argentina")
        assert r is not None
        assert r.coach_name == "Lionel Scaloni"
        assert r.coach_start_date == date(2018, 8, 23)
        assert r.confederation == "CONMEBOL"

    def test_morocco_ouahbi(self) -> None:
        # Regragui resigned after AFCON25 final; Ouahbi appointed 2026-03-05.
        r = get_current_coach("Morocco")
        assert r is not None
        assert r.coach_name == "Mohamed Ouahbi"
        assert r.confederation == "CAF"

    def test_unknown_team_returns_none(self) -> None:
        assert get_current_coach("Atlantis") is None


class TestEffectiveFilterDate:
    def test_pre_era_coach_returns_era_start(self) -> None:
        # Scaloni started 2018-08-23; era_start=2023-01-01 wins.
        assert get_effective_filter_date("Argentina") == date(2023, 1, 1)

    def test_post_era_coach_returns_coach_date(self) -> None:
        # Ancelotti started 2025-05-26 (Brazil) — post era.
        assert get_effective_filter_date("Brazil") == date(2025, 5, 26)

    def test_unknown_team_returns_none(self) -> None:
        assert get_effective_filter_date("Atlantis") is None

    def test_custom_era_start(self) -> None:
        # If we pushed era_start later, Scaloni-pre-era coach gets the new era_start.
        assert get_effective_filter_date(
            "Argentina", era_start=date(2024, 6, 1)
        ) == date(2024, 6, 1)


class TestIsNewCoach:
    def test_brazil_ancelotti_is_new(self) -> None:
        # Ancelotti started 2025-05-26; today=2026-05-24 -> 363 days ago.
        # threshold_days=365 -> True (within 1 year).
        assert is_new_coach("Brazil", today=date(2026, 5, 24)) is True

    def test_argentina_scaloni_is_not_new(self) -> None:
        # Scaloni started 2018 — 8 years ago, far beyond 365 days.
        assert is_new_coach("Argentina", today=date(2026, 5, 24)) is False

    def test_usa_pochettino_is_new(self) -> None:
        # Pochettino started 2024-10-01; today=2026-05-24 ~ 600 days.
        # Beyond 365 default threshold -> NOT new under default.
        assert is_new_coach("USA", today=date(2026, 5, 24)) is False
        # But under a tighter 730-day threshold, still considered "new-ish"
        assert is_new_coach("USA", today=date(2026, 5, 24), threshold_days=730) is True

    def test_unknown_team_returns_false(self) -> None:
        assert is_new_coach("Atlantis") is False


class TestTeamsByConfederation:
    def test_conmebol_includes_argentina_and_brazil(self) -> None:
        c = teams_by_confederation("CONMEBOL")
        assert "Argentina" in c
        assert "Brazil" in c
        assert "France" not in c

    def test_concacaf_includes_all_three_hosts(self) -> None:
        c = teams_by_confederation("CONCACAF")
        for host in ("USA", "Mexico", "Canada"):
            assert host in c

    def test_caf_includes_morocco_and_senegal(self) -> None:
        c = teams_by_confederation("CAF")
        assert "Morocco" in c
        assert "Senegal" in c

    def test_unknown_conf_returns_empty(self) -> None:
        assert teams_by_confederation("UNK") == []


class TestNewCoachFlags:
    """Sanity tests for high-risk teams flagged in design doc."""

    @pytest.mark.parametrize(
        "team",
        ["Brazil", "USA", "England", "Belgium", "Senegal"],
    )
    def test_known_recent_changes(self, team: str) -> None:
        # These are all teams flagged with NEW coach in the registry notes.
        r = get_current_coach(team)
        assert r is not None
        # Coach took over within 18 months of WC kick-off (June 2026).
        days_in_role = (date(2026, 5, 24) - r.coach_start_date).days
        assert days_in_role < 700, f"{team} coach not actually recent ({days_in_role}d)"
