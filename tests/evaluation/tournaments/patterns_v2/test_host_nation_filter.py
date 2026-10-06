"""Tests for host-nation filter (L5.6 finding)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    HOST_DOWNGRADE_FACTOR,
    HOST_NATIONS,
    host_nation_in_fixture,
    host_nation_verdict,
)


class TestHostNations:
    def test_wc2026_has_three_hosts(self) -> None:
        hosts = HOST_NATIONS["world_cup_2026"]
        assert "USA" in hosts
        assert "Mexico" in hosts
        assert "Canada" in hosts

    def test_qatar_is_wc2022_host(self) -> None:
        assert "Qatar" in HOST_NATIONS["wc_2022"]

    def test_unknown_tournament_returns_empty(self) -> None:
        assert HOST_NATIONS.get("unknown_2050", frozenset()) == frozenset()


class TestHostInFixture:
    @pytest.mark.parametrize(
        "home,away,slug,expected",
        [
            ("USA", "Argentina", "world_cup_2026", True),
            ("Argentina", "Mexico", "world_cup_2026", True),
            ("Brazil", "Argentina", "world_cup_2026", False),
            ("Qatar", "Ecuador", "wc_2022", True),
            ("Ecuador", "Qatar", "wc_2022", True),
            ("Russia", "Saudi Arabia", "wc_2018", True),
            ("Côte d'Ivoire", "Guinea-Bissau", "afcon_2023", True),
            ("Ivory Coast", "Guinea-Bissau", "afcon_2023", True),
            ("Germany", "Scotland", "euro_2024", True),
            ("Spain", "Croatia", "euro_2024", False),
            # No host tournament => never returns True.
            ("USA", "Mexico", "copa_2024", False),
            ("Germany", "France", "euro_2020", False),
        ],
    )
    def test_host_detection(
        self, home: str, away: str, slug: str, expected: bool
    ) -> None:
        assert host_nation_in_fixture(home, away, slug) is expected


class TestHostNationVerdict:
    def test_non_host_returns_neutral(self) -> None:
        v = host_nation_verdict("Brazil", "Argentina", "world_cup_2026")
        assert v.host_in_fixture is False
        assert v.downgrade_factor == 1.0
        assert "non-host" in v.rationale.lower()

    def test_host_returns_downgrade(self) -> None:
        v = host_nation_verdict("USA", "Argentina", "world_cup_2026")
        assert v.host_in_fixture is True
        assert v.downgrade_factor == HOST_DOWNGRADE_FACTOR
        # rationale must cite L5.6 evidence for auditability
        assert "L5.6" in v.rationale

    def test_downgrade_factor_is_one_point_two(self) -> None:
        # The factor isn't arbitrary — 20% edge widening derived from a 3.32x
        # over-rep with moderate sample. Locking this in protects against
        # accidental drift.
        assert HOST_DOWNGRADE_FACTOR == 1.2

    def test_host_side_recorded_for_audit(self) -> None:
        v_home = host_nation_verdict("Mexico", "France", "world_cup_2026")
        v_away = host_nation_verdict("France", "Canada", "world_cup_2026")
        assert "home" in v_home.rationale
        assert "away" in v_away.rationale
