"""Tests for regime-shift warning (L3.1 + L4.* finding)."""
from __future__ import annotations

from datetime import date

from bip.evaluation.tournaments.patterns_v2 import regime_warning_for_fixture


class TestRegimeWarning:
    def test_modern_ht_00_activates(self) -> None:
        w = regime_warning_for_fixture(date(2026, 6, 12), (0, 0))
        assert w.warning_active is True
        assert w.is_modern_era is True
        assert w.is_ht_zero_zero is True
        assert "62%" in w.rationale  # cites discovery rate
        assert "41%" in w.rationale  # cites hold-out rate

    def test_modern_non_zero_ht_does_not_activate(self) -> None:
        w = regime_warning_for_fixture(date(2026, 6, 12), (1, 0))
        assert w.warning_active is False
        assert w.is_modern_era is True
        assert w.is_ht_zero_zero is False

    def test_classic_ht_00_does_not_activate(self) -> None:
        w = regime_warning_for_fixture(date(2018, 6, 14), (0, 0))
        assert w.warning_active is False
        assert w.is_modern_era is False
        assert "pre-2022" in w.rationale.lower() or "62%" in w.rationale

    def test_pre_kick_modern_surfaces_potential(self) -> None:
        w = regime_warning_for_fixture(date(2026, 6, 12), None)
        assert w.warning_active is False
        assert "will activate" in w.rationale.lower()

    def test_era_boundary_is_2022_01_01(self) -> None:
        # WC22 was Nov-Dec but the regime shift is the discovery/hold-out split
        # boundary, which we encode as Jan 1 2022 to include the modern fold
        # tournaments fully.
        w_just_before = regime_warning_for_fixture(date(2021, 12, 31), (0, 0))
        w_just_after = regime_warning_for_fixture(date(2022, 1, 1), (0, 0))
        assert w_just_before.is_modern_era is False
        assert w_just_after.is_modern_era is True
