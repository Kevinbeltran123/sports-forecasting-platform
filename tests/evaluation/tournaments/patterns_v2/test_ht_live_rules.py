"""Tests for HT-state live rules (Lens 3 favorite-stratified matrix)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import HT_LIVE_TABLE, ht_live_lookup
from bip.evaluation.tournaments.patterns_v2.ht_live_rules import derive_ht_state


class TestDeriveHTState:
    @pytest.mark.parametrize(
        "h,a,fav,expected",
        [
            (0, 0, "home", "tied"),
            (0, 0, "away", "tied"),
            (1, 1, "home", "tied"),
            (2, 2, "away", "tied"),
            (1, 0, "home", "fav_up"),
            (1, 0, "away", "fav_down"),
            (0, 1, "home", "fav_down"),
            (0, 1, "away", "fav_up"),
            (3, 1, "home", "fav_up"),
            (1, 3, "home", "fav_down"),
        ],
    )
    def test_derivation(self, h: int, a: int, fav: str, expected: str) -> None:
        assert derive_ht_state(h, a, fav) == expected

    def test_no_favorite_returns_none(self) -> None:
        assert derive_ht_state(1, 0, "none") is None
        assert derive_ht_state(0, 0, "neither") is None


class TestHTLiveTable:
    def test_three_known_states(self) -> None:
        assert set(HT_LIVE_TABLE.keys()) == {"tied", "fav_up", "fav_down"}

    def test_p_fav_win_monotonic_by_ht_state(self) -> None:
        # fav_down < tied < fav_up — sanity check on table values.
        assert HT_LIVE_TABLE["fav_down"].p_fav_win < HT_LIVE_TABLE["tied"].p_fav_win
        assert HT_LIVE_TABLE["tied"].p_fav_win < HT_LIVE_TABLE["fav_up"].p_fav_win

    def test_sample_sizes(self) -> None:
        # n derived from session-1 EDA. Locking protects against silent drift.
        assert HT_LIVE_TABLE["tied"].n == 124
        assert HT_LIVE_TABLE["fav_up"].n == 94
        assert HT_LIVE_TABLE["fav_down"].n == 37

    def test_ci_bounds_well_formed(self) -> None:
        for state, cell in HT_LIVE_TABLE.items():
            assert (
                cell.p_fav_win_ci_low <= cell.p_fav_win <= cell.p_fav_win_ci_high
            ), state
            assert (
                cell.p_draw_ci_low <= cell.p_draw <= cell.p_draw_ci_high
            ), state


class TestHTLiveLookup:
    def test_fav_down_emits_fade_alert(self) -> None:
        v = ht_live_lookup("fav_down")
        assert "FADE_LIVE_FAV_WIN" in v.high_value_alerts
        assert "BACK_LIVE_UNDERDOG_WIN" in v.high_value_alerts
        # CI upper bound is the operationally relevant ceiling for fades.
        assert v.cell.p_fav_win_ci_high <= 0.30

    def test_fav_up_emits_consolidate_alert(self) -> None:
        v = ht_live_lookup("fav_up")
        assert "BACK_LIVE_FAV_CONSOLIDATES" in v.high_value_alerts
        assert v.cell.p_fav_win >= 0.80

    def test_tied_emits_draw_alert(self) -> None:
        v = ht_live_lookup("tied")
        assert "BACK_LIVE_DRAW_IF_ODDS_GE_2_78" in v.high_value_alerts
        # 1/0.36 = 2.78 — breakeven odds for the empirical 36% draw rate.
        assert abs(1 / v.cell.p_draw - 2.78) < 0.05

    def test_unknown_state_raises(self) -> None:
        with pytest.raises(KeyError):
            ht_live_lookup("nonsense")
