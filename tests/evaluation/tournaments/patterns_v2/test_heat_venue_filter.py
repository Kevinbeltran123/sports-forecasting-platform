"""Tests for heat-venue filter (Mohr 2012 + Nature SR 2024)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    HEAT_KICKOFF_DAYTIME_END_H,
    HEAT_KICKOFF_DAYTIME_START_H,
    HEAT_TILT_HOT,
    HEAT_TILT_NONE,
    HEAT_TILT_VERY_HOT,
    heat_venue_verdict,
)


class TestDaytimeWindow:
    def test_thresholds_lock_to_kickoff_window(self) -> None:
        # 12:00-18:00 local = peak heat exposure. Lock in to prevent drift.
        assert HEAT_KICKOFF_DAYTIME_START_H == 12
        assert HEAT_KICKOFF_DAYTIME_END_H == 18


class TestVeryHotVenueDaytime:
    @pytest.mark.parametrize("hour", [12, 13, 14, 15, 16, 17, 18])
    def test_dallas_daytime_triggers_very_hot_tilt(self, hour: int) -> None:
        v = heat_venue_verdict("dallas", hour)
        assert v.is_very_hot_venue is True
        assert v.is_daytime_kickoff is True
        assert v.tilt_direction == HEAT_TILT_VERY_HOT
        assert "Mohr 2012" in v.rationale
        assert "Nature SR 2024" in v.rationale

    @pytest.mark.parametrize("hour", [19, 20, 21, 22])
    def test_dallas_evening_no_heat_tilt(self, hour: int) -> None:
        v = heat_venue_verdict("dallas", hour)
        assert v.is_very_hot_venue is True
        assert v.is_daytime_kickoff is False
        assert v.tilt_direction == HEAT_TILT_NONE

    def test_houston_daytime(self) -> None:
        v = heat_venue_verdict("houston", 15)
        assert v.tilt_direction == HEAT_TILT_VERY_HOT

    def test_monterrey_daytime(self) -> None:
        v = heat_venue_verdict("monterrey", 15)
        assert v.tilt_direction == HEAT_TILT_VERY_HOT


class TestHotButNotVeryHotVenue:
    def test_atlanta_daytime_triggers_hot_tilt(self) -> None:
        v = heat_venue_verdict("atlanta", 15)
        assert v.is_very_hot_venue is False
        assert v.is_hot_venue is True
        assert v.tilt_direction == HEAT_TILT_HOT
        assert "Nature SR 2024" not in v.rationale  # only very_hot cites Nature
        assert "Mohr 2012" in v.rationale

    def test_kansas_city_daytime(self) -> None:
        v = heat_venue_verdict("kansas_city", 15)
        assert v.tilt_direction == HEAT_TILT_HOT

    def test_miami_daytime(self) -> None:
        v = heat_venue_verdict("miami", 15)
        assert v.tilt_direction == HEAT_TILT_HOT


class TestColdVenueNeverTilt:
    @pytest.mark.parametrize(
        "slug", ["vancouver", "seattle", "san_francisco", "mexico_city"]
    )
    def test_mild_venues_no_tilt_any_hour(self, slug: str) -> None:
        for hour in [12, 15, 18, 21]:
            v = heat_venue_verdict(slug, hour)
            assert v.tilt_direction == HEAT_TILT_NONE

    def test_los_angeles_mild_no_tilt(self) -> None:
        v = heat_venue_verdict("los_angeles", 15)
        assert v.tilt_direction == HEAT_TILT_NONE


class TestEdgeCases:
    def test_unknown_venue_returns_no_tilt(self) -> None:
        v = heat_venue_verdict("nonexistent", 15)
        assert v.tilt_direction == HEAT_TILT_NONE
        assert "unknown venue" in v.rationale

    def test_kickoff_at_boundary_hours(self) -> None:
        # 12:00 is start of window (inclusive).
        assert heat_venue_verdict("dallas", 12).is_daytime_kickoff is True
        # 18:00 is end of window (inclusive).
        assert heat_venue_verdict("dallas", 18).is_daytime_kickoff is True
        # 11 and 19 are outside.
        assert heat_venue_verdict("dallas", 11).is_daytime_kickoff is False
        assert heat_venue_verdict("dallas", 19).is_daytime_kickoff is False
