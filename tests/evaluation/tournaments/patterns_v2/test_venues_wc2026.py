"""Tests for WC2026 venues database."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    HIGH_ALTITUDE_VENUES,
    HOT_VENUES,
    VENUES_WC2026,
    VERY_HOT_VENUES,
    venue_by_slug,
)


class TestVenuesDB:
    def test_sixteen_venues_total(self) -> None:
        # FIFA-confirmed: 11 USA + 3 Mexico + 2 Canada = 16
        assert len(VENUES_WC2026) == 16

    def test_eleven_usa_venues(self) -> None:
        usa = [v for v in VENUES_WC2026.values() if v.country == "USA"]
        assert len(usa) == 11

    def test_three_mexico_venues(self) -> None:
        mex = [v for v in VENUES_WC2026.values() if v.country == "Mexico"]
        assert len(mex) == 3

    def test_two_canada_venues(self) -> None:
        can = [v for v in VENUES_WC2026.values() if v.country == "Canada"]
        assert len(can) == 2

    def test_mexico_city_is_highest(self) -> None:
        # Azteca is the highest WC venue ever played (2240m).
        assert VENUES_WC2026["mexico_city"].altitude_m == 2240
        assert (
            VENUES_WC2026["mexico_city"].altitude_m
            == max(v.altitude_m for v in VENUES_WC2026.values())
        )

    def test_guadalajara_above_threshold(self) -> None:
        assert VENUES_WC2026["guadalajara"].altitude_m == 1566

    def test_monterrey_is_very_hot(self) -> None:
        v = VENUES_WC2026["monterrey"]
        assert v.climate_zone == "very_hot"
        assert v.expected_temp_c_kickoff[1] >= 34

    def test_venue_by_slug_returns_none_for_unknown(self) -> None:
        assert venue_by_slug("nonexistent") is None


class TestVenueClassifications:
    def test_high_altitude_set(self) -> None:
        assert HIGH_ALTITUDE_VENUES == frozenset(
            {"guadalajara", "mexico_city"}
        )

    def test_very_hot_set(self) -> None:
        assert VERY_HOT_VENUES == frozenset(
            {"dallas", "houston", "monterrey"}
        )

    def test_hot_includes_very_hot(self) -> None:
        assert VERY_HOT_VENUES <= HOT_VENUES

    @pytest.mark.parametrize(
        "slug,expected_zone",
        [
            ("vancouver", "mild"),
            ("seattle", "mild"),
            ("mexico_city", "mild"),
            ("toronto", "warm"),
            ("new_york", "warm"),
            ("atlanta", "hot"),
            ("dallas", "very_hot"),
            ("houston", "very_hot"),
            ("monterrey", "very_hot"),
        ],
    )
    def test_climate_zone_classification(
        self, slug: str, expected_zone: str
    ) -> None:
        assert VENUES_WC2026[slug].climate_zone == expected_zone


class TestTimezones:
    @pytest.mark.parametrize(
        "slug,expected_offset",
        [
            ("los_angeles", -7.0),  # PDT
            ("san_francisco", -7.0),
            ("seattle", -7.0),
            ("vancouver", -7.0),
            ("dallas", -5.0),  # CDT
            ("kansas_city", -5.0),
            ("houston", -5.0),
            ("mexico_city", -6.0),  # CST no DST in 2024+
            ("guadalajara", -6.0),
            ("monterrey", -6.0),
            ("atlanta", -4.0),  # EDT
            ("new_york", -4.0),
            ("toronto", -4.0),
            ("miami", -4.0),
        ],
    )
    def test_dst_offsets(self, slug: str, expected_offset: float) -> None:
        assert VENUES_WC2026[slug].timezone_utc_offset_h == expected_offset
