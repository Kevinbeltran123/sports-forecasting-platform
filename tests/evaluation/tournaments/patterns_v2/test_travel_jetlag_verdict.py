"""Tests for travel jet-lag verdict (Janse van Rensburg 2021 + Fowler 2014/2017)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    EASTWARD_PENALTY_MULTIPLIER,
    JETLAG_DAYS_PER_ZONE,
    JETLAG_ZONE_THRESHOLD,
    travel_jetlag_verdict,
)


class TestConstants:
    def test_days_per_zone_baseline(self) -> None:
        # Janse van Rensburg 2021 IOC consensus.
        assert JETLAG_DAYS_PER_ZONE == 1.0

    def test_eastward_multiplier(self) -> None:
        # Fowler 2014/2017: eastward ~1.5x as costly.
        assert EASTWARD_PENALTY_MULTIPLIER == 1.5

    def test_zone_threshold_two(self) -> None:
        # Single-zone shifts below literature noise floor.
        assert JETLAG_ZONE_THRESHOLD == 2


class TestZoneCounting:
    def test_la_to_boston_three_zones_eastward(self) -> None:
        v = travel_jetlag_verdict("Japan", "los_angeles", "boston", 0)
        assert v.zones_crossed == 3
        assert v.direction == "eastward"

    def test_boston_to_la_three_zones_westward(self) -> None:
        v = travel_jetlag_verdict("USA", "boston", "los_angeles", 0)
        assert v.zones_crossed == 3
        assert v.direction == "westward"

    def test_same_venue_zero_zones(self) -> None:
        v = travel_jetlag_verdict("Mexico", "mexico_city", "mexico_city", 0)
        assert v.zones_crossed == 0
        assert v.direction == "none"

    def test_intra_east_coast_zero_zones(self) -> None:
        # All UTC-4: atlanta, new_york, toronto, miami, boston, philadelphia
        v = travel_jetlag_verdict("USA", "new_york", "atlanta", 0)
        assert v.zones_crossed == 0


class TestEastwardPenalty:
    def test_three_zones_eastward_needs_4p5_days(self) -> None:
        # 3 zones * 1.0 day/zone * 1.5 eastward = 4.5 days
        v = travel_jetlag_verdict("Japan", "los_angeles", "boston", 5)
        assert v.expected_recovery_days_needed == pytest.approx(4.5, abs=0.01)
        # 5 days arrived >= 4.5 needed -> no tilt
        assert v.has_insufficient_recovery is False
        assert v.tilt_direction == "no jetlag tilt"

    def test_three_zones_eastward_insufficient_at_4_days(self) -> None:
        v = travel_jetlag_verdict("Japan", "los_angeles", "boston", 4)
        # 4 < 4.5 -> insufficient
        assert v.has_insufficient_recovery is True
        assert v.tilt_direction == "fade_this_team"

    def test_two_zones_eastward_needs_three_days(self) -> None:
        v = travel_jetlag_verdict("Korea", "los_angeles", "kansas_city", 2)
        # LA UTC-7, KC UTC-5 = 2 zones eastward
        assert v.zones_crossed == 2
        assert v.direction == "eastward"
        assert v.expected_recovery_days_needed == pytest.approx(3.0, abs=0.01)
        assert v.has_insufficient_recovery is True


class TestWestwardBaseline:
    def test_three_zones_westward_needs_three_days(self) -> None:
        v = travel_jetlag_verdict("USA", "boston", "los_angeles", 4)
        # 3 zones * 1.0 day/zone * 1.0 (westward) = 3 days
        assert v.expected_recovery_days_needed == pytest.approx(3.0, abs=0.01)
        assert v.has_insufficient_recovery is False

    def test_three_zones_westward_insufficient_at_2_days(self) -> None:
        v = travel_jetlag_verdict("USA", "boston", "los_angeles", 2)
        assert v.has_insufficient_recovery is True
        assert v.tilt_direction == "fade_this_team"


class TestThresholdBelowDoesNotTrigger:
    def test_single_zone_eastward_no_tilt(self) -> None:
        # mexico_city UTC-6 -> dallas UTC-5 = 1 zone eastward
        v = travel_jetlag_verdict(
            "Mexico", "mexico_city", "dallas", 0
        )
        assert v.zones_crossed == 1
        assert v.has_insufficient_recovery is False
        assert v.tilt_direction == "no jetlag tilt"
        assert "below" in v.rationale and "threshold" in v.rationale

    def test_same_timezone_no_tilt(self) -> None:
        v = travel_jetlag_verdict("USA", "atlanta", "miami", 0)
        assert v.zones_crossed == 0
        assert v.has_insufficient_recovery is False


class TestEastwardWorseThanWestward:
    """Verify Fowler 2014/2017 directional asymmetry quantitatively."""

    def test_symmetric_distance_asymmetric_recovery(self) -> None:
        east = travel_jetlag_verdict("Japan", "los_angeles", "boston", 0)
        west = travel_jetlag_verdict("USA", "boston", "los_angeles", 0)
        # Both cross 3 zones but eastward needs 1.5x more recovery.
        assert east.zones_crossed == west.zones_crossed == 3
        assert (
            east.expected_recovery_days_needed
            > west.expected_recovery_days_needed
        )
        assert east.expected_recovery_days_needed == pytest.approx(
            west.expected_recovery_days_needed * EASTWARD_PENALTY_MULTIPLIER,
            abs=0.01,
        )


class TestEdgeCases:
    def test_unknown_origin_returns_no_tilt(self) -> None:
        v = travel_jetlag_verdict(
            "Japan", "nonexistent", "boston", 0
        )
        assert v.tilt_direction == "no jetlag tilt"
        assert "unknown" in v.rationale.lower()

    def test_unknown_arrival_returns_no_tilt(self) -> None:
        v = travel_jetlag_verdict(
            "Japan", "los_angeles", "nonexistent", 0
        )
        assert v.tilt_direction == "no jetlag tilt"

    def test_rationale_cites_evidence(self) -> None:
        v_east = travel_jetlag_verdict("Japan", "los_angeles", "boston", 0)
        assert "Fowler" in v_east.rationale
        assert "Janse van Rensburg" in v_east.rationale
        v_west = travel_jetlag_verdict("USA", "boston", "los_angeles", 0)
        assert "Janse van Rensburg" in v_west.rationale
