"""Heat-venue filter — Mohr 2012 + Nature SR 2024 WC2026 heat-stress.

EVIDENCE.
- Mohr, Nybo, Grantham & Racinais (2012, PLOS ONE) — high-intensity
  running drops 7% at >30°C; muscle temp >41°C in last 15 min.
- Nature SR 2024 prospective WC2026 heat-stress assessment — identifies
  Dallas, Houston, Monterrey, Atlanta as highest-risk venues for
  June 2026 daytime kickoffs (35-40°C habitual).

WHY. Lock_v1 has no venue-temperature feature. Daytime kickoffs in
very_hot venues compress high-intensity actions: corners and shots-in-box
decrease, total goals fall in second halves. Expected directional tilts:
- UNDER 2.5 goals (intensity-driven scoring drops)
- UNDER total corners (high-intensity runs that earn corners decay)
- Reduced second-half pressing -> low-block scenarios more likely

APPLICATION. Flag fixtures where (a) venue in VERY_HOT_VENUES or
HOT_VENUES, AND (b) kickoff is daytime (12:00-18:00 local). Returns a
verdict with tilt direction the analyst can incorporate. NOT a hard
exclude.

CAVEAT. Effect attenuates after round-1 as teams acclimatize. We do not
model that decay here — analyst should apply judgment for KO-stage
fixtures in same venue.
"""
from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.tournaments.patterns_v2.venues_wc2026 import (
    HOT_VENUES,
    VENUES_WC2026,
    VERY_HOT_VENUES,
    Venue,
)


HEAT_KICKOFF_DAYTIME_START_H: int = 12
HEAT_KICKOFF_DAYTIME_END_H: int = 18
"""Hours (local time) defining 'daytime kickoff' — peak heat exposure
window. Kickoffs at 21:00 local in hot venues still see warm conditions
but compressed risk because solar load is gone."""


HEAT_TILT_VERY_HOT: str = "under_2p5_goals + under_corners + low_block"
HEAT_TILT_HOT: str = "under_corners (mild) + reduced_second_half_intensity"
HEAT_TILT_NONE: str = "no heat tilt"


@dataclass(frozen=True)
class HeatVenueVerdict:
    """Outcome of heat-venue filter for a single fixture."""

    venue_slug: str
    kickoff_local_hour: int
    is_very_hot_venue: bool
    is_hot_venue: bool
    is_daytime_kickoff: bool
    tilt_direction: str
    """Empirical tilt the analyst should consider. One of HEAT_TILT_*."""
    rationale: str


def heat_venue_verdict(
    venue_slug: str, kickoff_local_hour: int
) -> HeatVenueVerdict:
    """Returns a HeatVenueVerdict for the fixture.

    Args:
        venue_slug: WC2026 venue identifier (e.g., "dallas").
        kickoff_local_hour: 0-23 hour of kickoff in venue local time.

    >>> v = heat_venue_verdict("dallas", 15)
    >>> v.is_very_hot_venue, v.is_daytime_kickoff
    (True, True)
    >>> heat_venue_verdict("vancouver", 15).tilt_direction
    'no heat tilt'
    >>> heat_venue_verdict("dallas", 21).is_daytime_kickoff
    False
    """
    venue: Venue | None = VENUES_WC2026.get(venue_slug)
    if venue is None:
        return HeatVenueVerdict(
            venue_slug=venue_slug,
            kickoff_local_hour=kickoff_local_hour,
            is_very_hot_venue=False,
            is_hot_venue=False,
            is_daytime_kickoff=False,
            tilt_direction=HEAT_TILT_NONE,
            rationale=f"unknown venue '{venue_slug}'",
        )

    is_very_hot = venue_slug in VERY_HOT_VENUES
    is_hot = venue_slug in HOT_VENUES
    is_daytime = (
        HEAT_KICKOFF_DAYTIME_START_H
        <= kickoff_local_hour
        <= HEAT_KICKOFF_DAYTIME_END_H
    )

    if not is_daytime or not is_hot:
        return HeatVenueVerdict(
            venue_slug=venue_slug,
            kickoff_local_hour=kickoff_local_hour,
            is_very_hot_venue=is_very_hot,
            is_hot_venue=is_hot,
            is_daytime_kickoff=is_daytime,
            tilt_direction=HEAT_TILT_NONE,
            rationale=(
                f"{venue.city} kickoff {kickoff_local_hour}h "
                f"({venue.climate_zone}); no heat tilt"
            ),
        )

    tmin, tmax = venue.expected_temp_c_kickoff
    if is_very_hot:
        tilt = HEAT_TILT_VERY_HOT
        evidence = "Mohr 2012 -7% high-intensity @ >30°C + Nature SR 2024"
    else:
        tilt = HEAT_TILT_HOT
        evidence = "Mohr 2012 -7% high-intensity @ >30°C"

    return HeatVenueVerdict(
        venue_slug=venue_slug,
        kickoff_local_hour=kickoff_local_hour,
        is_very_hot_venue=is_very_hot,
        is_hot_venue=is_hot,
        is_daytime_kickoff=is_daytime,
        tilt_direction=tilt,
        rationale=(
            f"{venue.city} daytime kickoff {kickoff_local_hour}h, "
            f"expected {tmin}-{tmax}°C ({venue.climate_zone}); "
            f"{evidence}"
        ),
    )
