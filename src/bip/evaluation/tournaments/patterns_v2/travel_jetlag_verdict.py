"""Travel jet-lag verdict — Janse van Rensburg 2021 + Fowler 2014/2017.

EVIDENCE.
- Janse van Rensburg et al. (2021, Sports Medicine) — IOC-endorsed
  consensus: ~1 day recovery per time-zone crossed; 1.5 days for sleep
  quality. ≥4 days arrival buffer recommended for full performance.
- Fowler, Duffield & Murray (2014, 2017) A-League long-haul travel
  series — EASTWARD travel significantly worse than westward.
  Maximal-sprint and repeated-sprint performance reduced for 72h
  after eastward arrival; westward primarily affects sleep, not output.
- Song, Severini & Allen (2017, Frontiers Physiology) NBA — eastbound
  jet lag reduces home-team win probability by 3.5-5% even with a
  single time-zone shift.

WHY. WC2026 forces teams to travel between venues across up to 4 time
zones (Vancouver UTC-7 to Boston UTC-4). Asian and Oceanic teams
(Japan, Korea, Australia, Saudi, Iran) flying to USA East Coast face
eastbound jet lag with limited buffer. European teams flying westward
to USA/Mexico are penalized less than the symmetric eastbound case.

APPLICATION. Given a team's recent journey (origin venue + arrival
venue + days_since_arrival), flag fixtures where the team has
insufficient recovery time after meaningful zone-crossing. Direction
(east/west) matters: eastward triggers more conservative thresholds.

CAVEAT.
- We model only the zone-crossing penalty, not absolute jet-lag from
  baseline residence (e.g., Saudi team UTC+3 flying to Boston UTC-4 has
  cumulative impact beyond venue-to-venue computation).
- Teams that arrived in tournament country ≥7 days pre-tournament are
  considered baseline-acclimatized; the verdict applies only to
  intra-tournament inter-venue moves.

NOT a hard exclude; provides a confidence-downgrade or tilt for the
affected side.
"""
from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.tournaments.patterns_v2.venues_wc2026 import (
    VENUES_WC2026,
    Venue,
)


JETLAG_DAYS_PER_ZONE: float = 1.0
"""Days of recovery needed per time-zone crossed (Janse van Rensburg
2021 consensus)."""


EASTWARD_PENALTY_MULTIPLIER: float = 1.5
"""Eastward travel is ~1.5x as costly as westward for athletic
performance (Fowler 2014/2017 series, Song et al. 2017)."""


JETLAG_ZONE_THRESHOLD: int = 2
"""Minimum time-zones crossed to consider triggering a verdict.
Single-zone shifts are below the noise floor of the literature."""


@dataclass(frozen=True)
class TravelJetlagVerdict:
    """Outcome of travel jet-lag verdict for a single team-fixture."""

    team: str
    origin_venue_slug: str
    arrival_venue_slug: str
    days_since_arrival: int
    zones_crossed: int
    """Absolute number of time-zones crossed (positive integer)."""
    direction: str
    """'eastward' / 'westward' / 'none'. Eastward more costly."""
    expected_recovery_days_needed: float
    """Recommended days of buffer for full performance recovery."""
    has_insufficient_recovery: bool
    tilt_direction: str
    """'fade_this_team' if recovery insufficient; 'no jetlag tilt' else."""
    rationale: str


def _zones_between(origin: Venue, arrival: Venue) -> tuple[int, str]:
    """Returns (abs_zones_crossed, direction).

    Direction: 'eastward' if arrival is east of origin (smaller UTC
    offset; e.g., Vancouver UTC-7 -> Boston UTC-4 is eastward).
    'westward' otherwise. 'none' for zero crossing.
    """
    delta_h = arrival.timezone_utc_offset_h - origin.timezone_utc_offset_h
    abs_zones = int(abs(delta_h))
    if abs_zones == 0:
        return 0, "none"
    # If arrival has LARGER UTC offset (e.g., -4 > -7), it is east of origin.
    direction = "eastward" if delta_h > 0 else "westward"
    return abs_zones, direction


def travel_jetlag_verdict(
    team: str,
    origin_venue_slug: str,
    arrival_venue_slug: str,
    days_since_arrival: int,
) -> TravelJetlagVerdict:
    """Returns a TravelJetlagVerdict for the team's intra-tournament move.

    Args:
        team: Team name (for the verdict record).
        origin_venue_slug: Previous match venue.
        arrival_venue_slug: Upcoming match venue.
        days_since_arrival: Calendar days team has been at arrival venue
            before next kickoff. 0 = arriving day of match.

    >>> v = travel_jetlag_verdict("Japan", "los_angeles", "boston", 1)
    >>> v.zones_crossed, v.direction
    (3, 'eastward')
    >>> v.has_insufficient_recovery
    True
    >>> travel_jetlag_verdict("USA", "boston", "los_angeles", 1).direction
    'westward'
    >>> travel_jetlag_verdict("USA", "boston", "philadelphia", 0).tilt_direction
    'no jetlag tilt'
    """
    origin = VENUES_WC2026.get(origin_venue_slug)
    arrival = VENUES_WC2026.get(arrival_venue_slug)

    if origin is None or arrival is None:
        return TravelJetlagVerdict(
            team=team,
            origin_venue_slug=origin_venue_slug,
            arrival_venue_slug=arrival_venue_slug,
            days_since_arrival=days_since_arrival,
            zones_crossed=0,
            direction="none",
            expected_recovery_days_needed=0.0,
            has_insufficient_recovery=False,
            tilt_direction="no jetlag tilt",
            rationale=(
                f"unknown venue(s): origin='{origin_venue_slug}' "
                f"arrival='{arrival_venue_slug}'"
            ),
        )

    zones, direction = _zones_between(origin, arrival)

    if zones < JETLAG_ZONE_THRESHOLD:
        return TravelJetlagVerdict(
            team=team,
            origin_venue_slug=origin_venue_slug,
            arrival_venue_slug=arrival_venue_slug,
            days_since_arrival=days_since_arrival,
            zones_crossed=zones,
            direction=direction,
            expected_recovery_days_needed=0.0,
            has_insufficient_recovery=False,
            tilt_direction="no jetlag tilt",
            rationale=(
                f"{zones} zone(s) {direction} below "
                f"{JETLAG_ZONE_THRESHOLD}-zone threshold"
            ),
        )

    if direction == "eastward":
        recovery_needed = (
            zones * JETLAG_DAYS_PER_ZONE * EASTWARD_PENALTY_MULTIPLIER
        )
        evidence = (
            "Fowler 2014/2017: eastward ~1.5x worse + "
            "Janse van Rensburg 2021: 1 day/zone"
        )
    else:
        recovery_needed = zones * JETLAG_DAYS_PER_ZONE
        evidence = "Janse van Rensburg 2021: 1 day/zone (westward baseline)"

    insufficient = days_since_arrival < recovery_needed
    tilt = "fade_this_team" if insufficient else "no jetlag tilt"

    if insufficient:
        rationale = (
            f"{team} {zones} zones {direction} "
            f"{origin.city}->{arrival.city}, "
            f"{days_since_arrival}d since arrival < "
            f"{recovery_needed:.1f}d needed; {evidence}"
        )
    else:
        rationale = (
            f"{team} {zones} zones {direction} but "
            f"{days_since_arrival}d >= {recovery_needed:.1f}d needed "
            f"(recovered); {evidence}"
        )

    return TravelJetlagVerdict(
        team=team,
        origin_venue_slug=origin_venue_slug,
        arrival_venue_slug=arrival_venue_slug,
        days_since_arrival=days_since_arrival,
        zones_crossed=zones,
        direction=direction,
        expected_recovery_days_needed=recovery_needed,
        has_insufficient_recovery=insufficient,
        tilt_direction=tilt,
        rationale=rationale,
    )
