"""Altitude verdict — FIFA 2010 + Apunts 2022.

EVIDENCE.
- FIFA 2010 prospective WC South Africa altitude study — +0.5 gol home
  edge per 1000m of venue altitude when visitor is unacclimatized;
  -3.1% distance covered by sea-level visitors at >1500m.
- Apunts 2022 (Mexican professional football altitude) — ball physics
  shift at high altitude: ball travels faster, dip reduced -> long-range
  shots more dangerous, goalkeepers less effective beyond 25m.

WHY. Lock_v1 has no altitude feature. WC2026 hosts two high-altitude
venues — Guadalajara (1566m) and Mexico City (2240m, the Azteca being
the highest WC venue ever played). European and sea-level CONMEBOL teams
who arrive without ≥7 days of altitude acclimatization face quantifiable
disadvantage that no current rating absorbs.

APPLICATION. Flag fixtures where (a) venue altitude > 1000m, AND
(b) the away team is from a sea-level / low-altitude football culture.
Returns a verdict with direction (home_edge) and operational tilt
("late-game goals more likely 75-90' as visitor stamina degrades").

CAVEAT.
- We do not model arrival_days_pre_match here; if visitor arrived
  ≥7 days early the effect is largely mitigated. Analyst applies judgment.
- Acclimatized away teams (Bolivia at altitude, Ecuador at altitude,
  Mexico itself) are exempt.

NOT a hard exclude.
"""
from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.tournaments.patterns_v2.venues_wc2026 import (
    HIGH_ALTITUDE_VENUES,
    VENUES_WC2026,
    Venue,
)


ALTITUDE_HOME_EDGE_GOALS_PER_1000M: float = 0.5
"""Empirical home-edge in goals per 1000m of venue altitude (FIFA 2010)."""


ALTITUDE_THRESHOLD_M: int = 1000
"""Minimum venue altitude (meters) to trigger altitude verdict. Below this,
effects are negligible per FIFA 2010."""


# Teams whose home football culture sits at meaningful altitude (>1500m
# baseline). These teams are considered ACCLIMATIZED and NOT subject to
# the altitude penalty when playing in Guadalajara / Mexico City.
# Source: national-team home venues + city altitudes.
ACCLIMATIZED_TEAMS: frozenset[str] = frozenset(
    {
        "Mexico",  # home country, plays at Azteca regularly
        "Bolivia",  # La Paz 3640m, El Alto 4150m
        "Ecuador",  # Quito 2850m
        "Colombia",  # Bogotá 2640m partial baseline (mixed venues)
        "Peru",  # Lima sea-level but Cusco / Andean training history
    }
)


@dataclass(frozen=True)
class AltitudeVerdict:
    """Outcome of altitude verdict for a single fixture."""

    venue_slug: str
    venue_altitude_m: int
    home_team: str
    away_team: str
    is_high_altitude: bool
    away_team_acclimatized: bool
    expected_home_edge_goals: float
    """Goals of home edge attributable to altitude. 0.0 if not applicable
    or away team acclimatized."""
    tilt_direction: str
    """Operational tilt: 'home_edge + late_game_goals' or 'no altitude
    tilt'."""
    rationale: str


def altitude_verdict(
    venue_slug: str, home_team: str, away_team: str
) -> AltitudeVerdict:
    """Returns an AltitudeVerdict for the fixture.

    Args:
        venue_slug: WC2026 venue identifier.
        home_team: Home team name (for record; not used in computation —
            venue altitude is what matters).
        away_team: Away team name. Used to check ACCLIMATIZED_TEAMS.

    >>> v = altitude_verdict("mexico_city", "Mexico", "Germany")
    >>> v.is_high_altitude, v.away_team_acclimatized
    (True, False)
    >>> round(v.expected_home_edge_goals, 2)
    1.12
    >>> altitude_verdict("mexico_city", "Mexico", "Bolivia").away_team_acclimatized
    True
    >>> altitude_verdict("dallas", "USA", "England").is_high_altitude
    False
    """
    venue: Venue | None = VENUES_WC2026.get(venue_slug)
    if venue is None:
        return AltitudeVerdict(
            venue_slug=venue_slug,
            venue_altitude_m=0,
            home_team=home_team,
            away_team=away_team,
            is_high_altitude=False,
            away_team_acclimatized=False,
            expected_home_edge_goals=0.0,
            tilt_direction="no altitude tilt",
            rationale=f"unknown venue '{venue_slug}'",
        )

    is_high = venue.altitude_m >= ALTITUDE_THRESHOLD_M
    away_acclimatized = away_team in ACCLIMATIZED_TEAMS

    if not is_high:
        return AltitudeVerdict(
            venue_slug=venue_slug,
            venue_altitude_m=venue.altitude_m,
            home_team=home_team,
            away_team=away_team,
            is_high_altitude=False,
            away_team_acclimatized=away_acclimatized,
            expected_home_edge_goals=0.0,
            tilt_direction="no altitude tilt",
            rationale=(
                f"{venue.city} {venue.altitude_m}m below "
                f"{ALTITUDE_THRESHOLD_M}m threshold"
            ),
        )

    if away_acclimatized:
        return AltitudeVerdict(
            venue_slug=venue_slug,
            venue_altitude_m=venue.altitude_m,
            home_team=home_team,
            away_team=away_team,
            is_high_altitude=True,
            away_team_acclimatized=True,
            expected_home_edge_goals=0.0,
            tilt_direction="no altitude tilt",
            rationale=(
                f"{venue.city} {venue.altitude_m}m but {away_team} "
                f"acclimatized (home football culture at altitude)"
            ),
        )

    expected_edge = (
        venue.altitude_m / 1000.0
    ) * ALTITUDE_HOME_EDGE_GOALS_PER_1000M

    return AltitudeVerdict(
        venue_slug=venue_slug,
        venue_altitude_m=venue.altitude_m,
        home_team=home_team,
        away_team=away_team,
        is_high_altitude=True,
        away_team_acclimatized=False,
        expected_home_edge_goals=expected_edge,
        tilt_direction="home_edge + late_game_goals_75_90",
        rationale=(
            f"{venue.city} {venue.altitude_m}m, {away_team} sea-level; "
            f"FIFA 2010 +{ALTITUDE_HOME_EDGE_GOALS_PER_1000M} gol/1000m "
            f"= ~{expected_edge:.2f} expected home edge"
        ),
    )
