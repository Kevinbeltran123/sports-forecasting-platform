"""WC2026 venues database — 16 sedes (USA × 11, Mexico × 3, Canada × 2).

EVIDENCE. Public sources: FIFA WC2026 venue confirmations + geographic data.

WHY. Lock_v1 has no venue feature. Three signals depend on venue characteristics:
- Heat exposure (Dallas/Houston/Monterrey daytime kickoffs)
- Altitude (Mexico City 2240m, Guadalajara 1566m)
- Travel/jet-lag (sequence of venues for a team across the tournament)

APPLICATION. Single source-of-truth for venue metadata. Other patterns_v2
modules (heat_venue_filter, altitude_verdict, travel_jetlag_verdict) read
this dict to compute their flags.

NOT a hard exclude; venue metadata feeds confidence-downgrades only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    """Venue metadata for a WC2026 host city."""

    city: str
    country: str  # "USA" / "Mexico" / "Canada"
    stadium: str
    altitude_m: int
    """Stadium elevation above sea level (meters). 0 == sea level."""
    timezone_utc_offset_h: float
    """UTC offset during June-July 2026 (DST applies). E.g., -7 for PDT."""
    expected_temp_c_kickoff: tuple[int, int]
    """Typical (min, max) °C at typical kickoff window 15:00-21:00 local
    in June-July. Sources: Nature SR 2024 WC2026 heat-stress assessment +
    NOAA / SMN climate normals."""
    climate_zone: str
    """One of: 'mild', 'warm', 'hot', 'very_hot'.
    Threshold: max kickoff temp <26°C = mild; 26-30°C = warm;
    30-34°C = hot; >34°C = very_hot."""


VENUES_WC2026: dict[str, Venue] = {
    # USA — 11 venues
    "atlanta": Venue(
        city="Atlanta",
        country="USA",
        stadium="Mercedes-Benz Stadium",
        altitude_m=320,
        timezone_utc_offset_h=-4.0,
        expected_temp_c_kickoff=(25, 32),
        climate_zone="hot",
    ),
    "boston": Venue(
        city="Boston",
        country="USA",
        stadium="Gillette Stadium",
        altitude_m=60,
        timezone_utc_offset_h=-4.0,
        expected_temp_c_kickoff=(22, 28),
        climate_zone="warm",
    ),
    "dallas": Venue(
        city="Dallas",
        country="USA",
        stadium="AT&T Stadium",
        altitude_m=175,
        timezone_utc_offset_h=-5.0,
        expected_temp_c_kickoff=(32, 38),
        climate_zone="very_hot",
    ),
    "houston": Venue(
        city="Houston",
        country="USA",
        stadium="NRG Stadium",
        altitude_m=15,
        timezone_utc_offset_h=-5.0,
        expected_temp_c_kickoff=(32, 37),
        climate_zone="very_hot",
    ),
    "kansas_city": Venue(
        city="Kansas City",
        country="USA",
        stadium="Arrowhead Stadium",
        altitude_m=285,
        timezone_utc_offset_h=-5.0,
        expected_temp_c_kickoff=(28, 33),
        climate_zone="hot",
    ),
    "los_angeles": Venue(
        city="Los Angeles",
        country="USA",
        stadium="SoFi Stadium",
        altitude_m=32,
        timezone_utc_offset_h=-7.0,
        expected_temp_c_kickoff=(20, 26),
        climate_zone="mild",
    ),
    "miami": Venue(
        city="Miami",
        country="USA",
        stadium="Hard Rock Stadium",
        altitude_m=5,
        timezone_utc_offset_h=-4.0,
        expected_temp_c_kickoff=(28, 32),
        climate_zone="hot",
    ),
    "new_york": Venue(
        city="New York/New Jersey",
        country="USA",
        stadium="MetLife Stadium",
        altitude_m=7,
        timezone_utc_offset_h=-4.0,
        expected_temp_c_kickoff=(24, 30),
        climate_zone="warm",
    ),
    "philadelphia": Venue(
        city="Philadelphia",
        country="USA",
        stadium="Lincoln Financial Field",
        altitude_m=12,
        timezone_utc_offset_h=-4.0,
        expected_temp_c_kickoff=(25, 31),
        climate_zone="warm",
    ),
    "san_francisco": Venue(
        city="San Francisco Bay Area",
        country="USA",
        stadium="Levi's Stadium",
        altitude_m=5,
        timezone_utc_offset_h=-7.0,
        expected_temp_c_kickoff=(18, 24),
        climate_zone="mild",
    ),
    "seattle": Venue(
        city="Seattle",
        country="USA",
        stadium="Lumen Field",
        altitude_m=17,
        timezone_utc_offset_h=-7.0,
        expected_temp_c_kickoff=(18, 24),
        climate_zone="mild",
    ),
    # Mexico — 3 venues
    "guadalajara": Venue(
        city="Guadalajara",
        country="Mexico",
        stadium="Estadio Akron",
        altitude_m=1566,
        timezone_utc_offset_h=-6.0,
        expected_temp_c_kickoff=(22, 26),
        climate_zone="mild",
    ),
    "mexico_city": Venue(
        city="Mexico City",
        country="Mexico",
        stadium="Estadio Banorte (Azteca)",
        altitude_m=2240,
        timezone_utc_offset_h=-6.0,
        expected_temp_c_kickoff=(20, 24),
        climate_zone="mild",
    ),
    "monterrey": Venue(
        city="Monterrey",
        country="Mexico",
        stadium="Estadio BBVA",
        altitude_m=537,
        timezone_utc_offset_h=-6.0,
        expected_temp_c_kickoff=(30, 36),
        climate_zone="very_hot",
    ),
    # Canada — 2 venues
    "toronto": Venue(
        city="Toronto",
        country="Canada",
        stadium="BMO Field",
        altitude_m=76,
        timezone_utc_offset_h=-4.0,
        expected_temp_c_kickoff=(21, 27),
        climate_zone="warm",
    ),
    "vancouver": Venue(
        city="Vancouver",
        country="Canada",
        stadium="BC Place",
        altitude_m=4,
        timezone_utc_offset_h=-7.0,
        expected_temp_c_kickoff=(18, 22),
        climate_zone="mild",
    ),
}


HIGH_ALTITUDE_VENUES: frozenset[str] = frozenset({"guadalajara", "mexico_city"})
"""Venues with altitude > 1000m. FIFA 2010 study: +0.5 gol home edge
per 1000m for unacclimatized visitors; -3.1% distance covered."""


VERY_HOT_VENUES: frozenset[str] = frozenset(
    {"dallas", "houston", "monterrey"}
)
"""Venues with max kickoff temp >34°C. Nature SR 2024 WC2026 heat-stress
assessment + Mohr 2012 PLOS ONE: high-intensity running -7% at >30°C."""


HOT_VENUES: frozenset[str] = frozenset(
    {"atlanta", "kansas_city", "miami"}
) | VERY_HOT_VENUES
"""Venues with max kickoff temp >=30°C (hot or very_hot zone)."""


def venue_by_slug(slug: str) -> Venue | None:
    """Look up venue metadata by slug.

    >>> venue_by_slug("mexico_city").altitude_m
    2240
    >>> venue_by_slug("unknown") is None
    True
    """
    return VENUES_WC2026.get(slug)
