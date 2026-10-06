"""Sportmonks v3 Football API integration.

Spike: live-edge-with-sportmonks (2026-05-09 → 2026-05-23, 14d trial window).

This package provides:
- ``client`` — async httpx client with rate-limit awareness
- ``schemas`` — Pydantic v2 models for API responses
- ``types`` — type_id constants (statistics, predictions)
- ``cache`` — Parquet store for snapshot persistence

The Sportmonks Pro subscription gives us:
- Livescores (poll every 10s)
- Per-fixture statistics with 41 type_ids (corners, shots, possession,
  big chances, dangerous attacks, etc.)
- Trends (minute-by-minute stats — same 41 types but time-series)
- Pressure (per-minute 0-100 attack pressure index)
- Predictions (29 markets pre-built: 1X2, BTTS, OU 1.5/2.5/3.5, HT/FT,
  First Half Winner, Correct Score, Double Chance, Team to Score First)
- Value bets (Sportmonks own value tags)
- In-play odds (42 markets including 1st Half Goals, HT Result,
  HT/FT, Asian Handicap HT, Asian Corners HT)
- Pre-match odds (3000+ entries per fixture across many bookmakers)

What we do NOT have access to (subscription gated):
- ``premiumOdds`` (403)
- ``expectedLineups`` (403)
- ``aiOverviews`` (404)

What is documented but missing for our fixture:
- ``xGFixture`` (200 but key absent from payload — may not be
  populated for all leagues; we use Big Chances Created/Missed as proxy)
- ``ballCoordinates``, ``weatherReport``, ``tvStations`` (empty for sample)
"""

from bip.sports.football.sportmonks.client import SportmonksClient
from bip.sports.football.sportmonks.schemas import (
    Fixture,
    FixtureStatistic,
    LiveScore,
    Prediction,
    Trend,
    PressureMinute,
)
from bip.sports.football.sportmonks.types import (
    StatType,
    PredictionType,
    MarketID,
)

__all__ = [
    "SportmonksClient",
    "Fixture",
    "FixtureStatistic",
    "LiveScore",
    "Prediction",
    "Trend",
    "PressureMinute",
    "StatType",
    "PredictionType",
    "MarketID",
]
