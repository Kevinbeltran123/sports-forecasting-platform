"""SportPlugin ABC -- all sports must implement this interface.

Defined here (sports/__init__.py) per D-02a.
core/ calls only this interface -- zero sport-specific code in core/.
"""

import abc
from datetime import datetime

from pydantic import BaseModel


class FixtureData(BaseModel):
    fixture_id: int
    league: str
    sport: str
    home_team: str
    away_team: str
    kickoff_utc: datetime


class FeatureMatrix(BaseModel):
    fixture_id: int
    sport: str
    league: str
    computed_at: datetime  # CRITICAL: point-in-time correctness (DATA-05)
    features: dict[str, float]
    # G-MAINT-08/11: thread fixture-level metadata so downstream Prediction
    # rows have real home/away/kickoff (no blanks; no computed_at masquerade).
    kickoff_utc: datetime
    home_team: str
    away_team: str


class ProbabilityMap(BaseModel):
    fixture_id: int
    market: str
    probabilities: dict[str, float]  # e.g., {"1": 0.45, "X": 0.28, "2": 0.27}
    model_version: str
    computed_at: datetime


class ClaudeContext(BaseModel):
    fixture_id: int
    sport: str
    summary: str


class SportPlugin(abc.ABC):
    """Plugin interface all sports must implement."""

    @abc.abstractmethod
    async def get_fixtures(self, date: datetime) -> list[FixtureData]:
        """Fetch today's fixtures from the sport's data provider."""
        ...

    @abc.abstractmethod
    async def build_features(self, fixture: FixtureData) -> FeatureMatrix:
        """Build a point-in-time safe feature matrix for a fixture."""
        ...

    @abc.abstractmethod
    async def predict(self, features: FeatureMatrix, market: str) -> ProbabilityMap:
        """Run the model and return probability map for the market."""
        ...

    @abc.abstractmethod
    async def build_claude_context(self, fixture: FixtureData) -> ClaudeContext:
        """Build context for Claude AI enrichment (Role B/C)."""
        ...

    @abc.abstractmethod
    def get_available_markets(self) -> list[str]:
        """Return market keys this sport supports (loaded from YAML)."""
        ...

    @abc.abstractmethod
    async def get_opening_odds(self, fixture_id: int) -> dict[str, float]:
        """Return opening decimal odds for the fixture's primary 1X2 market.

        Shape: {"1": <home_odd>, "X": <draw_odd>, "2": <away_odd>}.
        Returns empty dict {} when the configured bookmaker has no coverage
        (caller is responsible for handling empty-dict gracefully -- no
        exceptions are raised for coverage gaps; only network/API failures
        bubble up).
        """
        ...
