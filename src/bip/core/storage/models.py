"""Pydantic data models for Supabase entities.

These are the Python-side representations of the 6 Supabase tables.
Each model provides validation and a to_supabase_dict() method for inserts.
"""

from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field

from bip.core.types import AggregationPeriod, PickStatus


class Prediction(BaseModel):
    """Model output for a match/market combination.

    WR-06: ``created_at`` is server-defaulted (`DEFAULT now()`) on the
    Supabase ``predictions`` table. The model accepts the round-tripped
    column through ``model_config = extra="ignore"`` so reading a row from
    Supabase via ``Prediction.model_validate(row)`` does not raise on the
    extra ``created_at`` field. The model itself never writes
    ``created_at`` (the server fills it on INSERT).
    """

    model_config = ConfigDict(extra="ignore")

    fixture_id: int
    league: str
    sport: str = "football"
    market: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    probabilities: dict
    model_version: str
    is_lineup_adjusted: bool = False
    is_shadow: bool = False  # ML-05: shadow-mode flag (migration 003 adds column)

    def to_supabase_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        data = self.model_dump()
        data["kickoff_utc"] = self.kickoff_utc.isoformat()
        return data


class Pick(BaseModel):
    """A prediction that met the edge threshold — a recommended bet.

    WR-06: ``created_at`` is server-defaulted on ``picks``; ``extra="ignore"``
    lets ``Pick.model_validate(row)`` round-trip without raising on the
    extra column.
    """

    model_config = ConfigDict(extra="ignore")

    prediction_id: int | None = None
    fixture_id: int
    league: str
    sport: str = "football"
    market: str
    selection: str
    model_probability: float
    implied_probability: float
    edge: float
    best_odds: float
    bookmaker: str
    kelly_fraction: float | None = None
    suggested_stake: float | None = None
    status: PickStatus = PickStatus.pending
    # D-08: Claude Role C validation fields (added by migration 004).
    # Defaults None so Phase 1+2 picks (no Claude) still construct cleanly.
    claude_validation: str | None = None         # CONFIRM | FLAG | REJECT | SKIPPED | None
    claude_reasoning: str | None = None
    claude_summary: str | None = None
    claude_validated_at: datetime | None = None

    def to_supabase_dict(self) -> dict:
        """Convert to dict for Supabase insert.

        D-08: claude_validated_at must be ISO-formatted (PATTERNS.md drift risk #17 —
        forgetting this ships None for valid timestamps).
        """
        data = self.model_dump()
        data["status"] = self.status.value
        if self.claude_validated_at is not None:
            data["claude_validated_at"] = self.claude_validated_at.isoformat()
        return data


class OddsSnapshot(BaseModel):
    """Odds captured at a point in time for CLV tracking.

    WR-06: the SQL column is ``captured_at TIMESTAMPTZ NOT NULL DEFAULT
    now()`` — relying on the server default would silently disagree with
    the Python-side capture time by milliseconds-to-seconds and break
    CLV's "captured at exactly T+105m" semantic. ``to_supabase_dict``
    therefore always emits a concrete ``captured_at`` (defaulting to
    "now" at dump-time when the caller did not supply one); the server
    default is now defensive only.
    """

    model_config = ConfigDict(extra="ignore")

    fixture_id: int
    sport: str = "football"
    market: str
    bookmaker: str
    odds: dict
    is_closing: bool = False
    captured_at: datetime | None = None

    def to_supabase_dict(self) -> dict:
        """Convert to dict for Supabase insert.

        Always emits ``captured_at`` so the server default cannot fire and
        produce a timestamp that disagrees with the Python-side capture
        time (WR-06).
        """
        data = self.model_dump()
        captured = self.captured_at or datetime.now(UTC)
        data["captured_at"] = captured.isoformat()
        return data


class Result(BaseModel):
    """Match outcome.

    WR-06: ``created_at`` is server-defaulted on ``results``; ``extra="ignore"``
    permits round-tripping that column without a Pydantic field.
    """

    model_config = ConfigDict(extra="ignore")

    fixture_id: int
    league: str
    sport: str = "football"
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    match_stats: dict | None = None
    kickoff_utc: datetime
    finished_at: datetime | None = None

    def to_supabase_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        data = self.model_dump()
        data["kickoff_utc"] = self.kickoff_utc.isoformat()
        if self.finished_at:
            data["finished_at"] = self.finished_at.isoformat()
        return data


class ClvRecord(BaseModel):
    """Per-bet CLV (Closing Line Value) calculation.

    WR-06: ``created_at`` is server-defaulted on ``clv_records``;
    ``extra="ignore"`` permits round-tripping that column.
    """

    model_config = ConfigDict(extra="ignore")

    pick_id: int
    fixture_id: int
    sport: str = "football"
    market: str
    odds_at_pick: float
    pinnacle_closing_odds: float | None = None
    implied_prob_at_pick: float
    implied_prob_closing: float | None = None
    clv_percentage: float | None = None
    odds_fetched_at: datetime | None = None

    def to_supabase_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        data = self.model_dump()
        if self.odds_fetched_at:
            data["odds_fetched_at"] = self.odds_fetched_at.isoformat()
        return data


class PerformanceMetric(BaseModel):
    """Aggregated performance metrics by league/market/period.

    WR-06: ``created_at`` is server-defaulted on ``performance_metrics``;
    ``extra="ignore"`` permits round-tripping that column.
    """

    model_config = ConfigDict(extra="ignore")

    league: str
    sport: str = "football"
    market: str
    period: AggregationPeriod
    period_start: date
    period_end: date
    total_picks: int = Field(default=0)
    won: int = Field(default=0)
    lost: int = Field(default=0)
    void: int = Field(default=0)
    total_staked: float = Field(default=0.0)
    total_pnl: float = Field(default=0.0)
    roi: float | None = None
    yield_pct: float | None = None
    avg_clv: float | None = None
    avg_edge: float | None = None

    def to_supabase_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        data = self.model_dump()
        data["period"] = self.period.value
        data["period_start"] = self.period_start.isoformat()
        data["period_end"] = self.period_end.isoformat()
        return data
