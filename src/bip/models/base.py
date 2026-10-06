"""BaseModel ABC + PredictionRecord schema for the v4 plugin registry.

PredictionRecord lands in Supabase table `predictions_raw` (migration
20260524000000). It is the producer-side record written by any BaseModel
implementation; downstream consumers (PickEngine refactor in Sprint 2,
Delivery Worker, CLV worker) read from `predictions_raw` to apply gates,
Claude enrichment, Kelly sizing, and Telegram delivery.

Design rationale (see plan §2.2):
  - `PredictionRecord` is denormalized (selection + p_model + odds_at_pick
    in one row) by design — the `predictions_raw` table is the prediction
    *event log* — not the legacy `predictions` (probabilities dict) + `picks`
    (selection + edge) split. The two schemas coexist for 90d during the
    v3→v4 migration.
  - `BaseModel.train()` and `evaluate()` raise NotImplementedError by
    default — MundialModel (Ola C) uses a shipped lock so it never
    re-trains; LigasModel (Sprint 1) overrides both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel as _PydanticBase
from pydantic import ConfigDict, Field


class PredictionRecord(_PydanticBase):
    """One prediction emitted by a BaseModel for a fixture/market/selection.

    Maps 1:1 to a row in `predictions_raw`. The schema is intentionally
    flatter than the legacy `Prediction`+`Pick` split; downstream code
    treats this as the authoritative pre-delivery record.
    """

    model_config = ConfigDict(extra="ignore")

    source: str  # 'ligas' | 'mundial'
    fixture_id: str
    competition: str  # 'PL' | 'WC2026' | 'La Liga' | ...
    home_team: str
    away_team: str
    match_datetime: datetime
    market: str  # '1x2' | 'ou_2.5' | 'btts' | 'corners_ou' | ...
    selection: str  # 'home' | 'draw' | 'away' | 'over' | 'under' | 'yes' | 'no' | ...
    p_model: float = Field(ge=0.0, le=1.0)
    ev: float | None = None  # None when no odds available (e.g. Mundial pre-tournament)
    odds_at_pick: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    model_version: str
    status: str = "pending"  # 'pending' | 'sent' | 'killed' | 'invalidated' | 'shadow'
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_supabase_dict(self) -> dict[str, Any]:
        """Serialize for INSERT into predictions_raw (ISO timestamps)."""
        data = self.model_dump()
        data["match_datetime"] = self.match_datetime.isoformat()
        data["created_at"] = self.created_at.isoformat()
        return data


class ModelMetrics(_PydanticBase):
    """Evaluation metrics produced by BaseModel.evaluate(holdout).

    Sprint 0 ships the schema; Sprint 1 fills it via walk-forward
    backtest harness. Bootstrap-CI bounds are added in Sprint 1.
    """

    model_config = ConfigDict(extra="ignore")

    n_picks: int
    hit_rate: float
    roi: float
    brier_score: float | None = None
    clv_mean: float | None = None
    notes: str = ""


class BaseModel(ABC):
    """Contract every prediction model in the registry must implement.

    Producers (LigasModel, MundialModel) implement `can_handle` + `predict`.
    `train` and `evaluate` are optional — default raises NotImplementedError
    so MundialModel (which uses a shipped lock) doesn't have to fake them.
    """

    name: str = "unknown"
    version: str = "0.0.0"

    @abstractmethod
    def can_handle(self, fixture: dict[str, Any]) -> bool:
        """Return True if this model should produce predictions for the fixture.

        Routing is by competition + fixture metadata. The registry calls
        this on every registered model and dispatches to those returning
        True. Multiple models MAY handle the same fixture (e.g., a future
        LigasModel and a CornersModel).
        """

    @abstractmethod
    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        """Produce 0..N PredictionRecord rows for a single fixture.

        Returning [] is valid (no qualifying pick). Implementations
        must set source, market, selection, p_model, and model_version.
        Setting ev / odds_at_pick is optional (None when unknown).
        """

    def train(self, data: Any) -> None:  # noqa: ANN401
        """Optional. Override in subclasses that need offline training."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement train()")

    def evaluate(self, holdout: Any) -> ModelMetrics:  # noqa: ANN401
        """Optional. Override in subclasses that need backtest evaluation."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement evaluate()")


__all__ = ["BaseModel", "ModelMetrics", "PredictionRecord"]
