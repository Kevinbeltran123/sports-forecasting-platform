"""bip.models — Plugin Registry for prediction models (v4 architecture).

Sprint 0 Ola B. See plan: internal design notes (not published)

This package introduces the BaseModel contract and ModelRegistry that
LigasModel (Sprint 1) and MundialModel (Sprint 0 Ola C) plug into.

NOTE: The pydantic model exposed here is PredictionRecord — deliberately
NOT named "Prediction" to avoid colliding with the existing
bip.core.storage.models.Prediction (which serializes the `predictions`
Supabase table). The two coexist: `Prediction` is the legacy v1/v2
schema; `PredictionRecord` is the v4 plugin-registry record that lands
in the new `predictions_raw` table.
"""

from bip.models.base import BaseModel, ModelMetrics, PredictionRecord
from bip.models.mundial import (
    DEFAULT_LOCK_PATH,
    MatchdayValidator,
    MundialModel,
    P_MAX_THRESHOLD,
)
from bip.models.registry import ModelRegistry

__all__ = [
    "BaseModel",
    "DEFAULT_LOCK_PATH",
    "MatchdayValidator",
    "ModelMetrics",
    "ModelRegistry",
    "MundialModel",
    "P_MAX_THRESHOLD",
    "PredictionRecord",
]
