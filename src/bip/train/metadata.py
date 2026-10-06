"""Model metadata schema + feature set hash — ML-04."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def feature_set_hash(feature_names: list[str]) -> str:
    """Deterministic 16-char hex hash of sorted(feature_names) — ML-04.

    Detects when a model was trained with a different feature set without
    comparing full metadata.
    """
    canonical = json.dumps(sorted(feature_names)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


class ModelMetadata(BaseModel):
    """Serialized to models/football/{league}/{version}/metadata.json."""

    sport: str = "football"
    league: str
    version: str
    training_date: datetime
    training_data_seasons: list[str]
    training_rows: int
    feature_names: list[str]
    feature_set_hash: str
    calibration_method: str            # "platt" | "isotonic"
    calibration_samples: int
    walk_forward_folds: int
    walk_forward_mean_clv_pct: float | None   # promotion criterion (None when no fold has >=20 picks; D-11)
    walk_forward_fold_details: list[dict] = Field(default_factory=list)
    base_model_params: dict[str, Any] = Field(default_factory=dict)
    base_model_packages: dict[str, str] = Field(default_factory=dict)
    sklearn_version: str
    null_clv_rows: int = 0
    slippage_pct: float = 0.015
    git_commit: str | None = None

    # --- NEW in Phase 02.1 (D-13) ---
    # Nullable defaults preserve Phase 2 test compatibility (research finding 5; Pitfall 7).
    # logloss_improvement_pct: POSITIVE = calibration improved loss (Pitfall 6).
    #   Negative values are valid — they signal a regression, which D-16 exit
    #   criterion flags as a failed gate.
    logloss_uncalibrated: float | None = None
    logloss_calibrated: float | None = None
    logloss_improvement_pct: float | None = None

    def to_dict(self) -> dict:
        data = self.model_dump()
        data["training_date"] = self.training_date.isoformat()
        return data
