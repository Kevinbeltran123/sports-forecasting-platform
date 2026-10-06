"""D-13 — ModelMetadata logloss field round-trip and defaults."""
from __future__ import annotations

import json
from datetime import UTC, datetime


def _minimal_kwargs() -> dict:
    """Phase 2-era construction shape — MUST remain valid after 02.1 (Pitfall 7)."""
    return dict(
        league="premier_league",
        version="v1",
        training_date=datetime(2026, 4, 24, tzinfo=UTC),
        training_data_seasons=["2024-2025"],
        training_rows=100,
        feature_names=["home_xg", "away_xg"],
        feature_set_hash="abc123def4567890",
        calibration_method="platt",
        calibration_samples=50,
        walk_forward_folds=3,
        walk_forward_mean_clv_pct=1.5,
        sklearn_version="1.8.0",
    )


class TestModelMetadataLogloss:
    """D-13 — the three new fields default to None (Pitfall 7)."""

    def test_defaults_to_none(self):
        from bip.train.metadata import ModelMetadata
        meta = ModelMetadata(**_minimal_kwargs())
        assert meta.logloss_uncalibrated is None
        assert meta.logloss_calibrated is None
        assert meta.logloss_improvement_pct is None

    def test_accepts_float_values(self):
        from bip.train.metadata import ModelMetadata
        meta = ModelMetadata(
            **_minimal_kwargs(),
            logloss_uncalibrated=1.1,
            logloss_calibrated=1.0,
            logloss_improvement_pct=9.0909,
        )
        assert meta.logloss_uncalibrated == 1.1
        assert meta.logloss_calibrated == 1.0
        assert meta.logloss_improvement_pct == 9.0909

    def test_to_dict_roundtrip_populated(self):
        """JSON serialization preserves float values."""
        from bip.train.metadata import ModelMetadata
        meta = ModelMetadata(
            **_minimal_kwargs(),
            logloss_uncalibrated=1.1,
            logloss_calibrated=1.0,
            logloss_improvement_pct=9.0909,
        )
        roundtrip = json.loads(json.dumps(meta.to_dict()))
        assert roundtrip["logloss_uncalibrated"] == 1.1
        assert roundtrip["logloss_calibrated"] == 1.0
        assert roundtrip["logloss_improvement_pct"] == 9.0909

    def test_to_dict_roundtrip_none_defaults(self):
        """JSON serialization preserves None as null."""
        from bip.train.metadata import ModelMetadata
        meta = ModelMetadata(**_minimal_kwargs())
        roundtrip = json.loads(json.dumps(meta.to_dict()))
        assert roundtrip["logloss_uncalibrated"] is None
        assert roundtrip["logloss_calibrated"] is None
        assert roundtrip["logloss_improvement_pct"] is None

    def test_phase_2_construction_still_works(self):
        """Pitfall 7: existing test call sites must not break."""
        from bip.train.metadata import ModelMetadata
        # No logloss fields passed — MUST NOT raise ValidationError.
        meta = ModelMetadata(**_minimal_kwargs())
        assert meta.league == "premier_league"

    def test_negative_improvement_allowed(self):
        """Pitfall 6: calibration may regress; negative pct is a valid value."""
        from bip.train.metadata import ModelMetadata
        meta = ModelMetadata(
            **_minimal_kwargs(),
            logloss_uncalibrated=1.0,
            logloss_calibrated=1.02,
            logloss_improvement_pct=-2.0,
        )
        assert meta.logloss_improvement_pct == -2.0
