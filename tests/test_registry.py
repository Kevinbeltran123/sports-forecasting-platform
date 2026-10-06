"""Model registry + metadata schema — ML-04."""

from __future__ import annotations

import json
from datetime import UTC, datetime


class TestModelMetadata:
    def test_metadata_schema(self):
        """ModelMetadata has all required fields (training_date, feature_set_hash, ...)."""
        from bip.train.metadata import ModelMetadata
        m = ModelMetadata(
            league="premier_league",
            version="v1",
            training_date=datetime(2026, 4, 22, tzinfo=UTC),
            training_data_seasons=["2023-2024", "2024-2025", "2025-2026"],
            training_rows=1000,
            feature_names=["a", "b", "c"],
            feature_set_hash="abc123",
            calibration_method="platt",
            calibration_samples=200,
            walk_forward_folds=5,
            walk_forward_mean_clv_pct=2.5,
            sklearn_version="1.8.0",
        )
        assert m.sport == "football"
        assert m.slippage_pct == 0.015  # default

    def test_feature_set_hash(self):
        """feature_set_hash = sha256(sorted(feature_names))[:16] — deterministic."""
        from bip.train.metadata import feature_set_hash
        h1 = feature_set_hash(["elo_home", "elo_away", "form_3"])
        h2 = feature_set_hash(["form_3", "elo_away", "elo_home"])  # different order
        assert h1 == h2, "Hash must be order-invariant (sorted)"
        assert len(h1) == 16

    def test_feature_set_hash_detects_addition(self):
        """Adding a feature changes the hash."""
        from bip.train.metadata import feature_set_hash
        h1 = feature_set_hash(["a", "b"])
        h2 = feature_set_hash(["a", "b", "c"])
        assert h1 != h2


class TestModelRegistry:
    def test_promote_atomic(self, tmp_model_dir):
        """promote() writes via .tmp then renames — no stray .tmp left behind."""
        from bip.train.registry import ModelRegistry
        reg_path = tmp_model_dir / "football" / "registry.json"
        reg = ModelRegistry.load(reg_path)
        reg.promote("premier_league", "v1")
        reg.save()
        assert reg_path.exists()
        tmp_files = list((tmp_model_dir / "football").glob("*.tmp"))
        assert tmp_files == []

    def test_promote_updates_production(self, tmp_model_dir):
        """promote(league, version) sets leagues[league]['production'] = version."""
        from bip.train.registry import ModelRegistry
        reg_path = tmp_model_dir / "football" / "registry.json"
        reg = ModelRegistry.load(reg_path)
        reg.promote("la_liga", "v7")
        reg.save()
        data = json.loads(reg_path.read_text())
        assert data["leagues"]["la_liga"]["production"] == "v7"

    def test_history_appended(self, tmp_model_dir):
        """promote() appends version to leagues[league]['history']."""
        from bip.train.registry import ModelRegistry
        reg_path = tmp_model_dir / "football" / "registry.json"
        reg = ModelRegistry.load(reg_path)
        reg.promote("bundesliga", "v1")
        reg.promote("bundesliga", "v2")
        reg.save()
        data = json.loads(reg_path.read_text())
        assert data["leagues"]["bundesliga"]["history"] == ["v1", "v2"]
