"""ModelLoader round-trip + feature-hash gate — ML-04."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


def _write_fake_artifact(artifact_dir: Path, feature_names: list[str]) -> None:
    from bip.train.metadata import ModelMetadata, feature_set_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Fake ensemble: a fitted LogReg (tiny, serializable)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, len(feature_names)))
    y = rng.integers(0, 2, size=20)
    clf = LogisticRegression().fit(X, y)
    joblib.dump(clf, artifact_dir / "ensemble.joblib")
    joblib.dump(clf, artifact_dir / "calibrator.joblib")

    meta = ModelMetadata(
        league="premier_league",
        version="v1",
        training_date=datetime(2026, 4, 22, tzinfo=UTC),
        training_data_seasons=["2025-2026"],
        training_rows=20,
        feature_names=feature_names,
        feature_set_hash=feature_set_hash(feature_names),
        calibration_method="platt",
        calibration_samples=20,
        walk_forward_folds=3,
        walk_forward_mean_clv_pct=1.2,
        sklearn_version="1.8.0",
    )
    (artifact_dir / "metadata.json").write_text(json.dumps(meta.to_dict()))


class TestModelLoader:
    def test_save_load_roundtrip(self, tmp_model_dir):
        """Saved ensemble reloaded produces identical predictions — ML-04."""
        from bip.train.loader import ModelLoader
        from bip.train.registry import ModelRegistry

        artifact_dir = tmp_model_dir / "football" / "premier_league" / "v1"
        _write_fake_artifact(artifact_dir, ["a", "b", "c"])

        reg_path = tmp_model_dir / "football" / "registry.json"
        reg = ModelRegistry.load(reg_path)
        reg.promote("premier_league", "v1")
        reg.save()

        loader = ModelLoader(model_dir=tmp_model_dir, registry=reg)
        ensemble, meta = loader.load("premier_league")
        assert meta.league == "premier_league"
        assert meta.version == "v1"
        # Predictions are deterministic
        X = np.zeros((1, 3))
        probs1 = ensemble.predict_proba(X)
        probs2 = ensemble.predict_proba(X)
        assert np.allclose(probs1, probs2)

    def test_feature_hash_mismatch_raises(self, tmp_model_dir):
        """check_feature_hash raises StorageError on mismatch."""
        from bip.core.errors import StorageError
        from bip.train.loader import ModelLoader
        from bip.train.registry import ModelRegistry

        artifact_dir = tmp_model_dir / "football" / "premier_league" / "v1"
        _write_fake_artifact(artifact_dir, ["a", "b", "c"])
        reg_path = tmp_model_dir / "football" / "registry.json"
        reg = ModelRegistry.load(reg_path)
        reg.promote("premier_league", "v1")
        reg.save()
        loader = ModelLoader(model_dir=tmp_model_dir, registry=reg)
        _ens, meta = loader.load("premier_league")
        with pytest.raises(StorageError, match="Feature set mismatch"):
            loader.check_feature_hash(
                meta, current_feature_names=["a", "b", "c", "d"]
            )

    def test_load_rejects_path_outside_model_dir(self, tmp_model_dir, tmp_path):
        """Security: loader refuses paths resolved outside model_dir — T-reg-01."""
        from bip.core.errors import StorageError
        from bip.train.loader import ModelLoader
        from bip.train.registry import ModelRegistry

        # Point registry to a league whose artifact directory escapes model_dir via ../
        reg = ModelRegistry.load(tmp_model_dir / "football" / "registry.json")
        # Intentionally malicious league slug: '../../etc'
        reg.promote("../../etc", "v1")
        reg.save()
        loader = ModelLoader(model_dir=tmp_model_dir, registry=reg)
        with pytest.raises(StorageError, match="outside model_dir"):
            loader.load("../../etc")
