"""FootballPlugin.predict() E2E -- ML-01 + ML-05."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


def _plant_model(model_root: Path, league: str, version: str, n_features: int = 3) -> None:
    """Install a tiny fitted model + metadata at model_root/football/{league}/{version}/."""
    from bip.train.metadata import ModelMetadata, feature_set_hash

    artifact_dir = model_root / "football" / league / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, n_features))  # noqa: N806 — sklearn convention
    y = rng.integers(0, 3, size=30)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    joblib.dump(clf, artifact_dir / "ensemble.joblib")
    joblib.dump(clf, artifact_dir / "calibrator.joblib")

    feature_names = [f"f{i}" for i in range(n_features)]
    meta = ModelMetadata(
        league=league,
        version=version,
        training_date=datetime(2026, 4, 22, tzinfo=UTC),
        training_data_seasons=["2025-2026"],
        training_rows=30,
        feature_names=feature_names,
        feature_set_hash=feature_set_hash(feature_names),
        calibration_method="platt",
        calibration_samples=30,
        walk_forward_folds=3,
        walk_forward_mean_clv_pct=1.5,
        sklearn_version="1.8.0",
    )
    (artifact_dir / "metadata.json").write_text(json.dumps(meta.to_dict()))


def _settings_for_model_dir(monkeypatch, model_dir: Path):
    """Build a Settings instance pointed at a tmp model_dir."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-af-key")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds-key")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from bip.core.settings import Settings
    return Settings()


def _features(league: str, n: int = 3):
    from bip.sports import FeatureMatrix
    return FeatureMatrix(
        fixture_id=12345,
        sport="football",
        league=league,
        computed_at=datetime(2026, 4, 22, 15, 0, 0, tzinfo=UTC),
        features={f"f{i}": 0.5 for i in range(n)},
        kickoff_utc=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        home_team="Home FC",
        away_team="Away FC",
    )


class TestFootballPluginPredict:
    async def test_predict_returns_probability_map(self, tmp_model_dir, monkeypatch):
        """FootballPlugin.predict() returns ProbabilityMap with probs summing to 1 -- ML-01."""
        from bip.sports.football.plugin import FootballPlugin
        from bip.train.registry import ModelRegistry

        _plant_model(tmp_model_dir, "premier_league", "v1")
        reg = ModelRegistry.load(tmp_model_dir / "football" / "registry.json")
        reg.promote("premier_league", "v1")
        reg.save()

        settings = _settings_for_model_dir(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)
        fm = _features("premier_league")
        result = await plugin.predict(features=fm, market="1X2")
        assert set(result.probabilities.keys()) == {"1", "X", "2"}
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
        assert result.fixture_id == 12345
        assert result.market == "1X2"

    async def test_predict_uses_model_version_from_registry(self, tmp_model_dir, monkeypatch):
        """ProbabilityMap.model_version matches registry production version -- ML-04."""
        from bip.sports.football.plugin import FootballPlugin
        from bip.train.registry import ModelRegistry

        _plant_model(tmp_model_dir, "la_liga", "v7")
        reg = ModelRegistry.load(tmp_model_dir / "football" / "registry.json")
        reg.promote("la_liga", "v7")
        reg.save()

        settings = _settings_for_model_dir(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)
        fm = _features("la_liga")
        result = await plugin.predict(features=fm, market="1X2")
        assert result.model_version == "v7"

    async def test_shadow_and_production_paths(self, tmp_model_dir, monkeypatch):
        """When registry has both production and shadow, predict writes both -- ML-05."""
        from bip.sports.football.plugin import FootballPlugin
        from bip.train.registry import ModelRegistry

        _plant_model(tmp_model_dir, "bundesliga", "v3")
        _plant_model(tmp_model_dir, "bundesliga", "v4")
        reg = ModelRegistry.load(tmp_model_dir / "football" / "registry.json")
        reg.promote("bundesliga", "v3")
        reg.set_shadow("bundesliga", "v4")
        reg.save()

        # Mock repository captures both writes
        mock_repo = MagicMock()
        settings = _settings_for_model_dir(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings, prediction_repo=mock_repo)
        fm = _features("bundesliga")
        result = await plugin.predict(features=fm, market="1X2")
        assert result.model_version == "v3"  # caller gets production
        # Both production + shadow writes were attempted
        assert mock_repo.insert.call_count == 2
        written_preds = [call.args[0] for call in mock_repo.insert.call_args_list]
        is_shadow_flags = sorted([p.is_shadow for p in written_preds])
        assert is_shadow_flags == [False, True]
        shadow_pred = next(p for p in written_preds if p.is_shadow)
        assert shadow_pred.model_version == "v4"

    async def test_cold_start_returns_stub(self, tmp_model_dir, monkeypatch):
        """No production model registered -> stub 1/3-1/3-1/3, model_version='stub-v0'."""
        from bip.sports.football.plugin import FootballPlugin

        settings = _settings_for_model_dir(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)
        fm = _features("serie_a")
        result = await plugin.predict(features=fm, market="1X2")
        assert result.model_version == "stub-v0"
        # Probabilities are uniform
        assert result.probabilities["1"] == pytest.approx(1 / 3)
        assert result.probabilities["X"] == pytest.approx(1 / 3)
        assert result.probabilities["2"] == pytest.approx(1 / 3)
