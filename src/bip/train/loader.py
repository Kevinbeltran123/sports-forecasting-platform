"""Model loader — ML-04 with security-constrained path + feature-hash gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import structlog

from bip.core.errors import StorageError
from bip.train.metadata import ModelMetadata, feature_set_hash
from bip.train.registry import ModelRegistry

logger = structlog.get_logger(__name__)


@dataclass
class ModelLoader:
    """Loads saved ensemble + calibrator from models/football/{league}/{version}/.

    Security (ASVS V4):
      - Only accepts absolute paths under model_dir; rejects any path that
        resolves outside (defends against symlink / path-traversal tampering).
    """
    model_dir: Path
    registry: ModelRegistry

    def load(
        self, league: str, version: str | None = None,
    ) -> tuple[object, ModelMetadata]:
        """Load (calibrated_ensemble, metadata) for a league/version.

        If version is None, uses registry's production version for the league.
        Raises StorageError on missing artifacts, path-traversal, or feature-hash
        mismatch.
        """
        resolved_version = version or self.registry.get_production_version(league)
        if resolved_version is None:
            raise StorageError(f"No production version registered for {league}")

        artifact_dir = self.model_dir / "football" / league / resolved_version
        try:
            # Path validation — reject anything outside model_dir
            artifact_dir.resolve().relative_to(self.model_dir.resolve())
        except ValueError as e:
            raise StorageError(
                f"Refusing to load from path outside model_dir: {artifact_dir}"
            ) from e

        meta_path = artifact_dir / "metadata.json"
        ensemble_path = artifact_dir / "ensemble.joblib"
        if not meta_path.exists():
            raise StorageError(f"metadata.json missing at {meta_path}")
        if not ensemble_path.exists():
            raise StorageError(f"ensemble.joblib missing at {ensemble_path}")

        try:
            meta = ModelMetadata.model_validate_json(meta_path.read_text())
            ensemble = joblib.load(ensemble_path)
        except Exception as e:
            raise StorageError(
                f"Failed to load artifact at {artifact_dir}: {e}"
            ) from e

        logger.info("model_loaded", league=league, version=resolved_version)
        return ensemble, meta

    def check_feature_hash(
        self, meta: ModelMetadata, current_feature_names: list[str],
    ) -> None:
        """Raise StorageError if current_feature_names hash differs from meta.feature_set_hash."""
        current_hash = feature_set_hash(current_feature_names)
        if meta.feature_set_hash != current_hash:
            raise StorageError(
                f"Feature set mismatch for {meta.league}/{meta.version}: "
                f"model={meta.feature_set_hash!r} current={current_hash!r}"
            )
