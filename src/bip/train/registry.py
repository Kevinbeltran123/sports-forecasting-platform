"""Model registry — D-05 manual CLI promotion, atomic JSON write."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from bip.core.errors import StorageError

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = 1


def _new_skeleton() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "leagues": {},
    }


@dataclass
class ModelRegistry:
    """Read/write models/football/registry.json.

    Schema:
      {
        "schema_version": 1,
        "updated_at": "...",
        "leagues": {
          "premier_league": {"production": "v3", "shadow": "v4", "history": ["v1","v2","v3"]}
        }
      }
    """
    registry_path: Path
    data: dict = field(default_factory=_new_skeleton)

    @classmethod
    def load(cls, registry_path: Path) -> "ModelRegistry":
        """Load registry from disk or return a fresh skeleton if missing."""
        try:
            if registry_path.exists():
                data = json.loads(registry_path.read_text())
            else:
                data = _new_skeleton()
            return cls(registry_path=registry_path, data=data)
        except Exception as e:
            raise StorageError(f"Failed to read registry {registry_path}: {e}") from e

    def promote(self, league: str, version: str) -> None:
        """Mark version as production for this league (D-05). No auto-promotion."""
        leagues = self.data.setdefault("leagues", {})
        entry = leagues.setdefault(
            league, {"production": None, "shadow": None, "history": []}
        )
        entry["production"] = version
        if version not in entry["history"]:
            entry["history"].append(version)
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        logger.info("model_promoted", league=league, version=version)

    def set_shadow(self, league: str, version: str | None) -> None:
        """Set shadow version (ML-05). None removes shadow."""
        leagues = self.data.setdefault("leagues", {})
        entry = leagues.setdefault(
            league, {"production": None, "shadow": None, "history": []}
        )
        entry["shadow"] = version
        if version and version not in entry["history"]:
            entry["history"].append(version)
        self.data["updated_at"] = datetime.now(UTC).isoformat()

    def save(self) -> None:
        """Atomic write: .tmp then Path.replace()."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.registry_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
            tmp.replace(self.registry_path)
        except Exception as e:
            raise StorageError(
                f"Failed to write registry {self.registry_path}: {e}"
            ) from e

    def get_production_version(self, league: str) -> str | None:
        return self.data.get("leagues", {}).get(league, {}).get("production")

    def get_shadow_version(self, league: str) -> str | None:
        return self.data.get("leagues", {}).get(league, {}).get("shadow")
