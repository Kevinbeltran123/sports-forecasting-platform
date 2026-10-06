"""YAML loader for Tournament and QualifyingLeague configs.

Configs live in src/bip/evaluation/tournaments/configs/. The loader resolves
paths relative to the package, not the CWD (lesson learned from
fix(builder): anchor _LEAGUES_DIR to package root, not CWD — commit 91f4acc).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bip.core.errors import ConfigurationError
from bip.evaluation.tournaments.models import (
    QualifyingLeague,
    Tournament,
)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_tournament(slug: str) -> Tournament:
    """Load a tournament config by slug (filename stem)."""
    path = _CONFIGS_DIR / f"{slug}.yaml"
    if not path.exists():
        raise ConfigurationError(f"Tournament config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return Tournament.model_validate(data)


def load_qualifying_leagues() -> list[QualifyingLeague]:
    """Load _qualifying_leagues.yaml — confederation -> AF league mappings."""
    path = _CONFIGS_DIR / "_qualifying_leagues.yaml"
    if not path.exists():
        raise ConfigurationError(f"Qualifying leagues config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return [QualifyingLeague.model_validate(entry) for entry in data["leagues"]]
