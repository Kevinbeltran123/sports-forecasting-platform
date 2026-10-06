"""Integration test: MundialModel registered in ModelRegistry routes correctly.

Sprint 0 Ola C. End-to-end smoke that mirrors what the v4 orchestrator
(Sprint 1+) will do: register the model, iter applicable, predict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bip.models.base import PredictionRecord
from bip.models.mundial.model import DEFAULT_LOCK_PATH, MundialModel
from bip.models.registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / DEFAULT_LOCK_PATH


@pytest.fixture(scope="module")
def lock_data() -> dict:
    return json.loads(LOCK_PATH.read_text())


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    r = ModelRegistry()
    r.register(MundialModel(lock_path=LOCK_PATH, p_max_threshold=0.0))
    return r


def test_registry_contains_mundial(registry: ModelRegistry):
    assert "mundial_wc2026" in registry
    assert len(registry) == 1


def test_iter_applicable_routes_wc2026(registry: ModelRegistry, lock_data: dict):
    first = lock_data["fixtures"][0]
    fixture = {"competition": "WC2026", "match_id": first["match_id"]}
    applicable = list(registry.iter_applicable(fixture))
    assert [m.name for m in applicable] == ["mundial_wc2026"]


def test_iter_applicable_skips_non_wc(registry: ModelRegistry, lock_data: dict):
    first = lock_data["fixtures"][0]
    fixture = {"competition": "PL", "match_id": first["match_id"]}
    assert list(registry.iter_applicable(fixture)) == []


def test_end_to_end_smoke(registry: ModelRegistry, lock_data: dict):
    """Mirrors what the v4 orchestrator will do at runtime."""
    all_records: list[PredictionRecord] = []
    for f in lock_data["fixtures"]:
        fixture = {
            "competition": "WC2026",
            "match_id": f["match_id"],
        }
        for model in registry.iter_applicable(fixture):
            all_records.extend(model.predict(fixture))
    # With threshold 0.0 we expect exactly 72 records.
    assert len(all_records) == 72
    # All records have the contract fields set
    for rec in all_records:
        assert rec.source == "mundial"
        assert rec.competition == "WC2026"
        assert rec.payload["calibration_status"] == "below-gate"
