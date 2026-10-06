"""Orchestrator tests — Sprint 2 Ola A.

Mocks Supabase via FakeSupabaseClient from conftest. Verifies:
  - registry.iter_applicable routes correctly per fixture
  - PredictionRecord.to_supabase_dict payloads are inserted
  - Errors in one model do not block others
  - run summary counts are accurate
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bip.models.registry import ModelRegistry
from bip.pipeline.orchestrator import (
    PREDICTIONS_RAW_TABLE,
    Orchestrator,
    OrchestratorRunSummary,
)
from tests.pipeline.conftest import BrokenModel, StubLigasModel, StubMundialModel


def _fixture(**overrides):
    base = {
        "fixture_id": "fx-001",
        "competition": "PL",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_datetime": datetime(2026, 6, 1, 19, 0, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
class TestOrchestrator:
    async def test_empty_registry_empty_fixtures_returns_zero_counts(self, fake_client):
        orch = Orchestrator(registry=ModelRegistry(), supabase_client=fake_client)
        summary = await orch.run_for_fixtures([])
        assert isinstance(summary, OrchestratorRunSummary)
        assert summary.n_fixtures == 0
        assert summary.n_models_invoked == 0
        assert summary.n_inserted == 0

    async def test_single_model_inserts_per_fixture(self, fake_client):
        registry = ModelRegistry()
        registry.register(StubLigasModel())
        orch = Orchestrator(registry=registry, supabase_client=fake_client)

        fixtures = [_fixture(fixture_id="fx-1"), _fixture(fixture_id="fx-2")]
        summary = await orch.run_for_fixtures(fixtures)

        assert summary.n_fixtures == 2
        assert summary.n_models_invoked == 2
        assert summary.n_predictions_emitted == 2
        assert summary.n_inserted == 2

        # Two inserts on predictions_raw
        inserts = fake_client.tables[PREDICTIONS_RAW_TABLE].inserts
        assert len(inserts) == 2
        assert all(row["source"] == "ligas" for row in inserts)
        # IDs were returned by the fake client
        assert len(summary.inserted_ids) == 2

    async def test_can_handle_routes_correctly(self, fake_client):
        registry = ModelRegistry()
        registry.register(StubLigasModel())
        registry.register(StubMundialModel())
        orch = Orchestrator(registry=registry, supabase_client=fake_client)

        fixtures = [
            _fixture(fixture_id="pl-1", competition="PL"),
            _fixture(fixture_id="wc-1", competition="WC2026"),
            _fixture(fixture_id="mls-1", competition="MLS"),  # no model handles
        ]
        summary = await orch.run_for_fixtures(fixtures)

        assert summary.n_fixtures == 3
        # PL fixture → ligas only; WC → mundial only; MLS → none
        assert summary.n_models_invoked == 2
        assert summary.n_predictions_emitted == 2

        sources = [row["source"] for row in fake_client.tables[PREDICTIONS_RAW_TABLE].inserts]
        assert sorted(sources) == ["ligas", "mundial"]

    async def test_multiple_records_per_fixture(self, fake_client):
        registry = ModelRegistry()
        registry.register(StubLigasModel(emit_per_fixture=3))
        orch = Orchestrator(registry=registry, supabase_client=fake_client)

        summary = await orch.run_for_fixtures([_fixture(fixture_id="fx-1")])
        assert summary.n_predictions_emitted == 3
        assert summary.n_inserted == 3
        selections = [
            row["selection"]
            for row in fake_client.tables[PREDICTIONS_RAW_TABLE].inserts
        ]
        assert sorted(selections) == ["away", "draw", "home"]

    async def test_broken_model_does_not_block_others(self, fake_client):
        registry = ModelRegistry()
        registry.register(BrokenModel())
        registry.register(StubLigasModel())
        orch = Orchestrator(registry=registry, supabase_client=fake_client)

        summary = await orch.run_for_fixtures([_fixture(fixture_id="fx-1")])
        # Broken model raised — recorded as error; ligas still inserted
        assert summary.n_errors == 1
        assert summary.n_inserted == 1

    async def test_supabase_insert_error_is_recorded(self, fake_client, monkeypatch):
        registry = ModelRegistry()
        registry.register(StubLigasModel())
        orch = Orchestrator(registry=registry, supabase_client=fake_client)

        # Patch the table query's insert.execute to raise
        def explode(self):
            raise RuntimeError("Supabase down")

        from tests.pipeline.conftest import FakeQuery
        monkeypatch.setattr(FakeQuery, "execute", explode)

        summary = await orch.run_for_fixtures([_fixture(fixture_id="fx-1")])
        assert summary.n_errors == 1
        assert summary.n_inserted == 0

    async def test_payload_shape_matches_schema(self, fake_client):
        registry = ModelRegistry()
        registry.register(StubLigasModel())
        orch = Orchestrator(registry=registry, supabase_client=fake_client)

        await orch.run_for_fixtures([_fixture(fixture_id="fx-1")])
        row = fake_client.tables[PREDICTIONS_RAW_TABLE].inserts[0]
        for key in [
            "source", "fixture_id", "competition", "home_team", "away_team",
            "match_datetime", "market", "selection", "p_model", "ev",
            "odds_at_pick", "payload", "model_version", "status", "created_at",
        ]:
            assert key in row, f"Missing key: {key}"
        # ISO datetime serialization
        assert row["match_datetime"].endswith("+00:00")
