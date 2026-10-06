"""Factory + entry point tests — Sprint 3 Ola C.

Covers:
  - build_pipeline(dry_run=True) returns a fully-wired PipelineHandle
  - dry_run=False raises NotImplementedError (intentional — live wiring
    is deferred until data unblocks)
  - InMemorySupabaseClient stores inserts/updates
  - StubValidator + NoOpSender behave as expected
  - __main__.run_dry_smoke ends without exception and emits a summary
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from bip.pipeline.factory import (
    InMemorySupabaseClient,
    NoOpSender,
    PipelineHandle,
    StubApiFootballClient,
    StubValidator,
    build_pipeline,
)


class TestBuildPipeline:
    def test_dry_run_returns_handle(self):
        h = build_pipeline(dry_run=True)
        assert isinstance(h, PipelineHandle)
        assert h.dry_run is True
        assert h.registry is not None
        assert h.orchestrator is not None
        assert h.delivery_worker is not None
        assert h.clv_worker is not None
        assert h.hydrator is not None
        assert h.scheduler is not None
        assert isinstance(h.in_memory_client, InMemorySupabaseClient)
        assert isinstance(h.sender, NoOpSender)
        assert isinstance(h.api_client, StubApiFootballClient)

    def test_dry_run_registers_both_models(self):
        h = build_pipeline(dry_run=True)
        names = h.registry.names()
        assert "mundial_wc2026" in names
        assert "ligas_phase1" in names

    def test_exclude_mundial(self):
        h = build_pipeline(dry_run=True, include_mundial=False)
        assert "mundial_wc2026" not in h.registry.names()
        assert "ligas_phase1" in h.registry.names()

    def test_exclude_ligas(self):
        h = build_pipeline(dry_run=True, include_ligas=False)
        assert "ligas_phase1" not in h.registry.names()
        assert "mundial_wc2026" in h.registry.names()

    def test_live_mode_raises(self):
        with pytest.raises(NotImplementedError, match="Live wiring"):
            build_pipeline(dry_run=False)


@pytest.mark.asyncio
class TestInMemorySupabaseClient:
    async def test_insert_roundtrip(self):
        client = InMemorySupabaseClient()
        resp = client.table("predictions_raw").insert(
            {"source": "ligas", "fixture_id": "fx-1"}
        ).execute()
        assert resp.data
        assert resp.data[0]["id"].startswith("uuid-")
        assert len(client.inserts) == 1

    async def test_update_records_filter_and_payload(self):
        client = InMemorySupabaseClient()
        client.table("predictions_raw").insert({"source": "ligas"}).execute()
        client.table("predictions_raw").update({"status": "shadow"}).eq(
            "id", "uuid-1"
        ).execute()
        assert len(client.updates) == 1

    async def test_select_returns_rows(self):
        client = InMemorySupabaseClient()
        client.table("predictions_raw").insert({"source": "ligas"}).execute()
        rows = client.table("predictions_raw").select("*").execute().data
        assert len(rows) == 1


@pytest.mark.asyncio
class TestStubs:
    async def test_stub_validator_returns_confirm(self):
        v = StubValidator()
        result = await v.validate(pick_summary="x", curated_signals="y")
        assert result.verdict == "CONFIRM"
        assert result.confidence_modifier == 0.0

    async def test_noop_sender_records_sends(self):
        s = NoOpSender()
        await s.send_pick(text="hi")
        await s.send_pick(text="there")
        assert s.sent == ["hi", "there"]

    async def test_stub_api_football_returns_empty(self):
        c = StubApiFootballClient()
        resp = await c.get_fixtures(39, "2026-06-01")
        assert resp == {"response": []}


@pytest.mark.asyncio
class TestDryRunSmokeEntry:
    async def test_run_dry_smoke_completes(self):
        from bip.pipeline.__main__ import run_dry_smoke

        class Args:
            no_mundial = False
            no_ligas = False
            date = "2026-06-11"
            dry_run = True

        summary = await run_dry_smoke(Args())
        # Smoke runs all 3 worker cycles
        assert summary["dry_run"] is True
        assert summary["date"] == "2026-06-11"
        assert "v4_orchestrator_daily" in summary["cycles"]
        assert "v4_delivery_worker_interval" in summary["cycles"]
        assert "v4_clv_worker_interval" in summary["cycles"]
        # Mundial registered → orchestrator emits at least one prediction
        # (the synth_fixture inside run_dry_smoke triggers MundialModel)
        cycles = summary["cycles"]
        assert cycles["v4_orchestrator_daily"]["orchestrator"]["n_inserted"] >= 1
