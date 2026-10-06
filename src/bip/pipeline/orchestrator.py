"""Orchestrator — iterate ModelRegistry over fixtures, persist to predictions_raw.

Sprint 2 Ola A. The orchestrator is the v4 producer side: it takes an
iterable of fixture dicts (hydrated upstream by the data layer) plus a
ModelRegistry, and for every (model, fixture) pair where `can_handle`
is True it calls `model.predict()` and INSERTs the resulting
PredictionRecord rows into `predictions_raw`.

The orchestrator does NOT enrich predictions with Claude, EV gating, or
Kelly sizing — those live in DeliveryWorker (Ola B), which consumes
from `predictions_raw` independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import structlog

from bip.models.base import PredictionRecord
from bip.models.registry import ModelRegistry
from bip.pipeline.protocols import SupabaseClientProtocol

log = structlog.get_logger(__name__)

# Must match the table name in supabase/migrations/20260524000000
PREDICTIONS_RAW_TABLE = "predictions_raw"


@dataclass
class OrchestratorRunSummary:
    """Result of one Orchestrator.run_for_fixtures call.

    Counts are per-call (not cumulative). `inserted_ids` carries the
    Supabase-assigned UUIDs when available — empty when the client mock
    returns nothing.
    """

    n_fixtures: int = 0
    n_models_invoked: int = 0
    n_predictions_emitted: int = 0
    n_inserted: int = 0
    n_errors: int = 0
    inserted_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Orchestrator:
    """Drives the ModelRegistry over a set of fixtures and persists predictions.

    Typical usage (Sprint 2+):

        registry = ModelRegistry()
        registry.register(MundialModel(lock_path=...))
        registry.register(LigasModel())
        client = get_supabase_client(settings)

        orch = Orchestrator(registry=registry, supabase_client=client)
        summary = await orch.run_for_fixtures(today_fixtures)
    """

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        supabase_client: SupabaseClientProtocol,
        table_name: str = PREDICTIONS_RAW_TABLE,
    ) -> None:
        self._registry = registry
        self._client = supabase_client
        self._table_name = table_name

    async def run_for_fixtures(
        self, fixtures: Iterable[dict[str, Any]]
    ) -> OrchestratorRunSummary:
        """Iterate fixtures × applicable models, persist predictions.

        Note: the call is synchronous-style internally (supabase-py is
        sync). The async signature is kept so callers can `await` and
        future async upgrades don't change the public contract.
        """
        summary = OrchestratorRunSummary()
        fixture_list = list(fixtures)
        summary.n_fixtures = len(fixture_list)

        for fixture in fixture_list:
            fid = fixture.get("fixture_id") or fixture.get("match_id")
            for model in self._registry.iter_applicable(fixture):
                summary.n_models_invoked += 1
                try:
                    records = model.predict(fixture)
                except Exception as exc:  # noqa: BLE001
                    summary.n_errors += 1
                    summary.errors.append(
                        f"{model.name} predict failed for {fid}: {exc}"
                    )
                    log.warning(
                        "orchestrator_predict_error",
                        model=model.name,
                        fixture_id=fid,
                        error=str(exc),
                    )
                    continue
                summary.n_predictions_emitted += len(records)
                for rec in records:
                    inserted = self._insert(rec)
                    if inserted is None:
                        summary.n_errors += 1
                        continue
                    summary.n_inserted += 1
                    if isinstance(inserted, dict) and inserted.get("id"):
                        summary.inserted_ids.append(str(inserted["id"]))

        log.info(
            "orchestrator_run_complete",
            n_fixtures=summary.n_fixtures,
            n_predictions=summary.n_predictions_emitted,
            n_inserted=summary.n_inserted,
            n_errors=summary.n_errors,
        )
        return summary

    def _insert(self, record: PredictionRecord) -> dict | None:
        """INSERT one PredictionRecord into predictions_raw via supabase-py."""
        payload = record.to_supabase_dict()
        try:
            resp = (
                self._client.table(self._table_name).insert(payload).execute()
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "orchestrator_insert_error",
                fixture_id=record.fixture_id,
                market=record.market,
                selection=record.selection,
                error=str(exc),
            )
            return None
        data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
        if not data:
            return {"status": "inserted"}
        return data[0] if isinstance(data, list) and data else data


__all__ = ["Orchestrator", "OrchestratorRunSummary"]
