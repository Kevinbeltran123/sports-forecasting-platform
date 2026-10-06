"""factory.build_pipeline — assemble the v4 stack from settings.

Sprint 3 Ola C. Returns a `PipelineHandle` carrying the scheduler +
workers + orchestrator + registry, with sensible stub fallbacks in
dry-run mode.

  - dry_run=True (default in tests + smoke):
      * Supabase client → InMemorySupabaseClient (writes go to a dict)
      * Claude validator → StubValidator (always CONFIRM, modifier=0)
      * Telegram sender → NoOpSender (records calls, doesn't send)
      * ApiFootballClient → StubApiFootballClient (empty fixtures)
    All deps work without env vars; no network IO.

  - dry_run=False (future operational mode):
      Real ClaudeValidator + TelegramSender + supabase-py Client +
      ApiFootballClient instantiated from settings. Only constructed
      when the relevant credentials are present.

The factory is the SINGLE place where the v4 stack is wired together;
__main__.py just calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from bip.models.ligas import LigasModel
from bip.models.mundial import DEFAULT_LOCK_PATH, MundialModel
from bip.models.registry import ModelRegistry
from bip.pipeline.clv_worker import ClvWorker
from bip.pipeline.delivery_worker import DeliveryWorker
from bip.pipeline.fixture_hydrator import FixtureHydrator
from bip.pipeline.orchestrator import Orchestrator, PREDICTIONS_RAW_TABLE
from bip.pipeline.scheduler import PipelineScheduler, SchedulerConfig

log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Stub implementations for dry-run mode
# ──────────────────────────────────────────────────────────────────────


class InMemorySupabaseClient:
    """Trivial in-memory supabase-py stand-in for dry-run + tests."""

    def __init__(self) -> None:
        self._rows: dict[str, list[dict]] = {}
        self.inserts: list[tuple[str, dict]] = []
        self.updates: list[tuple[str, dict, dict]] = []

    def table(self, name: str) -> "_InMemoryQuery":
        return _InMemoryQuery(self, name)


class _InMemoryQuery:
    def __init__(self, parent: InMemorySupabaseClient, table_name: str) -> None:
        self.parent = parent
        self.table_name = table_name
        self._pending_filter: dict[str, Any] = {}
        self._pending_insert: dict | None = None
        self._pending_update: dict | None = None
        self._mode: str | None = None  # 'insert' | 'update' | 'select'

    def insert(self, payload: dict) -> "_InMemoryQuery":
        self._mode = "insert"
        self._pending_insert = payload
        return self

    def update(self, payload: dict) -> "_InMemoryQuery":
        self._mode = "update"
        self._pending_update = payload
        return self

    def select(self, _cols: str = "*") -> "_InMemoryQuery":
        self._mode = "select"
        return self

    def eq(self, col: str, val: Any) -> "_InMemoryQuery":
        self._pending_filter[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_InMemoryQuery":
        self._pending_filter[f">={col}"] = val
        return self

    def lte(self, col: str, val: Any) -> "_InMemoryQuery":
        self._pending_filter[f"<={col}"] = val
        return self

    def lt(self, col: str, val: Any) -> "_InMemoryQuery":
        self._pending_filter[f"<{col}"] = val
        return self

    def is_(self, col: str, val: Any) -> "_InMemoryQuery":
        self._pending_filter[f"is:{col}"] = val
        return self

    def in_(self, col: str, vals: list) -> "_InMemoryQuery":
        self._pending_filter[f"in:{col}"] = vals
        return self

    def execute(self) -> Any:
        rows = self.parent._rows.setdefault(self.table_name, [])

        class _Resp:
            def __init__(self, data: Any) -> None:
                self.data = data

        if self._mode == "insert" and self._pending_insert is not None:
            new_row = {**self._pending_insert, "id": f"uuid-{len(rows) + 1}"}
            rows.append(new_row)
            self.parent.inserts.append((self.table_name, dict(new_row)))
            return _Resp([new_row])
        if self._mode == "update" and self._pending_update is not None:
            for r in rows:
                if r.get("id") == self._pending_filter.get("id"):
                    r.update(self._pending_update)
            self.parent.updates.append((self.table_name, dict(self._pending_filter), dict(self._pending_update)))
            return _Resp([{"status": "ok"}])
        if self._mode == "select":
            # We don't implement true filter eval — return all rows.
            return _Resp(list(rows))
        return _Resp([])


class StubValidator:
    """Always CONFIRM with confidence_modifier=0 — used in dry-run."""

    async def validate(self, pick_summary: str, curated_signals: str) -> Any:  # noqa: ARG002
        return _StubVerdict("CONFIRM", 0.0)


@dataclass
class _StubVerdict:
    verdict: str
    confidence_modifier: float


class NoOpSender:
    """Records send_pick calls without dispatching anywhere."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_pick(self, *, text: str) -> None:
        self.sent.append(text)


class StubApiFootballClient:
    """Empty get_fixtures — dry-run does not pull real schedules."""

    async def get_fixtures(self, league_id: int, date: str) -> dict:  # noqa: ARG002
        return {"response": []}


# ──────────────────────────────────────────────────────────────────────
# PipelineHandle + build_pipeline
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PipelineHandle:
    """Bundle of constructed components — returned by build_pipeline."""

    registry: ModelRegistry
    orchestrator: Orchestrator
    delivery_worker: DeliveryWorker
    clv_worker: ClvWorker
    hydrator: FixtureHydrator
    scheduler: PipelineScheduler
    # Dry-run extras (None in live mode):
    in_memory_client: InMemorySupabaseClient | None = None
    sender: Any = None  # NoOpSender or real TelegramSender
    api_client: Any = None  # StubApiFootballClient or real
    dry_run: bool = True


def build_pipeline(
    *,
    dry_run: bool = True,
    lock_path: Any = DEFAULT_LOCK_PATH,
    include_ligas: bool = True,
    include_mundial: bool = True,
    scheduler_config: SchedulerConfig | None = None,
) -> PipelineHandle:
    """Construct and return a wired v4 PipelineHandle.

    dry_run=True (default): uses InMemorySupabaseClient + StubValidator +
    NoOpSender + StubApiFootballClient. No env vars or credentials are
    read; the whole stack can be exercised with `python -m bip.pipeline
    --dry-run` without any external dependency.

    dry_run=False: real wiring is INTENTIONALLY a single-line swap site —
    construction of ClaudeValidator + TelegramSender + supabase-py
    Client + ApiFootballClient lives here when the data blocker is
    lifted. Until then the function raises NotImplementedError so the
    factory can't accidentally be used in live mode without an explicit
    follow-up change.
    """
    if not dry_run:
        raise NotImplementedError(
            "Live wiring is intentionally deferred — credentials + data "
            "are blocked until the operator unblocks them. See plan "
            "Sprint 3 Ola C 'Fuera de scope'."
        )

    in_mem_client = InMemorySupabaseClient()
    validator = StubValidator()
    sender = NoOpSender()
    api_client = StubApiFootballClient()

    # Build registry with the two v4 models
    registry = ModelRegistry()
    if include_mundial:
        registry.register(MundialModel(lock_path=lock_path, p_max_threshold=0.0))
    if include_ligas:
        registry.register(LigasModel())

    orchestrator = Orchestrator(registry=registry, supabase_client=in_mem_client)
    delivery_worker = DeliveryWorker(
        supabase_client=in_mem_client,
        validator=validator,
        sender=sender,
        shadow_mode=True,
    )
    clv_worker = ClvWorker(supabase_client=in_mem_client)
    hydrator = FixtureHydrator(api_client=api_client)

    scheduler = PipelineScheduler(config=scheduler_config or SchedulerConfig())

    log.info(
        "pipeline_built",
        dry_run=dry_run,
        n_models=len(registry),
        include_ligas=include_ligas,
        include_mundial=include_mundial,
    )
    return PipelineHandle(
        registry=registry,
        orchestrator=orchestrator,
        delivery_worker=delivery_worker,
        clv_worker=clv_worker,
        hydrator=hydrator,
        scheduler=scheduler,
        in_memory_client=in_mem_client,
        sender=sender,
        api_client=api_client,
        dry_run=True,
    )


__all__ = [
    "InMemorySupabaseClient",
    "NoOpSender",
    "PipelineHandle",
    "StubApiFootballClient",
    "StubValidator",
    "build_pipeline",
]
