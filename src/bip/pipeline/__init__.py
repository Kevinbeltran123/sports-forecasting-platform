"""bip.pipeline — v4 orchestrator + delivery + CLV workers.

Sprint 2 of the v4 refactor. See plan:
  internal design notes (not published)

This package is the v4 replacement for the legacy src/bip/production/
orchestration code. It works exclusively over the predictions_raw
table + BaseModel contract. The old PickEngine / production/ stack
keeps running unchanged until v4 passes its gate.
"""

from bip.pipeline.clv_worker import (
    ClvRunSummary,
    ClvWorker,
    OddsClientProtocol,
    ResultClientProtocol,
)
from bip.pipeline.delivery_worker import (
    DeliveryRunSummary,
    DeliveryWorker,
)
from bip.pipeline.factory import (
    InMemorySupabaseClient,
    NoOpSender,
    PipelineHandle,
    StubApiFootballClient,
    StubValidator,
    build_pipeline,
)
from bip.pipeline.fixture_hydrator import (
    FixtureHydrator,
    HydratorSummary,
)
from bip.pipeline.orchestrator import (
    Orchestrator,
    OrchestratorRunSummary,
)
from bip.pipeline.protocols import (
    ClaudeValidatorProtocol,
    SupabaseClientProtocol,
    TelegramSenderProtocol,
)
from bip.pipeline.scheduler import (
    PipelineScheduler,
    SchedulerConfig,
)

__all__ = [
    "ClaudeValidatorProtocol",
    "ClvRunSummary",
    "ClvWorker",
    "DeliveryRunSummary",
    "DeliveryWorker",
    "FixtureHydrator",
    "HydratorSummary",
    "InMemorySupabaseClient",
    "NoOpSender",
    "OddsClientProtocol",
    "Orchestrator",
    "OrchestratorRunSummary",
    "PipelineHandle",
    "PipelineScheduler",
    "ResultClientProtocol",
    "SchedulerConfig",
    "StubApiFootballClient",
    "StubValidator",
    "SupabaseClientProtocol",
    "TelegramSenderProtocol",
    "build_pipeline",
]
