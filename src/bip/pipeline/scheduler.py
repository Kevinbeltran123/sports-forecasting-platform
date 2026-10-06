"""PipelineScheduler — APScheduler 3.x async wiring for the v4 workers.

Sprint 3 Ola A. Registers cron/interval triggers for the 3 workers:

  - Orchestrator: daily CronTrigger (default 06:00 UTC) — fetches the
    day's fixtures and pushes PredictionRecord rows to predictions_raw.
  - DeliveryWorker: IntervalTrigger every 15 min (spec §6.2) — drains
    pending rows, applies Claude + Kelly + safety gates, marks
    shadow/sent/killed.
  - ClvWorker: IntervalTrigger every 60 min — measures CLV + outcome
    for sent rows past T+kickoff_delay.

CRITICAL: APScheduler 3.x ONLY. 4.x is alpha and
the API has breaking changes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

log = structlog.get_logger(__name__)


# Job IDs — stable for fire_now() lookups
JOB_ORCHESTRATOR = "v4_orchestrator_daily"
JOB_DELIVERY = "v4_delivery_worker_interval"
JOB_CLV = "v4_clv_worker_interval"


@dataclass
class SchedulerConfig:
    """Tunable schedule for the v4 stack."""

    orchestrator_cron_hour: int = 6        # 06:00 UTC
    orchestrator_cron_minute: int = 0
    delivery_interval_minutes: int = 15
    clv_interval_minutes: int = 60
    timezone: str = "UTC"


class PipelineScheduler:
    """Owns the AsyncIOScheduler + the 3 worker callbacks."""

    def __init__(
        self,
        *,
        scheduler: AsyncIOScheduler | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        self._scheduler = scheduler or AsyncIOScheduler(timezone=(config or SchedulerConfig()).timezone)
        self.config = config or SchedulerConfig()
        self._callbacks: dict[str, Callable[[], Awaitable[object]]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register_orchestrator_daily(
        self, callback: Callable[[], Awaitable[object]]
    ) -> None:
        """Cron trigger at SchedulerConfig.orchestrator_cron_{hour,minute}."""
        self._remove_if_exists(JOB_ORCHESTRATOR)
        self._scheduler.add_job(
            callback,
            trigger=CronTrigger(
                hour=self.config.orchestrator_cron_hour,
                minute=self.config.orchestrator_cron_minute,
            ),
            id=JOB_ORCHESTRATOR,
            replace_existing=True,
            misfire_grace_time=600,
            name="v4 Orchestrator (daily fixture push)",
        )
        self._callbacks[JOB_ORCHESTRATOR] = callback
        log.info(
            "scheduler_registered_orchestrator",
            hour=self.config.orchestrator_cron_hour,
            minute=self.config.orchestrator_cron_minute,
        )

    def register_delivery_worker_interval(
        self, callback: Callable[[], Awaitable[object]]
    ) -> None:
        """IntervalTrigger every SchedulerConfig.delivery_interval_minutes."""
        self._remove_if_exists(JOB_DELIVERY)
        self._scheduler.add_job(
            callback,
            trigger=IntervalTrigger(minutes=self.config.delivery_interval_minutes),
            id=JOB_DELIVERY,
            replace_existing=True,
            misfire_grace_time=120,
            name="v4 DeliveryWorker (15-min poll)",
        )
        self._callbacks[JOB_DELIVERY] = callback
        log.info(
            "scheduler_registered_delivery",
            interval_minutes=self.config.delivery_interval_minutes,
        )

    def register_clv_worker_interval(
        self, callback: Callable[[], Awaitable[object]]
    ) -> None:
        """IntervalTrigger every SchedulerConfig.clv_interval_minutes."""
        self._remove_if_exists(JOB_CLV)
        self._scheduler.add_job(
            callback,
            trigger=IntervalTrigger(minutes=self.config.clv_interval_minutes),
            id=JOB_CLV,
            replace_existing=True,
            misfire_grace_time=300,
            name="v4 ClvWorker (hourly)",
        )
        self._callbacks[JOB_CLV] = callback
        log.info(
            "scheduler_registered_clv",
            interval_minutes=self.config.clv_interval_minutes,
        )

    def _remove_if_exists(self, job_id: str) -> None:
        """Best-effort job removal — APScheduler 3.x does not honor
        replace_existing=True before .start(), so we drop the existing
        job explicitly to keep re-register idempotent in both states."""
        try:
            self._scheduler.remove_job(job_id)
        except Exception:  # noqa: BLE001 — JobLookupError; nothing to remove
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._scheduler.start()
        log.info("scheduler_started", n_jobs=len(self._scheduler.get_jobs()))

    def shutdown(self, *, wait: bool = True) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            log.info("scheduler_shutdown")

    @property
    def running(self) -> bool:
        return self._scheduler.running

    @property
    def jobs(self) -> list[dict]:
        """Snapshot of registered jobs (id, name, next_run_time).

        APScheduler 3.x exposes ``next_run_time`` on Job only after the
        scheduler is started; before start the attribute is absent. We
        use ``getattr`` with default so the property is safe to call in
        either state.
        """
        out: list[dict] = []
        for j in self._scheduler.get_jobs():
            nrt = getattr(j, "next_run_time", None)
            out.append(
                {
                    "id": j.id,
                    "name": j.name,
                    "next_run_time": nrt.isoformat() if nrt else None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Testing entrypoint
    # ------------------------------------------------------------------

    async def fire_now(self, job_id: str) -> object:
        """Run a registered callback immediately (bypasses the scheduler).

        Used by smoke tests + dry-run mode in factory.build_pipeline.
        Returns whatever the callback returns (typically a Run Summary).
        """
        if job_id not in self._callbacks:
            raise KeyError(f"Unknown job_id: {job_id}")
        log.info("scheduler_fire_now", job_id=job_id)
        try:
            return await self._callbacks[job_id]()
        except Exception as exc:  # noqa: BLE001
            log.warning("scheduler_callback_error", job_id=job_id, error=str(exc))
            raise


__all__ = [
    "JOB_CLV",
    "JOB_DELIVERY",
    "JOB_ORCHESTRATOR",
    "PipelineScheduler",
    "SchedulerConfig",
]
