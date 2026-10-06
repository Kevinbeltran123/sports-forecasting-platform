"""PipelineScheduler tests — Sprint 3 Ola A.

Uses real AsyncIOScheduler instances (no jobstores, in-memory MemoryJobStore
default). Tests do NOT call .start() so no event loop pumping is needed
for the registration checks; the fire_now() path drives callbacks
directly via asyncio.run.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from bip.pipeline.scheduler import (
    JOB_CLV,
    JOB_DELIVERY,
    JOB_ORCHESTRATOR,
    PipelineScheduler,
    SchedulerConfig,
)


def _async_noop(return_value: object | None = None):
    """Build an async callable returning return_value (default None)."""
    return AsyncMock(return_value=return_value)


class TestRegistration:
    def test_register_all_three_jobs(self):
        sched = PipelineScheduler()
        sched.register_orchestrator_daily(_async_noop())
        sched.register_delivery_worker_interval(_async_noop())
        sched.register_clv_worker_interval(_async_noop())

        ids = {j["id"] for j in sched.jobs}
        assert ids == {JOB_ORCHESTRATOR, JOB_DELIVERY, JOB_CLV}

    def test_replace_existing_on_reregister(self):
        sched = PipelineScheduler()
        first = _async_noop()
        second = _async_noop()
        sched.register_orchestrator_daily(first)
        sched.register_orchestrator_daily(second)

        # Only one job after re-register
        ids = [j["id"] for j in sched.jobs]
        assert ids.count(JOB_ORCHESTRATOR) == 1

    def test_custom_config_applied(self):
        config = SchedulerConfig(
            orchestrator_cron_hour=12,
            orchestrator_cron_minute=30,
            delivery_interval_minutes=5,
            clv_interval_minutes=30,
        )
        sched = PipelineScheduler(config=config)
        assert sched.config.orchestrator_cron_hour == 12
        assert sched.config.delivery_interval_minutes == 5

    def test_jobs_have_names(self):
        sched = PipelineScheduler()
        sched.register_orchestrator_daily(_async_noop())
        sched.register_delivery_worker_interval(_async_noop())
        sched.register_clv_worker_interval(_async_noop())
        names = {j["name"] for j in sched.jobs}
        assert any("Orchestrator" in n for n in names)
        assert any("DeliveryWorker" in n for n in names)
        assert any("ClvWorker" in n for n in names)


@pytest.mark.asyncio
class TestFireNow:
    async def test_fire_now_invokes_callback(self):
        sched = PipelineScheduler()
        cb = _async_noop(return_value="done")
        sched.register_orchestrator_daily(cb)

        result = await sched.fire_now(JOB_ORCHESTRATOR)
        assert result == "done"
        cb.assert_awaited_once()

    async def test_fire_now_unknown_job_raises(self):
        sched = PipelineScheduler()
        with pytest.raises(KeyError):
            await sched.fire_now("unknown-job")

    async def test_callback_exception_propagates(self):
        sched = PipelineScheduler()
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        sched.register_delivery_worker_interval(cb)
        with pytest.raises(RuntimeError, match="boom"):
            await sched.fire_now(JOB_DELIVERY)


@pytest.mark.asyncio
class TestLifecycle:
    async def test_start_and_shutdown_idempotent(self):
        sched = PipelineScheduler()
        sched.register_orchestrator_daily(_async_noop())
        # AsyncIOScheduler.start() reads asyncio.get_running_loop(); this
        # test is async so a loop is already running.
        sched.start()
        assert sched.running
        sched.shutdown(wait=False)
        # AsyncIOScheduler 3.x flips `running` on the next loop tick, not
        # synchronously inside shutdown().
        await asyncio.sleep(0)
        assert not sched.running
        # shutdown again is a no-op
        sched.shutdown()
