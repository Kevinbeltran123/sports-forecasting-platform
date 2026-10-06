"""Tests for the APScheduler-backed weekly refit job wrapper.

T3 invariants:
- ``RefitJob.run`` counts success/failure correctly across multiple runs.
- A non-zero exit from the refit callable schedules a single-shot retry
  if attempts remain.
- After ``max_retries`` consecutive failures, the job stops scheduling
  retries and resets the counter for the next cron firing.
- A clean run resets ``failure_count`` to 0.
- ``register_refit_job`` calls ``scheduler.add_job`` exactly once with
  a cron trigger.
"""

from __future__ import annotations

import pytest

from bip.evaluation.live.engine_v3.runtime.refit_scheduler import (
    RefitJob,
    register_refit_job,
)


class _FakeScheduler:
    """Minimal scheduler stub. Records add_job calls for assertion."""

    def __init__(self) -> None:
        self.added: list[dict] = []

    def add_job(self, func, trigger=None, **kwargs):
        self.added.append({"func": func, "trigger": trigger, **kwargs})


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ──────────────────────────────────────────────────────────────────────
# RefitJob.run — success path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.anyio("asyncio")
async def test_run_increments_success_on_zero_exit():
    sched = _FakeScheduler()
    job = RefitJob(scheduler=sched, refit_callable=lambda: 0)
    await job.run()
    assert job.success_count == 1
    assert job.failure_count == 0
    # No retry scheduled on success
    assert sched.added == []


@pytest.mark.anyio("asyncio")
async def test_success_resets_failure_count_after_prior_failure():
    sched = _FakeScheduler()
    job = RefitJob(scheduler=sched, refit_callable=lambda: 0)
    job.failure_count = 2  # simulate prior failures
    await job.run()
    assert job.failure_count == 0
    assert job.success_count == 1


# ──────────────────────────────────────────────────────────────────────
# RefitJob.run — failure / retry behavior
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.anyio("asyncio")
async def test_nonzero_exit_schedules_retry():
    sched = _FakeScheduler()
    job = RefitJob(scheduler=sched, refit_callable=lambda: 1, max_retries=3)
    await job.run()
    assert job.failure_count == 1
    # One retry scheduled
    assert len(sched.added) == 1
    assert sched.added[0]["trigger"] == "date"


@pytest.mark.anyio("asyncio")
async def test_exception_in_callable_counted_and_retried():
    def _raises() -> int:
        raise RuntimeError("synthetic")

    sched = _FakeScheduler()
    job = RefitJob(scheduler=sched, refit_callable=_raises, max_retries=3)
    await job.run()
    assert job.failure_count == 1
    assert job.error_count == 1
    # Retry scheduled
    assert len(sched.added) == 1


@pytest.mark.anyio("asyncio")
async def test_gives_up_after_max_retries():
    sched = _FakeScheduler()
    job = RefitJob(scheduler=sched, refit_callable=lambda: 2, max_retries=3)
    # Three failures consume the budget
    await job.run()
    await job.run()
    await job.run()
    # On the 3rd failure (counter==3, equal to max_retries) we stop scheduling
    # AND reset to 0 for the next cron firing
    assert job.failure_count == 0
    # Retries were scheduled on the first two failures only
    assert len(sched.added) == 2


# ──────────────────────────────────────────────────────────────────────
# register_refit_job
# ──────────────────────────────────────────────────────────────────────


def test_register_refit_job_adds_cron_job():
    sched = _FakeScheduler()
    job = register_refit_job(sched, refit_callable=lambda: 0)
    assert isinstance(job, RefitJob)
    assert len(sched.added) == 1
    entry = sched.added[0]
    # CronTrigger is opaque but we can confirm replace_existing + id wired
    assert entry["id"] == "v3_weekly_refit"
    assert entry["replace_existing"] is True
    # Trigger has cron-like attributes
    assert hasattr(entry["trigger"], "fields")
