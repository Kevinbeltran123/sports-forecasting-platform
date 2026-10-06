"""Tests for the nightly cohort accountant scheduler wrapper.

Mirrors the refit_scheduler test patterns. Invariants:
- CohortJob.run counts success/failure correctly across multiple runs.
- A failure schedules a single-shot retry until max_retries exhausted.
- Success resets the failure counter.
- register_cohort_job adds exactly one cron-triggered job.
- The wrapper does not re-raise: scheduler stays alive across exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bip.evaluation.live.engine_v3.kill_criteria import (
    CohortMetrics,
    CohortStage,
    KillVerdict,
)
from bip.evaluation.live.engine_v3.runtime.cohort_accountant import (
    CohortEvalResult,
)
from bip.evaluation.live.engine_v3.runtime.cohort_scheduler import (
    CohortJob,
    register_cohort_job,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeScheduler:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def add_job(self, func, trigger=None, **kwargs):
        self.added.append({"func": func, "trigger": trigger, **kwargs})


def _make_eval_result(verdict: KillVerdict | None = None) -> CohortEvalResult:
    metrics = CohortMetrics(
        stage=CohortStage.A,
        n_settled=10,
        n_won=5,
        n_lost=5,
        n_void=0,
        roi_flat=0.0,
        roi_kelly=0.0,
        ood_denial_rate_50=0.0,
        drift_active_50=False,
        pipeline_error_count_100=0,
        shadow_log_failure_rate_100=0.0,
        market_concentration_top1_100=0.2,
    )
    return CohortEvalResult(
        verdict=verdict or KillVerdict.ok(),
        metrics=metrics,
        n_picks_total=10,
        n_outcomes_loaded=10,
        kill_switch_engaged=False,
        report_path=None,
        alert_sent=False,
    )


@pytest.mark.anyio("asyncio")
async def test_run_increments_success_on_clean_eval():
    sched = _FakeScheduler()

    async def _runner():
        return _make_eval_result()

    job = CohortJob(scheduler=sched, runner=_runner)
    await job.run()
    assert job.success_count == 1
    assert job.failure_count == 0
    assert sched.added == []


@pytest.mark.anyio("asyncio")
async def test_run_records_last_result_for_inspection():
    sched = _FakeScheduler()
    expected = _make_eval_result()

    async def _runner():
        return expected

    job = CohortJob(scheduler=sched, runner=_runner)
    await job.run()
    assert job.last_result is expected


@pytest.mark.anyio("asyncio")
async def test_run_records_hard_stop_verdict_in_log_via_success_count():
    """Even hard-stop verdicts count as a successful EVAL — the verdict
    itself is a runtime outcome, not a runner failure."""
    sched = _FakeScheduler()

    async def _runner():
        return _make_eval_result(
            KillVerdict.hard_stop("test_rule", "synthetic")
        )

    job = CohortJob(scheduler=sched, runner=_runner)
    await job.run()
    assert job.success_count == 1


@pytest.mark.anyio("asyncio")
async def test_exception_in_runner_counted_and_retried():
    sched = _FakeScheduler()

    async def _raises():
        raise RuntimeError("synthetic")

    job = CohortJob(scheduler=sched, runner=_raises, max_retries=3)
    await job.run()
    assert job.failure_count == 1
    assert job.error_count == 1
    assert len(sched.added) == 1
    assert sched.added[0]["trigger"] == "date"


@pytest.mark.anyio("asyncio")
async def test_gives_up_after_max_retries():
    sched = _FakeScheduler()

    async def _raises():
        raise RuntimeError("perma-fail")

    job = CohortJob(scheduler=sched, runner=_raises, max_retries=3)
    await job.run()
    await job.run()
    await job.run()
    # On the 3rd failure (counter==3, equal to max_retries) we stop scheduling
    # AND reset to 0 so the next cron firing starts fresh.
    assert job.failure_count == 0
    assert len(sched.added) == 2


@pytest.mark.anyio("asyncio")
async def test_success_resets_failure_counter():
    sched = _FakeScheduler()
    state = {"raises": True}

    async def _toggle():
        if state["raises"]:
            raise RuntimeError("once")
        return _make_eval_result()

    job = CohortJob(scheduler=sched, runner=_toggle)
    await job.run()
    assert job.failure_count == 1
    state["raises"] = False
    await job.run()
    assert job.failure_count == 0
    assert job.success_count == 1


@pytest.mark.anyio("asyncio")
async def test_run_does_not_raise_when_scheduler_add_job_fails():
    """Even if the retry scheduling fails, the job swallows."""
    class _BrokenScheduler:
        def add_job(self, *a, **kw):
            raise RuntimeError("scheduler down")

    async def _raises():
        raise RuntimeError("synthetic")

    job = CohortJob(scheduler=_BrokenScheduler(), runner=_raises, max_retries=3)
    # Should not raise out of run()
    await job.run()
    assert job.failure_count == 1


def test_register_cohort_job_adds_cron_job():
    sched = _FakeScheduler()
    job = register_cohort_job(sched, runner=lambda: _make_eval_result())
    assert isinstance(job, CohortJob)
    assert len(sched.added) == 1
    entry = sched.added[0]
    assert entry["id"] == "v3_nightly_cohort_eval"
    assert entry["replace_existing"] is True
    assert hasattr(entry["trigger"], "fields")


def test_register_cohort_job_default_hour_is_2am():
    sched = _FakeScheduler()
    register_cohort_job(sched, runner=lambda: _make_eval_result())
    trigger = sched.added[0]["trigger"]
    # CronTrigger fields include hour=2, minute=0 by default
    field_map = {f.name: str(f) for f in trigger.fields}
    assert "2" in field_map.get("hour", "")
    assert "0" in field_map.get("minute", "")
