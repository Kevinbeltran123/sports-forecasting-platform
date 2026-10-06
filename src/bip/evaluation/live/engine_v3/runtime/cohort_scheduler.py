"""APScheduler wiring for the nightly cohort accountant job.

Sibling to ``refit_scheduler.py`` and built on the same template:
APScheduler 3.x ``AsyncIOScheduler``, retry on transient failure with
exponential backoff capped at 3 attempts, structured logging.

Why a SEPARATE scheduler module instead of sharing the refit one:

- The nightly cadence (every night, e.g. 02:00 local) differs from the
  weekly refit (Sunday 04:00). Different triggers, different jobs.
- Failure semantics differ. A failed refit means stale detector pkls
  for one week (acceptable). A failed cohort run means hard stops are
  not enforced for one night — more urgent. The retry budget here is
  shorter (15 min spacing instead of 1 hour) and there are no silent
  retries: every failure logs at WARNING, every exhaustion at ERROR.
- Telegram alerting is wired here. The accountant takes an optional
  ``AlertSink`` — the scheduler is responsible for resolving the bot
  reference (lazy, from environment) so the registration call stays
  small and testable.

Usage from watch.py::

    scheduler = AsyncIOScheduler()
    register_cohort_job(scheduler, alert_sink=telegram_bot)
    scheduler.start()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from bip.evaluation.live.engine_v3.runtime.cohort_accountant import (
    AlertSink,
    CohortEvalResult,
    run_cohort_eval,
)

log = logging.getLogger("v3.cohort_scheduler")


_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 900  # 15 minutes


class CohortJob:
    """Wraps the cohort run with retry + structured logs.

    Constructed once at watch.py startup. The scheduler calls ``run()``
    per firing; ``run()`` invokes ``run_cohort_eval`` and counts
    failures. A failed run schedules a single-shot retry via the
    provided scheduler if budget remains. Retry counter resets on
    success.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        alert_sink: AlertSink | None = None,
        runner: Callable[[], Any] | None = None,
        max_retries: int = _MAX_RETRIES,
        retry_delay_sec: int = _RETRY_DELAY_SEC,
    ) -> None:
        self.scheduler = scheduler
        self.alert_sink = alert_sink
        self.runner = runner or self._default_runner
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.failure_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_result: CohortEvalResult | None = None

    async def _default_runner(self) -> CohortEvalResult:
        return await run_cohort_eval(alert_sink=self.alert_sink)

    async def run(self) -> None:
        """Single firing. Schedules a retry on failure; raises only if
        the runner does and retries are exhausted (we still swallow
        and log so the scheduler doesn't die).
        """
        try:
            result = await self.runner()
            self.last_result = result
            self.success_count += 1
            self.failure_count = 0  # reset on success
            verdict_kind = (
                "hard_stop" if result.verdict.is_hard_stop
                else "soft_warn" if result.verdict.is_soft_warning
                else "greenlight" if result.verdict.is_greenlight_ready
                else "ok"
            )
            log.info(
                "v3_cohort_run_success verdict=%s n_settled=%d kill_switch=%s",
                verdict_kind,
                result.metrics.n_settled,
                result.kill_switch_engaged,
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.failure_count += 1
            self.error_count += 1
            log.warning(
                "v3_cohort_run_error reason=%s failure_count=%d",
                type(exc).__name__,
                self.failure_count,
            )

        if self.failure_count < self.max_retries:
            run_at = datetime.now() + timedelta(seconds=self.retry_delay_sec)
            log.info(
                "v3_cohort_retry_scheduled at=%s attempt=%d/%d",
                run_at.isoformat(),
                self.failure_count + 1,
                self.max_retries,
            )
            try:
                self.scheduler.add_job(self.run, "date", run_date=run_at)
            except Exception as exc:  # noqa: BLE001
                log.warning("v3_cohort_retry_schedule_failed reason=%s", exc)
        else:
            log.error(
                "v3_cohort_giving_up failures=%d — waiting for next cron firing",
                self.failure_count,
            )
            self.failure_count = 0


def register_cohort_job(
    scheduler: Any,
    *,
    hour: int = 2,
    minute: int = 0,
    alert_sink: AlertSink | None = None,
    runner: Callable[[], Any] | None = None,
) -> CohortJob:
    """Register the nightly cohort accountant job on ``scheduler``.

    Default fire time: 02:00 local. Late enough that the previous day's
    last in-play fixtures have settled and any grading process has had
    its window. Early enough that the operator wakes up to fresh
    verdicts.

    Returns the ``CohortJob`` wrapper so callers can inspect counts or
    trigger ``await job.run()`` for testing.
    """
    from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

    job = CohortJob(scheduler=scheduler, alert_sink=alert_sink, runner=runner)
    trigger = CronTrigger(hour=hour, minute=minute)
    scheduler.add_job(
        job.run,
        trigger=trigger,
        id="v3_nightly_cohort_eval",
        replace_existing=True,
    )
    log.info(
        "v3_cohort_job_registered hour=%d minute=%d alert_sink=%s",
        hour,
        minute,
        type(alert_sink).__name__ if alert_sink is not None else "None",
    )
    return job


__all__ = ["CohortJob", "register_cohort_job"]
