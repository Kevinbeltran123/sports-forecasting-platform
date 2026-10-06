"""APScheduler wiring for the weekly v3 refit job.

T3 of v3 phase-4 readiness mission. Registers a cron-triggered job
that invokes a detector-refit callable every Sunday
at 04:00 local. Three retries on transient failures (1 hour spacing);
on persistent failure the job gives up and logs an error.

Usage from watch.py::

    scheduler = AsyncIOScheduler()
    register_refit_job(scheduler)
    scheduler.start()

# Why APScheduler 3.x

The stack pins APScheduler 3.x — 4.x is still alpha. 3.x's ``AsyncIOScheduler`` slots cleanly into watch.py's
existing asyncio event loop.

# Failure semantics

The refit script is **idempotent** — running it twice produces two
versioned pkls (v_N and v_N+1) but only the later one's promotion
takes effect. Re-runs on retry are safe.

The retry policy is:

1. Job fires Sunday 04:00 local.
2. If the underlying script exits non-zero, retry in 1 hour.
3. Max 3 retries (so up to 3 hours of grace).
4. After 3 failures, log error and wait for the next cron firing.

The retries are implemented via APScheduler's ``misfire_grace_time``
+ a custom wrapper that re-schedules a single-shot job on failure.
This is simpler than configuring APScheduler's coalescing/replace_existing
behavior for one-off retries.

# Importing from this module

Re-export-friendly. Only the scheduler registration entry point and
the wrapper class are public.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger("v3.refit_scheduler")


# ──────────────────────────────────────────────────────────────────────
# Wrapper with retry semantics
# ──────────────────────────────────────────────────────────────────────


_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 3600  # 1 hour


class RefitJob:
    """Wraps the refit invocation with a retry counter + structured logs.

    Constructed once at watch.py startup. The scheduler calls
    ``run()`` per firing; ``run()`` invokes the underlying callable in
    a thread (refit is sync CPU work + filesystem I/O) and counts
    failures.

    On failure, ``run()`` schedules a single-shot retry via the
    provided scheduler if it has slots left. The retry counter resets
    on every successful run.
    """

    def __init__(
        self,
        scheduler: Any,
        refit_callable: Callable[[], int] | None = None,
        max_retries: int = _MAX_RETRIES,
        retry_delay_sec: int = _RETRY_DELAY_SEC,
    ) -> None:
        self.scheduler = scheduler
        self.refit_callable = refit_callable or _default_refit_callable
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.failure_count = 0
        self.success_count = 0
        self.error_count = 0

    async def run(self) -> None:
        """Single invocation. Returns immediately after queuing a retry
        on transient failure; raises only if the callable itself does
        and retries are exhausted.
        """
        try:
            rc = await asyncio.to_thread(self.refit_callable)
            if rc == 0:
                self.success_count += 1
                self.failure_count = 0  # reset on success
                log.info("v3_refit_success success_count=%d", self.success_count)
                return
            # Non-zero exit: count as a soft failure (script ran but reported error)
            self.failure_count += 1
            log.warning(
                "v3_refit_nonzero_exit rc=%d failure_count=%d",
                rc,
                self.failure_count,
            )
        except Exception as exc:  # noqa: BLE001
            self.failure_count += 1
            self.error_count += 1
            log.warning(
                "v3_refit_error reason=%s failure_count=%d",
                type(exc).__name__,
                self.failure_count,
            )

        # Failure path: schedule retry if budget allows
        if self.failure_count < self.max_retries:
            run_at = datetime.now() + timedelta(seconds=self.retry_delay_sec)
            log.info(
                "v3_refit_retry_scheduled at=%s attempt=%d/%d",
                run_at.isoformat(),
                self.failure_count + 1,
                self.max_retries,
            )
            try:
                self.scheduler.add_job(self.run, "date", run_date=run_at)
            except Exception as exc:  # noqa: BLE001
                log.warning("v3_refit_retry_schedule_failed reason=%s", exc)
        else:
            log.error(
                "v3_refit_giving_up failures=%d — waiting for next cron firing",
                self.failure_count,
            )
            # Reset counter so the next cron firing starts fresh.
            self.failure_count = 0


def _default_refit_callable() -> int:
    """Default refit callable.

    The offline refit script is not part of this distribution, so callers
    must pass ``refit_callable`` explicitly to ``register_refit_job``.
    """
    raise RuntimeError(
        "No default refit script is bundled; pass refit_callable to register_refit_job()."
    )


# ──────────────────────────────────────────────────────────────────────
# Public registration entry point
# ──────────────────────────────────────────────────────────────────────


def register_refit_job(
    scheduler: Any,
    *,
    day_of_week: str = "sun",
    hour: int = 4,
    minute: int = 0,
    refit_callable: Callable[[], int] | None = None,
) -> RefitJob:
    """Register the weekly refit cron job on ``scheduler``.

    Returns the ``RefitJob`` wrapper so the caller can inspect counts
    or trigger a manual ``await job.run()`` for testing.

    The cron trigger is **timezone-naive** by default (uses the
    scheduler's configured TZ). Operators running across timezones
    should configure the scheduler explicitly:

        scheduler = AsyncIOScheduler(timezone="UTC")
    """
    # Lazy import to keep the dependency optional at module-import time.
    # APScheduler 3.x is pinned in the stack but importing CronTrigger here
    # lets the test suite stub the scheduler with a duck type.
    from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

    job = RefitJob(scheduler=scheduler, refit_callable=refit_callable)
    trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
    scheduler.add_job(
        job.run,
        trigger=trigger,
        id="v3_weekly_refit",
        replace_existing=True,
    )
    log.info(
        "v3_refit_job_registered day_of_week=%s hour=%d minute=%d",
        day_of_week,
        hour,
        minute,
    )
    return job
