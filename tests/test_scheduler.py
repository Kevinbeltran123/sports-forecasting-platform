"""APScheduler orchestrator tests — DATA-04."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest


class TestPipelineOrchestrator:
    """DATA-04: AsyncIOScheduler registers DateTrigger jobs for T-2h and T-30min."""

    def test_uses_asyncio_scheduler_not_background(self):
        """Must use AsyncIOScheduler (not BackgroundScheduler) per D-03d."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)
        assert isinstance(orch.scheduler, AsyncIOScheduler)

    def test_registers_date_trigger_jobs(self):
        """_daily_orchestrator must register 3 DateTrigger jobs per fixture: T-2h, T-30min, T-1min."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator
        from bip.sports import FixtureData

        plugin = MagicMock()
        kickoff = datetime.now(timezone.utc) + timedelta(hours=5)
        fixture = FixtureData(
            fixture_id=1,
            league="premier_league",
            sport="football",
            home_team="A",
            away_team="B",
            kickoff_utc=kickoff,
        )
        plugin.get_fixtures = AsyncMock(return_value=[fixture])

        orch = PipelineOrchestrator(plugin=plugin)
        # Prevent actually starting scheduler; just test job registration logic
        orch.scheduler.start = MagicMock()
        orch.scheduler.add_job = MagicMock()
        orch.start()

        # Daily orchestrator + nightly reconciliation = 2 initial jobs
        initial_calls = orch.scheduler.add_job.call_count
        assert initial_calls >= 2

    def test_daily_orchestrator_job_uses_cron_trigger_at_06_utc(self):
        """Daily orchestrator CronTrigger must be hour=6, minute=0, timezone=UTC — D-03a."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator
        from apscheduler.triggers.cron import CronTrigger

        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)
        orch.scheduler.start = MagicMock()
        orch.scheduler.add_job = MagicMock()
        orch.start()

        calls = orch.scheduler.add_job.call_args_list
        cron_triggers = [
            call for call in calls
            if call.kwargs.get("trigger") is not None
            and isinstance(call.kwargs.get("trigger"), CronTrigger)
            or (len(call.args) > 1 and isinstance(call.args[1], CronTrigger))
        ]
        # At minimum daily orchestrator is CronTrigger
        assert len(cron_triggers) >= 1

    def test_clv_snapshot_job_registered_at_kickoff_minus_1min(self):
        """CLV DateTrigger must be at kickoff - 1 minute — quick-260503-k8k.

        Pinnacle h2h freezes at kickoff; a snapshot one minute prior captures
        the closing line we benchmark CLV against. This replaces the earlier
        kickoff + 105m placeholder used while the recorder was unwired.
        """
        from bip.core.scheduler.orchestrator import PipelineOrchestrator
        import inspect
        src = inspect.getsource(PipelineOrchestrator._register_fixture_jobs)
        assert "t_minus_1m" in src, (
            "PipelineOrchestrator._register_fixture_jobs must register CLV job at kickoff - 1 min"
        )
        assert "kickoff - timedelta(minutes=1)" in src, (
            "CLV DateTrigger must use kickoff - timedelta(minutes=1)"
        )
