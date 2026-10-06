"""GREEN tests for D-16 reconcile (8 settlement paths + Pitfall 7 reschedule + Risk 9 unknown + Q3 4-attempt cap)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_orchestrator(api_response):
    from bip.core.scheduler.orchestrator import PipelineOrchestrator

    plugin = MagicMock()
    api_client = MagicMock()
    api_client.get_fixture = AsyncMock(return_value=api_response)
    pick_repo = MagicMock()
    pick_repo.update_status_by_fixture = MagicMock(return_value=2)
    pick_repo.update_status = MagicMock(return_value={"id": 1})

    orch = PipelineOrchestrator(
        plugin=plugin,
        settings=None,
        pick_repo=pick_repo,
        api_football_client=api_client,
    )
    # Mock scheduler to capture add_job calls without actually scheduling
    orch.scheduler = MagicMock()
    orch.scheduler.add_job = MagicMock()
    return orch, api_client, pick_repo


def _api_payload(status: str, home: int | None = 1, away: int | None = 0):
    return {"response": [{"fixture": {"status": {"short": status}},
                          "goals": {"home": home, "away": away}}]}


def _fixture(fid: int = 12345):
    f = MagicMock()
    f.fixture_id = fid
    return f


class TestReconcileSettlement:
    @pytest.mark.asyncio
    async def test_ft_home_win(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("FT", home=2, away=1))
        pick_repo.get_pending_for_fixture.return_value = [
            {"id": 1, "market": "1X2", "selection": "1"},
            {"id": 2, "market": "1X2", "selection": "X"},
            {"id": 3, "market": "1X2", "selection": "2"},
        ]
        await orch._reconcile_results(_fixture())
        calls = {c.args[0]: c.args[1] for c in pick_repo.update_status.call_args_list}
        assert calls == {1: "won", 2: "lost", 3: "lost"}

    @pytest.mark.asyncio
    async def test_aet_settles_on_regulation_plus_et_goals(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("AET", home=3, away=2))
        pick_repo.get_pending_for_fixture.return_value = [
            {"id": 1, "market": "1X2", "selection": "1"},
        ]
        await orch._reconcile_results(_fixture())
        pick_repo.update_status.assert_called_with(1, "won")

    @pytest.mark.asyncio
    async def test_pen_status_pushes_on_1x2(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("PEN", home=1, away=1))
        pick_repo.get_pending_for_fixture.return_value = [
            {"id": 1, "market": "1X2", "selection": "1"},
            {"id": 2, "market": "1X2", "selection": "X"},
        ]
        await orch._reconcile_results(_fixture())
        calls = {c.args[0]: c.args[1] for c in pick_repo.update_status.call_args_list}
        assert calls == {1: "push", 2: "push"}

    @pytest.mark.asyncio
    async def test_pst_voids(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("PST"))
        await orch._reconcile_results(_fixture())
        pick_repo.update_status_by_fixture.assert_called_once_with(12345, "void")

    @pytest.mark.asyncio
    async def test_canc_voids(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("CANC"))
        await orch._reconcile_results(_fixture())
        pick_repo.update_status_by_fixture.assert_called_once_with(12345, "void")

    @pytest.mark.asyncio
    async def test_abd_voids(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("ABD"))
        await orch._reconcile_results(_fixture())
        pick_repo.update_status_by_fixture.assert_called_once_with(12345, "void")


class TestReconcileReschedule:
    @pytest.mark.asyncio
    async def test_reschedule_on_in_play_2h(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("2H"))
        await orch._reconcile_results(_fixture())
        # In-play branch should NOT call update_status*
        pick_repo.update_status.assert_not_called()
        pick_repo.update_status_by_fixture.assert_not_called()
        # Should reschedule via add_job
        orch.scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "TBD", "NS"])
    async def test_all_in_play_codes_reschedule(self, status):
        orch, _, pick_repo = _make_orchestrator(_api_payload(status))
        await orch._reconcile_results(_fixture())
        pick_repo.update_status.assert_not_called()
        pick_repo.update_status_by_fixture.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_leaves_pending(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("2H"))
        await orch._reconcile_results(_fixture(), retries=4)
        pick_repo.update_status.assert_not_called()
        pick_repo.update_status_by_fixture.assert_not_called()
        # No add_job either — abandoned
        orch.scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_abandoned_after_4_retries(self, capsys):
        """Q3 RESOLVED: after 4 retries on INT/SUSP, log structlog.ERROR event=reconcile_abandoned."""
        orch, _, pick_repo = _make_orchestrator(_api_payload("SUSP"))
        await orch._reconcile_results(_fixture(), retries=4)
        pick_repo.update_status_by_fixture.assert_not_called()
        pick_repo.update_status.assert_not_called()
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "reconcile_abandoned" in combined, \
            f"expected 'reconcile_abandoned' in captured output; got:\n{combined}"


class TestReconcileUnknownAndAwarded:
    @pytest.mark.asyncio
    async def test_awd_settles_on_goals(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("AWD", home=3, away=0))
        pick_repo.get_pending_for_fixture.return_value = [
            {"id": 1, "market": "1X2", "selection": "1"},
        ]
        await orch._reconcile_results(_fixture())
        pick_repo.update_status.assert_called_with(1, "won")

    @pytest.mark.asyncio
    async def test_wo_settles_on_goals(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("WO", home=0, away=3))
        pick_repo.get_pending_for_fixture.return_value = [
            {"id": 1, "market": "1X2", "selection": "1"},
        ]
        await orch._reconcile_results(_fixture())
        pick_repo.update_status.assert_called_with(1, "lost")

    @pytest.mark.asyncio
    async def test_skips_non_1x2_markets(self):
        orch, _, pick_repo = _make_orchestrator(_api_payload("FT", home=2, away=1))
        pick_repo.get_pending_for_fixture.return_value = [
            {"id": 1, "market": "BTTS", "selection": "Yes"},
            {"id": 2, "market": "1X2", "selection": "1"},
        ]
        await orch._reconcile_results(_fixture())
        calls = {c.args[0]: c.args[1] for c in pick_repo.update_status.call_args_list}
        assert calls == {2: "won"}


class TestRegisterReconciliation:
    def test_reconcile_job_registered_on_fixture(self):
        from datetime import UTC, datetime, timedelta
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)
        orch.scheduler = MagicMock()
        orch.scheduler.add_job = MagicMock()

        fixture = MagicMock()
        fixture.fixture_id = 999
        fixture.kickoff_utc = datetime.now(UTC) + timedelta(hours=4)

        orch._register_fixture_jobs(fixture, datetime.now(UTC))

        reconcile_calls = [c for c in orch.scheduler.add_job.call_args_list
                           if c.kwargs.get("id") == "reconcile_999"]
        assert len(reconcile_calls) == 1
        kwargs = reconcile_calls[0].kwargs
        assert kwargs["misfire_grace_time"] == 600
        assert kwargs["replace_existing"] is True
