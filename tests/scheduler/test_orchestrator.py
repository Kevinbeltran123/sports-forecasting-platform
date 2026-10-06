"""GREEN tests for orchestrator extension (Pitfall 6 auto-recover + D-01 dup-alert guard
+ G-CODE-01/02/03 + G-MAINT-04 ABC compliance + full-Prediction build + sync repo guard).
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from bip.core.storage.models import Prediction
from bip.core.storage.repositories import PickRepository
from bip.sports import FeatureMatrix, FixtureData, ProbabilityMap, SportPlugin


def _make_orchestrator(pending_rows=None, existing_jobs=None):
    from bip.core.scheduler.orchestrator import PipelineOrchestrator

    plugin = MagicMock()
    pick_repo = MagicMock()
    pick_repo.query_pending_sends = MagicMock(return_value=pending_rows or [])

    pick_engine = MagicMock()
    pick_engine.evaluate = AsyncMock()

    orch = PipelineOrchestrator(plugin=plugin, pick_repo=pick_repo, pick_engine=pick_engine)

    fake_jobs = []
    for jid in (existing_jobs or []):
        j = MagicMock()
        j.id = jid
        fake_jobs.append(j)
    for jid in ("daily_orchestrator", "nightly_clv_reconciliation"):
        j = MagicMock()
        j.id = jid
        fake_jobs.append(j)

    orch.scheduler = MagicMock()
    orch.scheduler.get_jobs = MagicMock(return_value=fake_jobs)
    orch.scheduler.add_job = MagicMock()
    return orch, pick_repo


# ---------------------------------------------------------------------
# Helpers for ABC-compliant orchestrator builds (used by Task-2 tests)
# ---------------------------------------------------------------------

def _abc_plugin(
    *,
    league: str = "premier_league",
    model_version: str = "ensemble-v3",
    odds: dict | None = None,
) -> AsyncMock:
    """Build an AsyncMock(spec=SportPlugin) with sensible return values for the pipeline."""
    plugin = AsyncMock(spec=SportPlugin)

    fm = MagicMock(spec=FeatureMatrix)
    fm.fixture_id = 999
    fm.sport = "football"
    fm.league = league
    fm.kickoff_utc = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)
    fm.home_team = "Home FC"
    fm.away_team = "Away FC"
    plugin.build_features.return_value = fm

    pm = MagicMock(spec=ProbabilityMap)
    pm.fixture_id = 999
    pm.market = "1X2"
    pm.probabilities = {"1": 0.5, "X": 0.3, "2": 0.2}
    pm.model_version = model_version
    plugin.predict.return_value = pm

    plugin.get_opening_odds.return_value = (
        odds if odds is not None else {"1": 2.0, "X": 3.0, "2": 4.0}
    )
    return plugin


def _make_fixture(
    *,
    fixture_id: int = 999,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    league: str = "premier_league",
    kickoff: datetime | None = None,
) -> FixtureData:
    return FixtureData(
        fixture_id=fixture_id,
        league=league,
        sport="football",
        home_team=home_team,
        away_team=away_team,
        kickoff_utc=kickoff or datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )


class TestAutoRecover:
    @pytest.mark.asyncio
    async def test_recover_pending_sends(self, monkeypatch):
        from datetime import UTC, datetime
        import bip.core.scheduler.orchestrator as mod

        class _Now(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
        monkeypatch.setattr(mod, "datetime", _Now)

        orch, _ = _make_orchestrator(
            pending_rows=[{"id": 1, "fixture_id": 999, "market": "1X2", "sport": "football",
                           "claude_validation": "CONFIRM"}],
            existing_jobs=["pipeline_999_t_minus_2h"],
        )
        await orch._auto_recover()

        send_calls = [c for c in orch.scheduler.add_job.call_args_list
                      if c.kwargs.get("id") == "send_pick_999_1X2"]
        assert len(send_calls) == 1
        assert send_calls[0].kwargs["replace_existing"] is True
        assert send_calls[0].kwargs["misfire_grace_time"] == 300

    @pytest.mark.asyncio
    async def test_recover_skips_already_scheduled(self, monkeypatch):
        from datetime import UTC, datetime
        import bip.core.scheduler.orchestrator as mod

        class _Now(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
        monkeypatch.setattr(mod, "datetime", _Now)

        orch, _ = _make_orchestrator(
            pending_rows=[{"id": 1, "fixture_id": 999, "market": "1X2", "sport": "football",
                           "claude_validation": "CONFIRM"}],
            existing_jobs=["pipeline_999_t_minus_2h", "send_pick_999_1X2"],
        )
        await orch._auto_recover()

        send_calls = [c for c in orch.scheduler.add_job.call_args_list
                      if c.kwargs.get("id") == "send_pick_999_1X2"]
        assert len(send_calls) == 0

    @pytest.mark.asyncio
    async def test_recover_handles_no_pending(self, monkeypatch):
        from datetime import UTC, datetime
        import bip.core.scheduler.orchestrator as mod

        class _Now(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
        monkeypatch.setattr(mod, "datetime", _Now)

        orch, _ = _make_orchestrator(pending_rows=[], existing_jobs=["pipeline_999_t_minus_2h"])
        await orch._auto_recover()
        send_calls = [c for c in orch.scheduler.add_job.call_args_list
                      if c.kwargs.get("id", "").startswith("send_pick_")]
        assert send_calls == []

    @pytest.mark.asyncio
    async def test_recover_skips_when_unwired(self, monkeypatch):
        from datetime import UTC, datetime
        import bip.core.scheduler.orchestrator as mod

        class _Now(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
        monkeypatch.setattr(mod, "datetime", _Now)

        from bip.core.scheduler.orchestrator import PipelineOrchestrator
        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)
        # Provide a non-default fixture job so daily_orchestrator path is skipped
        existing = MagicMock()
        existing.id = "pipeline_999_t_minus_2h"
        orch.scheduler = MagicMock()
        orch.scheduler.get_jobs = MagicMock(return_value=[existing])
        # Should not raise (pick_repo None → early return after fixture jobs check)
        await orch._auto_recover()


class TestT30DupAlertGuard:
    """D-01 dup-alert guard tests (Blocker #1 fix) -- now with sync repo + dict access."""

    @pytest.mark.asyncio
    async def test_t30_skipped_when_t2h_pending(self, capsys):
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = _abc_plugin()
        pick_engine = MagicMock()
        pick_engine.evaluate = AsyncMock()

        # SYNC, returns list[dict] (matches PickRepository.get_pending_for_fixture real shape)
        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(
            return_value=[{"id": 42, "market": "1X2", "status": "pending"}]
        )

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_engine=pick_engine,
            pick_repo=pick_repo,
        )

        fixture = _make_fixture()
        await orch._run_pipeline(fixture, stage="t_minus_30m")

        pick_engine.evaluate.assert_not_called()
        plugin.predict.assert_not_called()
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "t30_skipped_pending_already" in combined, (
            f"expected 't30_skipped_pending_already' in stdout; got:\n{combined}"
        )

    @pytest.mark.asyncio
    async def test_t30_proceeds_when_t2h_rejected(self):
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = _abc_plugin()
        pick_engine = MagicMock()
        pick_engine.evaluate = AsyncMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(
            return_value=[{"id": 43, "market": "1X2", "status": "rejected"}]
        )

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_engine=pick_engine,
            pick_repo=pick_repo,
        )

        fixture = _make_fixture()
        await orch._run_pipeline(fixture, stage="t_minus_30m")
        pick_engine.evaluate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_t30_proceeds_when_t2h_filtered(self):
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = _abc_plugin()
        pick_engine = MagicMock()
        pick_engine.evaluate = AsyncMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(return_value=[])

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_engine=pick_engine,
            pick_repo=pick_repo,
        )

        fixture = _make_fixture()
        await orch._run_pipeline(fixture, stage="t_minus_30m")
        pick_engine.evaluate.assert_awaited_once()


class TestPipelineABCCompliance:
    """G-CODE-01/02/03 + G-MAINT-04 — wiring contracts."""

    @pytest.mark.asyncio
    async def test_pipeline_invokes_only_plugin_abc_methods(self):
        """No AttributeError when plugin is strictly spec'd to SportPlugin ABC."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = _abc_plugin()
        pick_engine = MagicMock()
        pick_engine.evaluate = AsyncMock()

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_engine=pick_engine,
            pick_repo=None,
        )

        fixture = _make_fixture()
        await orch._run_pipeline(fixture, stage="t_minus_2h")

        plugin.build_features.assert_awaited_once()
        plugin.predict.assert_awaited_once()
        plugin.get_opening_odds.assert_awaited_once_with(fixture.fixture_id)

    @pytest.mark.asyncio
    async def test_pipeline_builds_full_prediction(self):
        """Orchestrator constructs a typed Prediction from fixture+features+prob_map."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = _abc_plugin(
            league="premier_league",
            model_version="ensemble-v3",
            odds={"1": 2.0, "X": 3.0, "2": 4.0},
        )
        pick_engine = MagicMock()
        pick_engine.evaluate = AsyncMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(return_value=[])

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_engine=pick_engine,
            pick_repo=pick_repo,
        )

        kickoff = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)
        fixture = _make_fixture(
            fixture_id=999,
            home_team="Arsenal",
            away_team="Chelsea",
            league="premier_league",
            kickoff=kickoff,
        )

        await orch._run_pipeline(fixture, stage="t_minus_30m")

        pick_engine.evaluate.assert_awaited_once()
        prediction = pick_engine.evaluate.call_args.args[0]
        opening_odds = pick_engine.evaluate.call_args.args[1]

        assert isinstance(prediction, Prediction)
        assert prediction.fixture_id == 999
        assert prediction.home_team == "Arsenal"
        assert prediction.away_team == "Chelsea"
        assert prediction.league == "premier_league"
        assert prediction.sport == "football"
        assert prediction.kickoff_utc == kickoff
        # G-MAINT-05: orchestrator now persists the canonical "onextwo" key
        # via MarketKey.ONEXTWO; legacy "1X2" rows still recognized at the
        # filter boundary via MarketKey.from_str.
        assert prediction.market == "onextwo"
        assert prediction.model_version == "ensemble-v3"
        assert prediction.probabilities == {"1": 0.5, "X": 0.3, "2": 0.2}
        assert prediction.is_lineup_adjusted is True

        assert opening_odds == {"1": 2.0, "X": 3.0, "2": 4.0}

    @pytest.mark.asyncio
    async def test_dup_alert_guard_uses_sync_repo_and_dict_access(self):
        """G-CODE-02 — sync call (no await) + dict access via p['market']."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = _abc_plugin()
        pick_engine = MagicMock()
        pick_engine.evaluate = AsyncMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(
            return_value=[{"id": 42, "market": "1X2", "status": "pending"}]
        )

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_engine=pick_engine,
            pick_repo=pick_repo,
        )

        fixture = _make_fixture()
        await orch._run_pipeline(fixture, stage="t_minus_30m")

        pick_repo.get_pending_for_fixture.assert_called_once_with(fixture.fixture_id)
        pick_engine.evaluate.assert_not_called()
        plugin.predict.assert_not_called()


# ---------------------------------------------------------------------
# Task 2 helpers — record_clv wiring
# ---------------------------------------------------------------------

PINNACLE_BOOKMAKER_PAYLOAD = {
    "key": "pinnacle",
    "title": "Pinnacle",
    "markets": [
        {
            "key": "h2h",
            "outcomes": [
                {"name": "Arsenal", "price": 2.10},
                {"name": "Draw", "price": 3.40},
                {"name": "Chelsea", "price": 3.80},
            ],
        }
    ],
}


def _league_registry_for(slug: str = "premier_league", sport_key: str = "soccer_epl"):
    """Minimal stub: registry.get(slug).api_mappings.odds_api_sport_key."""
    config = MagicMock()
    config.api_mappings.odds_api_sport_key = sport_key
    registry = MagicMock()
    registry.get = MagicMock(return_value=config)
    return registry


class TestRecordClv:
    """Wire ClvRecorder + OddsApiClient into PipelineOrchestrator._record_clv."""

    @pytest.mark.asyncio
    async def test_record_clv_happy_path(self):
        """Pending pick + matched event + Pinnacle quote → ClvRecorder.record() called."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        odds_api_client = MagicMock()
        odds_api_client.find_event_by_fixture = AsyncMock(return_value="event-uuid-123")
        odds_api_client.fetch_pinnacle_closing_odds = AsyncMock(
            return_value=PINNACLE_BOOKMAKER_PAYLOAD
        )

        clv_recorder = MagicMock()
        clv_recorder.record = MagicMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(
            return_value=[
                {
                    "id": 42,
                    "market": "1X2",
                    "selection": "1",
                    "odds_at_pick": 2.05,
                    "status": "pending",
                }
            ]
        )

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_repo=pick_repo,
            odds_api_client=odds_api_client,
            clv_recorder=clv_recorder,
            league_registry=_league_registry_for("premier_league"),
        )

        fixture = _make_fixture(
            fixture_id=999,
            home_team="Arsenal",
            away_team="Chelsea",
            league="premier_league",
        )

        await orch._record_clv(fixture)

        odds_api_client.find_event_by_fixture.assert_awaited_once()
        odds_api_client.fetch_pinnacle_closing_odds.assert_awaited_once_with(
            sport_key="soccer_epl", event_id="event-uuid-123", market_key="h2h"
        )
        clv_recorder.record.assert_called_once()
        kwargs = clv_recorder.record.call_args.kwargs
        assert kwargs["pick_id"] == 42
        assert kwargs["fixture_id"] == 999
        assert kwargs["selection"] == "1"
        assert kwargs["odds_at_pick"] == 2.05
        assert set(kwargs["closing_odds_dict"].keys()) == {"1", "X", "2"}
        assert kwargs["closing_odds_dict"]["1"] == 2.10
        assert kwargs["closing_odds_dict"]["X"] == 3.40
        assert kwargs["closing_odds_dict"]["2"] == 3.80

    @pytest.mark.asyncio
    async def test_record_clv_skips_when_no_pending_picks(self, capsys):
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        odds_api_client = MagicMock()
        odds_api_client.find_event_by_fixture = AsyncMock()
        odds_api_client.fetch_pinnacle_closing_odds = AsyncMock()
        clv_recorder = MagicMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(return_value=[])

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_repo=pick_repo,
            odds_api_client=odds_api_client,
            clv_recorder=clv_recorder,
            league_registry=_league_registry_for(),
        )
        fixture = _make_fixture()

        await orch._record_clv(fixture)

        clv_recorder.record.assert_not_called()
        odds_api_client.find_event_by_fixture.assert_not_called()
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "clv_skip_no_pending" in combined

    @pytest.mark.asyncio
    async def test_record_clv_skips_when_pinnacle_event_not_found(self, capsys):
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        odds_api_client = MagicMock()
        odds_api_client.find_event_by_fixture = AsyncMock(return_value=None)
        odds_api_client.fetch_pinnacle_closing_odds = AsyncMock()
        clv_recorder = MagicMock()

        pick_repo = Mock(spec=PickRepository)
        pick_repo.get_pending_for_fixture = Mock(
            return_value=[
                {
                    "id": 42,
                    "market": "1X2",
                    "selection": "1",
                    "odds_at_pick": 2.05,
                    "status": "pending",
                }
            ]
        )

        orch = PipelineOrchestrator(
            plugin=plugin,
            pick_repo=pick_repo,
            odds_api_client=odds_api_client,
            clv_recorder=clv_recorder,
            league_registry=_league_registry_for(),
        )
        fixture = _make_fixture()

        await orch._record_clv(fixture)

        clv_recorder.record.assert_not_called()
        odds_api_client.fetch_pinnacle_closing_odds.assert_not_called()
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "clv_skip_no_event" in combined

    @pytest.mark.asyncio
    async def test_record_clv_skips_when_unwired(self, capsys):
        """No clv_recorder / odds_api_client wired → log + return; no crash."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)
        fixture = _make_fixture()

        # Should not raise — early returns on missing dependencies
        await orch._record_clv(fixture)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "clv_skip_unwired" in combined


# ─────────────────────────────────────────────────────────────────
# Phase 4 wiring tests (D-03, D-07, D-08, D-09–D-12, D-13, D-14)
# ─────────────────────────────────────────────────────────────────


class TestPhase4Wiring:
    """4 new cron jobs + drift_checker delegation + auto_recover_complete event."""

    @pytest.mark.asyncio
    async def test_phase4_jobs_registered_when_deps_provided(self):
        """All Phase 4 deps wired → start() registers heartbeat + clv_trend + metrics + drift."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        heartbeat = MagicMock()
        heartbeat.tick = MagicMock()
        clv_trend = MagicMock()
        metrics_agg = MagicMock()
        drift_checker = MagicMock()
        ops_sender = MagicMock()

        orch = PipelineOrchestrator(
            plugin=plugin,
            heartbeat_ticker=heartbeat,
            clv_trend_checker=clv_trend,
            metrics_aggregator=metrics_agg,
            drift_checker=drift_checker,
            ops_sender=ops_sender,
        )
        # AsyncIOScheduler.start() needs a running loop — provided by @pytest.mark.asyncio.
        orch.start()
        try:
            job_ids = {j.id for j in orch.scheduler.get_jobs()}
            assert "heartbeat" in job_ids
            assert "clv_trend" in job_ids
            assert "metrics_aggregator" in job_ids
            assert "weekly_drift" in job_ids
        finally:
            orch.scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_phase4_jobs_skipped_when_deps_none(self):
        """Phase 4 deps None → only the 2 Phase 3 jobs register."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)
        orch.start()
        try:
            job_ids = {j.id for j in orch.scheduler.get_jobs()}
            assert "heartbeat" not in job_ids
            assert "clv_trend" not in job_ids
            assert "metrics_aggregator" not in job_ids
            assert "weekly_drift" not in job_ids
            # Phase 3 jobs still register
            assert "daily_orchestrator" in job_ids
            assert "nightly_clv_reconciliation" in job_ids
        finally:
            orch.scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_check_drift_delegates_to_drift_checker(self):
        """_check_drift calls self._drift_checker.run() when wired."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        drift_checker = MagicMock()
        drift_checker.run = AsyncMock(return_value=2)
        orch = PipelineOrchestrator(plugin=plugin, drift_checker=drift_checker)
        await orch._check_drift()
        drift_checker.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_drift_skips_gracefully_without_drift_checker(self):
        """_check_drift logs skip when drift_checker is None — never crashes."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        plugin = MagicMock()
        orch = PipelineOrchestrator(plugin=plugin)  # no drift_checker
        # Should not raise.
        await orch._check_drift()

    @pytest.mark.asyncio
    async def test_auto_recover_complete_event_fires(self, capsys):
        """D-08: _auto_recover emits 'auto_recover_complete' with jobs_re_queued count."""
        from bip.core.scheduler.orchestrator import PipelineOrchestrator

        # Force the auto_recover path to reach the requeued log: need pick_repo + pick_engine
        # set, AND the early `now < today_start` guard must NOT fire (i.e. tests run after
        # 06:00 UTC). The recovery-already-skipped branch returns BEFORE the requeued log
        # only if no fixture jobs exist AND _daily_orchestrator runs; but the pending-sends
        # block runs unconditionally after that. To exercise just the pending-sends log
        # path, stub _daily_orchestrator to a no-op via plugin mock.
        plugin = MagicMock()
        pick_repo = MagicMock()
        pick_repo.query_pending_sends = MagicMock(return_value=[])  # no pending → count=0
        pick_engine = MagicMock()
        orch = PipelineOrchestrator(
            plugin=plugin, pick_repo=pick_repo, pick_engine=pick_engine
        )
        # Stub the daily orchestrator so _auto_recover doesn't fan out into fixture logic.
        orch._daily_orchestrator = AsyncMock()  # type: ignore[method-assign]
        await orch._auto_recover()
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "auto_recover_complete" in combined
        assert "jobs_re_queued=0" in combined
