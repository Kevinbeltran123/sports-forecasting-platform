"""GREEN tests for PickRepository extensions + PickEngine.evaluate (PICK-01..05, CLAUDE-01)."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestPickRepositoryExtensions:
    def test_get_window_picks_filters(self, mock_client):
        from tests.conftest import setup_mock_chain
        from bip.core.storage.repositories import PickRepository

        builder = setup_mock_chain(mock_client, data=[
            {"market": "1X2", "status": "pending", "created_at": "2026-05-02T10:00:00+00:00"},
        ])
        builder.neq.return_value = builder
        builder.gte.return_value = builder

        repo = PickRepository(client=mock_client)
        rows = repo.get_window_picks(sport="football", hours=168)
        assert isinstance(rows, list) and len(rows) == 1
        mock_client.table.assert_called_with("picks")

    def test_get_pending_for_fixture(self, mock_client):
        from tests.conftest import setup_mock_chain
        from bip.core.storage.repositories import PickRepository
        setup_mock_chain(mock_client, data=[{"id": 1, "status": "pending"}])
        repo = PickRepository(client=mock_client)
        assert len(repo.get_pending_for_fixture(fixture_id=999)) == 1
        mock_client.table.assert_called_with("picks")

    def test_update_status_by_fixture(self, mock_client):
        from tests.conftest import setup_mock_chain
        from bip.core.storage.repositories import PickRepository
        setup_mock_chain(mock_client, data=[{"id": 1}, {"id": 2}])
        repo = PickRepository(client=mock_client)
        assert repo.update_status_by_fixture(fixture_id=1, new_status="void") == 2

    def test_query_pending_sends_filters_by_age(self, mock_client):
        from tests.conftest import setup_mock_chain
        from bip.core.storage.repositories import PickRepository

        builder = setup_mock_chain(mock_client, data=[{"id": 1, "status": "pending", "claude_validation": "CONFIRM"}])
        not_obj = MagicMock()
        not_obj.is_.return_value = builder
        builder.not_ = not_obj
        builder.gte.return_value = builder

        repo = PickRepository(client=mock_client)
        rows = repo.query_pending_sends(sport="football", max_age_minutes=30)
        assert isinstance(rows, list)
        mock_client.table.assert_called_with("picks")

    def test_storage_errors_wrap_exceptions(self, mock_client):
        from bip.core.errors import StorageError
        from bip.core.storage.repositories import PickRepository
        mock_client.table.side_effect = RuntimeError("supabase unreachable")
        repo = PickRepository(client=mock_client)
        with pytest.raises(StorageError):
            repo.get_window_picks(sport="football")


def _make_prediction(probs: dict[str, float], market: str = "1X2"):
    p = MagicMock()
    p.id = 100
    p.fixture_id = 12345
    p.league = "premier_league"
    p.sport = "football"
    p.market = market
    p.probabilities = probs
    p.home_team = "Manchester United"
    p.away_team = "Chelsea"
    p.model_version = "v1"
    p.kickoff_utc = datetime(2026, 5, 2, 15, 0, 0, tzinfo=UTC)
    p.is_lineup_adjusted = False
    return p


def _default_league_registry():
    """Mock LeagueRegistry returning YAML-configured thresholds for premier_league."""
    league_registry = MagicMock()
    default_params = MagicMock(
        spec=[
            "edge_threshold_1x2",
            "edge_threshold_btts",
            "edge_threshold_ou",
            "edge_threshold_ah",
            "edge_threshold_corners",
        ]
    )
    default_params.edge_threshold_1x2 = 0.08
    default_params.edge_threshold_btts = 0.05
    default_params.edge_threshold_ou = 0.05
    default_params.edge_threshold_ah = 0.06
    default_params.edge_threshold_corners = 0.07
    default_cfg = MagicMock()
    default_cfg.model_params = default_params
    league_registry.get = MagicMock(return_value=default_cfg)
    return league_registry


def _make_engine(claude_verdict=None, window_picks=None, league_registry=None):
    from bip.core.picks.engine import PickEngine

    pick_repo = MagicMock()
    pick_repo.insert = MagicMock(return_value={"id": 999})
    pick_repo.get_window_picks = MagicMock(return_value=window_picks or [])

    validator = MagicMock()
    validator.validate = AsyncMock(return_value=claude_verdict)

    scheduler = MagicMock()
    scheduler.add_job = MagicMock()

    sender = MagicMock()

    settings = MagicMock()
    settings.max_kelly_fraction = 0.25
    settings.claude_failure_mode = "filter"  # D-01 default — preserves Phase 3 D-07 semantics

    if league_registry is None:
        league_registry = _default_league_registry()

    engine = PickEngine(
        pick_repo=pick_repo, validator=validator, scheduler=scheduler,
        sender=sender, settings=settings, league_registry=league_registry,
    )
    return engine, pick_repo, validator, scheduler, sender


class TestEvaluateAllPaths:
    @pytest.mark.asyncio
    async def test_no_edge_persists_filtered_no_edge(self):
        """D-02 + D-04: simulate_pick returns None → status=filtered, reason_code=no_edge."""
        from bip.core.types import PickStatus
        engine, pick_repo, validator, scheduler, _ = _make_engine()
        prediction = _make_prediction({"1": 0.50, "X": 0.30, "2": 0.20})
        opening_odds = {"1": 2.00, "X": 3.33, "2": 5.00, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick is not None
        assert pick.status == PickStatus.filtered
        assert "no_edge" in (pick.claude_reasoning or "")
        pick_repo.insert.assert_called_once()
        validator.validate.assert_not_awaited()
        scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_market_cap_persists_filtered_market_cap(self):
        """D-09: cap exceeded → status=filtered, reason_code=market_cap."""
        from bip.core.types import PickStatus
        window = [{"market": "1X2"}] * 6 + [{"market": "BTTS"}] * 4
        engine, _, validator, _, _ = _make_engine(window_picks=window)
        prediction = _make_prediction({"1": 0.60, "X": 0.25, "2": 0.15})
        opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.filtered
        assert "market_cap" in (pick.claude_reasoning or "")
        validator.validate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claude_unavailable_filters(self):
        """D-07: validator returns None → status=filtered, reason_code=claude_api_unavailable."""
        from bip.core.types import PickStatus
        engine, _, validator, scheduler, _ = _make_engine(claude_verdict=None)
        prediction = _make_prediction({"1": 0.60, "X": 0.25, "2": 0.15})
        opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.filtered
        assert "claude_api_unavailable" in (pick.claude_reasoning or "")
        validator.validate.assert_awaited_once()
        scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_claude_reject_persists_rejected(self):
        """D-03 + D-14: REJECT verdict → status=rejected, never reaches Telegram."""
        from bip.core.claude.validator import ClaudeVerdict
        from bip.core.types import PickStatus
        verdict = ClaudeVerdict(verdict="REJECT", reason_code="aggregate_stat_seduction",
                                reasoning="r", summary="s")
        engine, _, _, scheduler, _ = _make_engine(claude_verdict=verdict)
        prediction = _make_prediction({"1": 0.60, "X": 0.25, "2": 0.15})
        opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.rejected
        assert pick.claude_validation == "REJECT"
        scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_persists_pending_and_schedules_send(self):
        """D-11 + D-14: CONFIRM → status=pending + DateTrigger send job registered."""
        from bip.core.claude.validator import ClaudeVerdict
        from bip.core.types import PickStatus
        verdict = ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="r", summary="s")
        engine, _, _, scheduler, _ = _make_engine(claude_verdict=verdict)
        prediction = _make_prediction({"1": 0.60, "X": 0.25, "2": 0.15})
        opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.pending
        assert pick.claude_validation == "CONFIRM"
        scheduler.add_job.assert_called_once()
        kwargs = scheduler.add_job.call_args.kwargs
        assert kwargs["id"] == "send_pick_12345_1X2"
        assert kwargs["replace_existing"] is True
        assert kwargs["misfire_grace_time"] == 300

    @pytest.mark.asyncio
    async def test_flag_persists_pending_with_reason(self):
        """D-14: FLAG verdict → status=pending, claude_validation='FLAG'."""
        from bip.core.claude.validator import ClaudeVerdict
        from bip.core.types import PickStatus
        verdict = ClaudeVerdict(verdict="FLAG", reason_code="h2h_too_old",
                                reasoning="r", summary="s")
        engine, _, _, scheduler, _ = _make_engine(claude_verdict=verdict)
        prediction = _make_prediction({"1": 0.60, "X": 0.25, "2": 0.15})
        opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)
        assert pick.status == PickStatus.pending
        assert pick.claude_validation == "FLAG"
        scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_paths_persist(self):
        """D-04 audit: every outcome (filtered, rejected, pending) hits picks table exactly once."""
        from bip.core.claude.validator import ClaudeVerdict

        engine, repo, _, _, _ = _make_engine()
        await engine.evaluate(_make_prediction({"1": 0.50, "X": 0.30, "2": 0.20}),
                              {"1": 2.00, "X": 3.33, "2": 5.00, "bookmaker": "betano"})
        assert repo.insert.call_count == 1

        engine, repo, _, _, _ = _make_engine(claude_verdict=None)
        await engine.evaluate(_make_prediction({"1": 0.60, "X": 0.25, "2": 0.15}),
                              {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"})
        assert repo.insert.call_count == 1

        engine, repo, _, _, _ = _make_engine(
            claude_verdict=ClaudeVerdict(verdict="REJECT", reason_code="x", reasoning="r", summary="s")
        )
        await engine.evaluate(_make_prediction({"1": 0.60, "X": 0.25, "2": 0.15}),
                              {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"})
        assert repo.insert.call_count == 1

        engine, repo, _, _, _ = _make_engine(
            claude_verdict=ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="r", summary="s")
        )
        await engine.evaluate(_make_prediction({"1": 0.60, "X": 0.25, "2": 0.15}),
                              {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"})
        assert repo.insert.call_count == 1

    @pytest.mark.asyncio
    async def test_idempotent_on_fixture_market_prediction(self):
        """specifics §195: re-running for same input registers send job with same id + replace_existing=True."""
        from bip.core.claude.validator import ClaudeVerdict
        verdict = ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="r", summary="s")
        engine, _, _, scheduler, _ = _make_engine(claude_verdict=verdict)
        prediction = _make_prediction({"1": 0.60, "X": 0.25, "2": 0.15})
        opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}

        await engine.evaluate(prediction, opening_odds)
        await engine.evaluate(prediction, opening_odds)

        ids = [c.kwargs["id"] for c in scheduler.add_job.call_args_list]
        assert ids == ["send_pick_12345_1X2", "send_pick_12345_1X2"]
        assert all(c.kwargs["replace_existing"] is True for c in scheduler.add_job.call_args_list)


class TestPerMarketEdgeThreshold:
    """G-MAINT-01 / CORE-05: PickEngine reads per-market threshold from LeagueRegistry."""

    @pytest.mark.asyncio
    async def test_engine_uses_per_market_threshold_from_yaml(self):
        """1X2 pick at ~5.6% edge is filtered (no_edge) — premier_league requires 8%."""
        from bip.core.types import PickStatus
        engine, _, validator, scheduler, _ = _make_engine()
        # Edge for "1" = 0.55 * 1.92 - 1 = 0.056 (5.6%) -- above 5%, BELOW 8%.
        prediction = _make_prediction({"1": 0.55, "X": 0.25, "2": 0.20})
        opening_odds = {"1": 1.92, "X": 3.40, "2": 4.50, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.filtered
        assert "no_edge" in (pick.claude_reasoning or "")
        validator.validate.assert_not_awaited()
        scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_engine_uses_btts_threshold_from_yaml(self):
        """BTTS pick at ~5.6% edge proceeds to validator -- BTTS threshold is 5%."""
        from bip.core.claude.validator import ClaudeVerdict
        from bip.core.types import PickStatus
        verdict = ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="r", summary="s")
        engine, _, validator, _, _ = _make_engine(claude_verdict=verdict)
        prediction = _make_prediction({"1": 0.55, "X": 0.25, "2": 0.20}, market="btts")
        opening_odds = {"1": 1.92, "X": 3.40, "2": 4.50, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.pending
        validator.validate.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("market", ["1X2", "onextwo", "h2h"])
    async def test_engine_normalizes_market_alias(self, market):
        """All three 1X2 aliases resolve to edge_threshold_1x2 (8%), so 6% edge filters."""
        from bip.core.types import PickStatus
        engine, _, validator, _, _ = _make_engine()
        # Edge for "1" = 0.55 * 1.93 - 1 = 0.0615 (6.15%) -- above 5%, BELOW 8%.
        prediction = _make_prediction({"1": 0.55, "X": 0.25, "2": 0.20}, market=market)
        opening_odds = {"1": 1.93, "X": 3.40, "2": 4.50, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.filtered, (
            f"market={market} should resolve to edge_threshold_1x2=0.08 and filter ~6% edge"
        )
        assert "no_edge" in (pick.claude_reasoning or "")
        validator.validate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_engine_falls_back_to_default_for_unknown_market(self):
        """Unknown market keys fall back to EDGE_THRESHOLD_PCT (0.05) so picks still flow."""
        from bip.core.claude.validator import ClaudeVerdict
        from bip.core.types import PickStatus

        # Mock registry returns a config whose model_params has NO edge_threshold_unknown_xyz attr.
        league_registry = MagicMock()
        sparse_params = MagicMock(spec=["edge_threshold_1x2"])
        sparse_params.edge_threshold_1x2 = 0.08
        sparse_cfg = MagicMock()
        sparse_cfg.model_params = sparse_params
        league_registry.get = MagicMock(return_value=sparse_cfg)

        verdict = ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="r", summary="s")
        engine, _, validator, _, _ = _make_engine(
            claude_verdict=verdict, league_registry=league_registry,
        )
        # Edge for "1" = 0.55 * 1.93 - 1 = 0.0615 -- above 5% default fallback.
        prediction = _make_prediction(
            {"1": 0.55, "X": 0.25, "2": 0.20}, market="unknown_xyz",
        )
        opening_odds = {"1": 1.93, "X": 3.40, "2": 4.50, "bookmaker": "betano"}

        pick = await engine.evaluate(prediction, opening_odds)

        assert pick.status == PickStatus.pending
        validator.validate.assert_awaited_once()

    def test_engine_constructor_requires_league_registry(self):
        """Fail fast: PickEngine without league_registry raises TypeError."""
        from bip.core.picks.engine import PickEngine
        with pytest.raises(TypeError):
            PickEngine(  # type: ignore[call-arg]
                pick_repo=MagicMock(),
                validator=MagicMock(),
                scheduler=MagicMock(),
                sender=MagicMock(),
                settings=MagicMock(),
                # league_registry intentionally omitted
            )
