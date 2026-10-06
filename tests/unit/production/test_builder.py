"""build_orchestrator factory tests (D-05 / D-03 / drift_checker wiring)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-af")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "-1001234567890")
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL_ID", "-1009876543210")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    from bip.core.settings import Settings
    return Settings()


def _pick_engine_factory(**kw):
    """Mock factory that propagates the scheduler kwarg into _scheduler so the
    builder's `pick_engine._scheduler is orchestrator.scheduler` invariant assert
    passes against the same shared_scheduler the real PickEngine would store."""
    return MagicMock(_scheduler=kw.get("scheduler"))


def _orchestrator_factory(**kw):
    """Mirror the scheduler kwarg into the mock orchestrator's .scheduler attr
    so the invariant assert matches identity against the same shared_scheduler."""
    return MagicMock(scheduler=kw.get("scheduler"))


def _patch_all_deps():
    """Returns a dict of patcher contexts ready to enter — keys map to friendly names."""
    return {
        "create_client": patch("bip.production.builder.create_client"),
        "TelegramBot": patch("bip.production.builder.TelegramBot"),
        "TelegramSender": patch("bip.production.builder.TelegramSender"),
        "OddsApiClient": patch("bip.production.builder.OddsApiClient"),
        "ClvRecorder": patch("bip.production.builder.ClvRecorder"),
        "ClaudeValidator": patch("bip.production.builder.ClaudeValidator"),
        "PickEngine": patch(
            "bip.production.builder.PickEngine", side_effect=_pick_engine_factory
        ),
        "LeagueRegistry": patch("bip.production.builder.LeagueRegistry"),
        "FootballPlugin": patch("bip.production.builder.FootballPlugin"),
        "_load_learnings": patch(
            "bip.production.builder._load_learnings",
            return_value=("# learnings", "abc123"),
        ),
    }


def test_build_orchestrator_returns_4_tuple(fake_settings):
    """build_orchestrator returns (PipelineOrchestrator, picks_bot, ops_bot, sd_notifier)."""
    patchers = _patch_all_deps()
    with (
        patchers["create_client"] as mock_create_client,
        patchers["TelegramBot"] as mock_bot,
        patchers["TelegramSender"],
        patchers["OddsApiClient"],
        patchers["ClvRecorder"],
        patchers["ClaudeValidator"],
        patchers["PickEngine"],
        patchers["LeagueRegistry"],
        patchers["FootballPlugin"],
        patchers["_load_learnings"],
    ):
        mock_create_client.return_value = MagicMock()
        mock_bot.side_effect = lambda **kw: MagicMock(channel_id=kw.get("channel_id"))
        from bip.production.builder import build_orchestrator
        result = build_orchestrator(fake_settings)
        assert isinstance(result, tuple)
        assert len(result) == 4


def test_build_orchestrator_wires_two_telegram_bots(fake_settings):
    """Two distinct TelegramBot calls — one per channel (D-03 Option A)."""
    patchers = _patch_all_deps()
    with (
        patchers["create_client"] as mock_create_client,
        patchers["TelegramBot"] as mock_bot,
        patchers["TelegramSender"],
        patchers["OddsApiClient"],
        patchers["ClvRecorder"],
        patchers["ClaudeValidator"],
        patchers["PickEngine"],
        patchers["LeagueRegistry"],
        patchers["FootballPlugin"],
        patchers["_load_learnings"],
    ):
        mock_create_client.return_value = MagicMock()
        mock_bot.side_effect = lambda **kw: MagicMock(channel_id=kw.get("channel_id"))
        from bip.production.builder import build_orchestrator
        build_orchestrator(fake_settings)
        assert mock_bot.call_count == 2
        channel_ids = {call.kwargs.get("channel_id") for call in mock_bot.call_args_list}
        assert "-1001234567890" in channel_ids
        assert "-1009876543210" in channel_ids


def test_build_orchestrator_passes_settings_to_components(fake_settings):
    """Settings values flow to constructors (smoke test for non-trivial wiring)."""
    with (
        patch("bip.production.builder.create_client") as mock_create_client,
        patch("bip.production.builder.TelegramBot") as mock_bot,
        patch("bip.production.builder.TelegramSender"),
        patch("bip.production.builder.OddsApiClient") as mock_odds,
        patch("bip.production.builder.ClvRecorder"),
        patch("bip.production.builder.ClaudeValidator") as mock_claude,
        patch("bip.production.builder.PickEngine", side_effect=_pick_engine_factory),
        patch("bip.production.builder.LeagueRegistry"),
        patch("bip.production.builder.FootballPlugin"),
        patch(
            "bip.production.builder._load_learnings",
            return_value=("# learnings", "abc123"),
        ),
    ):
        mock_create_client.return_value = MagicMock()
        mock_bot.side_effect = lambda **kw: MagicMock(channel_id=kw.get("channel_id"))
        from bip.production.builder import build_orchestrator
        build_orchestrator(fake_settings)
        mock_odds.assert_called()
        mock_claude.assert_called()
        kw = mock_claude.call_args.kwargs
        assert kw.get("api_key") == "test-anthropic"
        # REAL learnings_text + learnings_sha (no None placeholders)
        assert kw.get("learnings_text") is not None
        assert kw.get("learnings_sha") is not None


def test_build_orchestrator_wires_drift_checker(fake_settings):
    """DriftChecker is constructed AND passed to PipelineOrchestrator(drift_checker=...).

    Without this wiring, the weekly_drift cron registers but the orchestrator's
    _check_drift body has nothing to delegate to (logs 'drift_check_skipped' forever).
    """
    with (
        patch("bip.production.builder.create_client") as mock_create_client,
        patch("bip.production.builder.TelegramBot") as mock_bot,
        patch("bip.production.builder.TelegramSender"),
        patch("bip.production.builder.OddsApiClient"),
        patch("bip.production.builder.ClvRecorder"),
        patch("bip.production.builder.ClaudeValidator"),
        patch("bip.production.builder.PickEngine", side_effect=_pick_engine_factory),
        patch("bip.production.builder.LeagueRegistry"),
        patch("bip.production.builder.FootballPlugin"),
        patch(
            "bip.production.builder._load_learnings",
            return_value=("# learnings", "abc123"),
        ),
        patch(
            "bip.production.builder.PipelineOrchestrator",
            side_effect=_orchestrator_factory,
        ) as mock_orch,
    ):
        mock_create_client.return_value = MagicMock()
        mock_bot.side_effect = lambda **kw: MagicMock(channel_id=kw.get("channel_id"))
        from bip.production.builder import build_orchestrator
        build_orchestrator(fake_settings)
        mock_orch.assert_called_once()
        kwargs = mock_orch.call_args.kwargs
        assert "drift_checker" in kwargs, (
            "drift_checker kwarg missing from PipelineOrchestrator call"
        )
        assert kwargs["drift_checker"] is not None, (
            "drift_checker is None — must be DriftChecker instance"
        )
        assert hasattr(kwargs["drift_checker"], "run"), (
            "drift_checker must expose .run() coroutine"
        )
