"""__main__ tests: SIGTERM/SIGINT handler wiring + sdnotify READY=1 (D-05)."""
from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_mocks():
    """Returns (orchestrator, picks_bot, ops_bot, sd_notifier) all MagicMock."""
    orch = MagicMock()
    orch.start = MagicMock()
    orch.shutdown = MagicMock()  # SYNC

    picks_bot = MagicMock()
    picks_bot.start = AsyncMock()
    picks_bot.shutdown = AsyncMock()
    ops_bot = MagicMock()
    ops_bot.start = AsyncMock()
    ops_bot.shutdown = AsyncMock()

    sd = MagicMock()
    sd.notify = MagicMock()
    return orch, picks_bot, ops_bot, sd


@pytest.mark.asyncio
async def test_sigterm_triggers_shutdown(monkeypatch):
    """SIGTERM signal triggers _shutdown which calls orch.shutdown + bot shutdowns."""
    from bip.production import __main__ as main_module

    orch, picks_bot, ops_bot, sd = _make_mocks()
    fake_settings = MagicMock()
    fake_settings.heartbeat_file_path = "/tmp/heartbeat"
    fake_settings.telegram_ops_channel_id = "-100xxx"

    monkeypatch.setattr(main_module, "Settings", lambda: fake_settings)
    monkeypatch.setattr(
        main_module, "build_orchestrator",
        lambda s: (orch, picks_bot, ops_bot, sd),
    )

    # Run main as a task so we can deliver a signal.
    task = asyncio.create_task(main_module.main())
    await asyncio.sleep(0.05)  # let main register handlers + signal READY

    # Manually invoke the shutdown coroutine (simulating signal delivery).
    stop_event = asyncio.Event()
    await main_module._shutdown(signal.SIGTERM, stop_event, orch, picks_bot, ops_bot)

    orch.shutdown.assert_called_once()
    picks_bot.shutdown.assert_awaited_once()
    ops_bot.shutdown.assert_awaited_once()
    assert stop_event.is_set()

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, BaseException):
        pass


@pytest.mark.asyncio
async def test_sigint_triggers_shutdown():
    """_shutdown logs the signal name (SIGTERM vs SIGINT)."""
    from bip.production import __main__ as main_module

    orch, picks_bot, ops_bot, sd = _make_mocks()
    stop_event = asyncio.Event()
    await main_module._shutdown(signal.SIGINT, stop_event, orch, picks_bot, ops_bot)

    orch.shutdown.assert_called_once()
    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_sdnotify_ready_called_after_start(monkeypatch):
    """sd_notifier.notify('READY=1') is called AFTER orchestrator.start (RESEARCH §Pattern 1)."""
    from bip.production import __main__ as main_module

    orch, picks_bot, ops_bot, sd = _make_mocks()

    call_order: list[str] = []
    orch.start = MagicMock(side_effect=lambda: call_order.append("orch_start"))
    sd.notify = MagicMock(side_effect=lambda msg: call_order.append(f"sd_notify:{msg}"))

    fake_settings = MagicMock()
    fake_settings.heartbeat_file_path = "/tmp/heartbeat"
    fake_settings.telegram_ops_channel_id = "-100xxx"

    monkeypatch.setattr(main_module, "Settings", lambda: fake_settings)
    monkeypatch.setattr(
        main_module, "build_orchestrator",
        lambda s: (orch, picks_bot, ops_bot, sd),
    )

    task = asyncio.create_task(main_module.main())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, BaseException):
        pass

    # orch_start must come BEFORE sd_notify:READY=1
    assert "sd_notify:READY=1" in call_order
    orch_idx = call_order.index("orch_start")
    sd_idx = call_order.index("sd_notify:READY=1")
    assert orch_idx < sd_idx
