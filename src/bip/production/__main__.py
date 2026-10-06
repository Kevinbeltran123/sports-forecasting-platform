"""Entry point for `python -m bip.production` (D-05).

systemd unit ExecStart points here. SIGTERM handled via asyncio signal handler
fanning out to scheduler.shutdown(wait=True) + telegram_bot.shutdown() per
RESEARCH §Pitfall 2 (in-flight async tasks must drain).

T-4-06 mitigation: NEVER pass `settings` instance to logger; only specific
non-secret attributes. Verify via `git grep "logger.*settings\\b" src/bip/production/`.
"""
from __future__ import annotations

import asyncio
import signal

import structlog

from bip.core.settings import Settings
from bip.production.builder import build_orchestrator

logger = structlog.get_logger(__name__)


async def main() -> None:
    """Asyncio main loop — run by `asyncio.run(main())` at module bottom."""
    settings = Settings()  # raises ValidationError on missing required env

    orchestrator, picks_bot, ops_bot, sd_notifier = build_orchestrator(settings)

    await picks_bot.start()
    await ops_bot.start()
    orchestrator.start()  # registers all scheduled jobs (Phase 1-3 + Phase 4)

    sd_notifier.notify("READY=1")  # required for systemd Type=notify
    logger.info(
        "production_started",
        heartbeat_path=settings.heartbeat_file_path,
        ops_channel_configured=bool(settings.telegram_ops_channel_id),
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Pitfall 6: loop.add_signal_handler is POSIX-only. Skip silently on Windows
    # (this codebase deploys to Hetzner Linux; tests run on macOS where it works).
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(
                    _shutdown(s, stop_event, orchestrator, picks_bot, ops_bot)
                ),
            )
        except NotImplementedError:
            logger.warning("signal_handler_unavailable", signal=sig.name)

    await stop_event.wait()


async def _shutdown(sig, stop_event, orch, picks_bot, ops_bot) -> None:
    """Graceful shutdown: scheduler (sync) → bots (async) → set stop_event.

    orch.shutdown() is SYNC (calls scheduler.shutdown(wait=True)). DO NOT await
    it. picks_bot.shutdown() and ops_bot.shutdown() are async.
    """
    logger.info("shutdown_started", signal=sig.name)
    orch.shutdown()  # sync
    await picks_bot.shutdown()
    await ops_bot.shutdown()
    stop_event.set()
    logger.info("shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
