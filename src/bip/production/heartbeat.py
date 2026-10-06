"""HeartbeatTicker (D-07).

Touches Settings.heartbeat_file_path every 5 min AND sends WATCHDOG=1 on the
systemd notify socket (D-06 with Type=notify + WatchdogSec=600).

The file mtime is for human/external observers (`stat -c %Y heartbeat`); the
sdnotify call is what systemd's WatchdogSec= actually consumes (RESEARCH
§Heartbeat & Watchdog Integration — D-07's file-mtime-only design is incorrect).

T-4-04 mitigation: /var/run/bip/ is mode 0750 owned by bip:bip (04-07-PLAN
install.sh enforces). Path.touch() under that directory is safe from symlink
race because other users cannot create files in it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


class _Notifier(Protocol):
    """sdnotify.SystemdNotifier shape — duck-typed for testability."""

    def notify(self, msg: str) -> bool: ...


class HeartbeatTicker:
    """File-touch + systemd watchdog notification for a process supervised by systemd.

    Registered as APScheduler IntervalTrigger(minutes=5) job by Wave 4 orchestrator
    extension. Sync method — fine in AsyncIOExecutor (Pitfall 7: <1ms work).
    """

    def __init__(self, path: str | Path, notifier: _Notifier) -> None:
        self._path = Path(path)
        self._notifier = notifier  # injected — silently no-ops on macOS when NOTIFY_SOCKET unset

    def tick(self) -> None:
        """Run every 5 min via APScheduler. Updates file mtime AND notifies systemd."""
        self._path.touch()
        self._notifier.notify("WATCHDOG=1")
        # Pitfall 7: debug-level — INFO would flood journal at 5-min cadence x 24h x 7d.
        logger.debug("heartbeat", path=str(self._path))
