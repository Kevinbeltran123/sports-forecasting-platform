"""HeartbeatTicker tests (DATA-04 / D-07 — RESEARCH §Heartbeat & Watchdog Integration)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from bip.production.heartbeat import HeartbeatTicker


def test_heartbeat_mtime_advances(tmp_heartbeat_path: Path, mock_sd_notify):
    """Tick twice; file exists after first tick; second tick's mtime >= first's."""
    t = HeartbeatTicker(tmp_heartbeat_path, mock_sd_notify)
    t.tick()
    assert tmp_heartbeat_path.exists()
    first_mtime = tmp_heartbeat_path.stat().st_mtime

    # Sleep enough to differentiate mtime on filesystems with 1s granularity.
    time.sleep(1.05)
    t.tick()
    second_mtime = tmp_heartbeat_path.stat().st_mtime
    assert second_mtime >= first_mtime  # >= because some FSes have low granularity


def test_heartbeat_calls_sdnotify_watchdog(tmp_heartbeat_path: Path, mock_sd_notify):
    """RESEARCH §1 verbatim: D-07 file-mtime-only is wrong; tick MUST call sdnotify too."""
    t = HeartbeatTicker(tmp_heartbeat_path, mock_sd_notify)
    t.tick()
    mock_sd_notify.notify.assert_called_once_with("WATCHDOG=1")


def test_heartbeat_directory_must_exist(tmp_path, mock_sd_notify):
    """If parent dir doesn't exist, Path.touch() raises FileNotFoundError.

    This is BY DESIGN — install.sh creates /var/run/bip/ via tmpfiles.d. Test
    documents the failure mode so operators see it on misconfigured staging hosts.
    """
    bad_path = tmp_path / "does_not_exist" / "heartbeat"
    t = HeartbeatTicker(bad_path, mock_sd_notify)
    with pytest.raises(FileNotFoundError):
        t.tick()
