"""Settings Phase 4 tests — D-01, D-03, D-07, D-09, D-14, RESEARCH §3 floor.

Covers: claude_failure_mode default + Literal enforcement, telegram_ops_channel_id
validator (T-4-01 mitigation), all numeric defaults, drift_stdev_floor_pp NEW field.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


def _required_env(monkeypatch) -> None:
    """Set the minimum required env vars for Settings() to construct."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-af")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds")


def test_claude_failure_mode_default_is_filter(monkeypatch):
    _required_env(monkeypatch)
    from bip.core.settings import Settings
    s = Settings()
    assert s.claude_failure_mode == "filter"  # D-01: conservative default


def test_claude_failure_mode_rejects_invalid_value(monkeypatch):
    _required_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_FAILURE_MODE", "bogus")
    from bip.core.settings import Settings
    with pytest.raises(ValidationError):
        Settings()


def test_telegram_ops_channel_id_validator_accepts_minus100_prefix(monkeypatch):
    _required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL_ID", "-1009876543210")
    from bip.core.settings import Settings
    s = Settings()
    assert s.telegram_ops_channel_id == "-1009876543210"


def test_telegram_ops_channel_id_validator_rejects_short_id(monkeypatch):
    _required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL_ID", "-100")  # 4 chars < 8
    from bip.core.settings import Settings
    with pytest.raises(ValidationError) as exc:
        Settings()
    assert "truncated" in str(exc.value) or "8" in str(exc.value)


def test_telegram_ops_channel_id_validator_rejects_no_minus100_prefix(monkeypatch):
    _required_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL_ID", "12345678901")
    from bip.core.settings import Settings
    with pytest.raises(ValidationError) as exc:
        Settings()
    assert "-100" in str(exc.value)


def test_heartbeat_file_path_default(monkeypatch):
    _required_env(monkeypatch)
    from bip.core.settings import Settings
    s = Settings()
    assert s.heartbeat_file_path == "/var/run/bip/heartbeat"  # D-07


def test_clv_trend_alert_threshold_default_1pp(monkeypatch):
    _required_env(monkeypatch)
    from bip.core.settings import Settings
    s = Settings()
    assert s.clv_trend_alert_threshold == 1.0  # D-09


def test_drift_stdev_floor_pp_default_1pp(monkeypatch):
    """RESEARCH §Drift Statistical Method: floor prevents zero-stdev tight gate (Pitfall 4)."""
    _required_env(monkeypatch)
    from bip.core.settings import Settings
    s = Settings()
    assert s.drift_stdev_floor_pp == 1.0


def test_drift_check_min_picks_per_window_default_30(monkeypatch):
    _required_env(monkeypatch)
    from bip.core.settings import Settings
    s = Settings()
    assert s.drift_check_min_picks_per_window == 30


def test_drift_absolute_threshold_pp_default_2pp(monkeypatch):
    _required_env(monkeypatch)
    from bip.core.settings import Settings
    s = Settings()
    assert s.drift_absolute_threshold_pp == 2.0
