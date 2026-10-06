"""Tests for Phase 3 Settings extensions: 4 new fields + telegram_channel_id validator.

Plan 03-02 Task 2 (TDD RED phase):
  - Settings gains telegram_channel_id, anthropic_api_key, claude_model, max_kelly_fraction (D-13)
  - telegram_channel_id @field_validator rejects non-'-100*' values (Pitfall 8)
  - extra='ignore' allows SUPABASE_DB_PASSWORD in env without raising
"""

import os

import pytest


# Required env vars for Settings to construct (always present in tests)
_REQUIRED = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_KEY": "k",
    "API_FOOTBALL_KEY": "k",
    "ODDS_API_KEY": "k",
}


def _make_settings(monkeypatch, extra_vars: dict | None = None, remove: list | None = None):
    """Helper: set required env vars, apply extras, remove keys, construct Settings."""
    for k, v in _REQUIRED.items():
        monkeypatch.setenv(k, v)
    if extra_vars:
        for k, v in extra_vars.items():
            monkeypatch.setenv(k, v)
    if remove:
        for k in remove:
            monkeypatch.delenv(k, raising=False)
    from importlib import reload
    import bip.core.settings as _mod
    reload(_mod)
    return _mod.Settings(_env_file=None)


class TestSettingsNewFields:
    """Test 1 + Test 3: New fields present with correct defaults."""

    def test_valid_channel_id_accepted(self, monkeypatch):
        s = _make_settings(monkeypatch, {"TELEGRAM_CHANNEL_ID": "-1001234567890"})
        assert s.telegram_channel_id == "-1001234567890"

    def test_default_claude_model(self, monkeypatch):
        s = _make_settings(monkeypatch, remove=["CLAUDE_MODEL"])
        assert s.claude_model == "claude-sonnet-4-6"

    def test_default_max_kelly_fraction(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.max_kelly_fraction == 0.25

    def test_default_telegram_channel_id_empty(self, monkeypatch):
        s = _make_settings(monkeypatch, remove=["TELEGRAM_CHANNEL_ID"])
        assert s.telegram_channel_id == ""

    def test_default_anthropic_api_key_empty(self, monkeypatch):
        s = _make_settings(monkeypatch, remove=["ANTHROPIC_API_KEY"])
        assert s.anthropic_api_key == ""

    def test_anthropic_api_key_loaded_from_env(self, monkeypatch):
        s = _make_settings(monkeypatch, {"ANTHROPIC_API_KEY": "sk-ant-test"})
        assert s.anthropic_api_key == "sk-ant-test"


class TestTelegramChannelIdValidator:
    """Test 2: validator rejects bad channel_id format (Pitfall 8)."""

    def test_no_minus100_prefix_raises(self, monkeypatch):
        from pydantic import ValidationError
        with pytest.raises((ValidationError, Exception)) as exc_info:
            _make_settings(monkeypatch, {"TELEGRAM_CHANNEL_ID": "12345"})
        msg = str(exc_info.value)
        assert "-100" in msg or "telegram_channel_id" in msg

    def test_positive_number_raises(self, monkeypatch):
        from pydantic import ValidationError
        with pytest.raises((ValidationError, Exception)):
            _make_settings(monkeypatch, {"TELEGRAM_CHANNEL_ID": "1001234567890"})

    def test_empty_string_default_allowed(self, monkeypatch):
        """Empty string allowed — CI/test envs construct without secrets."""
        s = _make_settings(monkeypatch, {"TELEGRAM_CHANNEL_ID": ""})
        assert s.telegram_channel_id == ""

    def test_valid_channel_id_minus100_format(self, monkeypatch):
        s = _make_settings(monkeypatch, {"TELEGRAM_CHANNEL_ID": "-1009876543210"})
        assert s.telegram_channel_id == "-1009876543210"


class TestSettingsExtraIgnore:
    """extra='ignore' allows unknown env vars like SUPABASE_DB_PASSWORD."""

    def test_extra_env_var_does_not_raise(self, monkeypatch):
        """SUPABASE_DB_PASSWORD in env should not cause ValidationError."""
        s = _make_settings(monkeypatch, {"SUPABASE_DB_PASSWORD": "supersecret"})
        assert s is not None  # constructed without error


class TestSettingsExistingFieldsPreserved:
    """Test 4: Phase 1+2 fields still present (regression)."""

    def test_existing_fields_present(self, monkeypatch):
        s = _make_settings(monkeypatch)
        assert s.supabase_url == "https://test.supabase.co"
        assert s.supabase_key == "k"
        assert s.parquet_base_path == "data/cache"
        assert s.model_dir == "models"
        assert s.api_football_key == "k"
        assert s.odds_api_key == "k"
        assert s.telegram_bot_token == ""
        assert s.log_level == "INFO"


class TestReconcileTuning:
    """G-MAINT-10 closeout (260503-txj): reconcile knobs are configurable via Settings."""

    def test_reconcile_defaults_match_historical_behavior(self, monkeypatch):
        """Defaults preserve the old 4-retry x 30-minute cadence (no behavior change)."""
        s = _make_settings(monkeypatch)
        assert s.reconcile_max_retries == 4
        assert s.reconcile_retry_minutes == 30

    def test_reconcile_max_retries_overridable_via_env(self, monkeypatch):
        s = _make_settings(monkeypatch, {"RECONCILE_MAX_RETRIES": "8"})
        assert s.reconcile_max_retries == 8

    def test_reconcile_retry_minutes_overridable_via_env(self, monkeypatch):
        s = _make_settings(monkeypatch, {"RECONCILE_RETRY_MINUTES": "15"})
        assert s.reconcile_retry_minutes == 15
