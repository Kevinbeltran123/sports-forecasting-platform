"""GREEN tests for TelegramBot init / no-polling / rate limiter."""
from __future__ import annotations

import inspect

import pytest


class TestBotInit:
    def test_init_no_polling(self):
        """D-13 + drift risk #8: bot.start() must NOT call updater.start_polling().

        v2 adds opt-in polling via ``enable_interactivity()`` for the
        callback-query/command handlers — that path is explicit and
        gated by an operator user-id. We only enforce the invariant on
        the default start path here.
        """
        from bip.core.telegram.bot import TelegramBot
        TelegramBot(token="test-token", channel_id="-1001234567890")
        src = inspect.getsource(TelegramBot.start)
        assert "start_polling" not in src

    def test_enable_interactivity_opt_in_only(self):
        """enable_interactivity must be a separate method, not invoked by start()."""
        from bip.core.telegram.bot import TelegramBot
        start_src = inspect.getsource(TelegramBot.start)
        assert "enable_interactivity" not in start_src
        assert hasattr(TelegramBot, "enable_interactivity")

    def test_aiorate_limiter_attached(self):
        """Pitfall 2 + Risks 2: AIORateLimiter must be wired (PTB 22.7 stores it on bot)."""
        from telegram.ext import AIORateLimiter

        from bip.core.telegram.bot import TelegramBot
        bot = TelegramBot(token="test-token", channel_id="-1001234567890")
        assert bot._app.bot.rate_limiter is not None
        assert isinstance(bot._app.bot.rate_limiter, AIORateLimiter)

    def test_initialize_then_start(self):
        """Drift risk #7: bot.start() must call initialize() then start()."""
        from bip.core.telegram.bot import TelegramBot
        src = inspect.getsource(TelegramBot.start)
        idx_init = src.find("initialize")
        idx_start = src.find("self._app.start")
        assert idx_init >= 0 and idx_start >= 0 and idx_init < idx_start

    def test_empty_token_raises(self):
        from bip.core.errors import TelegramError
        from bip.core.telegram.bot import TelegramBot
        with pytest.raises(TelegramError, match="telegram_bot_token"):
            TelegramBot(token="", channel_id="-1001234567890")

    def test_empty_channel_id_raises(self):
        from bip.core.errors import TelegramError
        from bip.core.telegram.bot import TelegramBot
        with pytest.raises(TelegramError, match="telegram_channel_id"):
            TelegramBot(token="test-token", channel_id="")

    def test_channel_id_validator_rejects_positive_int(self, monkeypatch):
        """Pitfall 8: TELEGRAM_CHANNEL_ID without -100 prefix raises ValidationError at Settings load."""
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "k")
        monkeypatch.setenv("API_FOOTBALL_KEY", "k")
        monkeypatch.setenv("ODDS_API_KEY", "k")
        monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "12345")
        from bip.core.settings import Settings
        with pytest.raises(Exception):
            Settings(_env_file=None)
