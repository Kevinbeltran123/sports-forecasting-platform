"""Two-channel TelegramBot routing (D-03 / T-4-01 mitigation).

Design: D-03 Option A — two TelegramBot instances, each bound to its channel_id at
construction. Verifies that the design prevents wrong-channel routing by
construction: each bot stores its channel_id at __init__ time and send_html uses
self._channel_id, never a runtime parameter.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _patch_application(mock_builder: MagicMock) -> MagicMock:
    """Wire ApplicationBuilder().token().rate_limiter().build() to a MagicMock app."""
    app = MagicMock()
    app.bot.send_message = AsyncMock()
    builder_chain = MagicMock()
    builder_chain.token.return_value.rate_limiter.return_value.build.return_value = app
    mock_builder.return_value = builder_chain
    return app


def test_two_separate_TelegramBot_instances_constructed():
    """Two TelegramBot(token, channel_id) instances coexist with different channel_ids."""
    from bip.core.telegram.bot import TelegramBot

    with patch("bip.core.telegram.bot.ApplicationBuilder") as mock_builder:
        _patch_application(mock_builder)
        picks_bot = TelegramBot(token="test-token", channel_id="-1001234567890")
        ops_bot = TelegramBot(token="test-token", channel_id="-1009876543210")

        assert picks_bot._channel_id == "-1001234567890"
        assert ops_bot._channel_id == "-1009876543210"
        # Two TelegramBot constructions → two ApplicationBuilder() calls
        assert mock_builder.call_count == 2


def test_picks_alert_routes_to_picks_channel():
    """Calling picks_bot.send_html(text) sends to the picks channel — not ops."""
    from bip.core.telegram.bot import TelegramBot

    with patch("bip.core.telegram.bot.ApplicationBuilder") as mock_builder:
        app = _patch_application(mock_builder)
        picks_bot = TelegramBot(token="test-token", channel_id="-1001111111111")

        asyncio.run(picks_bot.send_html("test pick"))
        call_kwargs = app.bot.send_message.await_args.kwargs
        # send_html casts channel_id to int before passing to chat_id
        assert call_kwargs.get("chat_id") == -1001111111111


def test_ops_alert_routes_to_ops_channel():
    """Calling ops_bot.send_html(text) sends to the ops channel — not picks."""
    from bip.core.telegram.bot import TelegramBot

    with patch("bip.core.telegram.bot.ApplicationBuilder") as mock_builder:
        app = _patch_application(mock_builder)
        ops_bot = TelegramBot(token="test-token", channel_id="-1002222222222")

        asyncio.run(ops_bot.send_html("ops alert"))
        call_kwargs = app.bot.send_message.await_args.kwargs
        assert call_kwargs.get("chat_id") == -1002222222222
