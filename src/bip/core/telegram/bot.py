"""Telegram bot wrapper (v2).

V1 lives here in send-only form (``send_html``) — used by every existing
caller, untouched in v2. Interactive features (callback queries, slash
commands) are opt-in via ``enable_interactivity()``, which adds a
filtered long-polling updater on top of the same Application.

D-13: Lives in the same asyncio loop as the watcher (in-process). AIORateLimiter
handles flood control (requires the [rate-limiter] extra in pyproject.toml).

Polling stance: when interactivity is enabled, we use ``allowed_updates=
["callback_query", "message"]`` and a user-ID filter — this is NOT a
custom getUpdates loop, it is PTB's built-in Updater scoped to one
operator. See design memo §A for the rationale (webhook vs polling
on a single VPS).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from telegram import Bot
from telegram.constants import ParseMode
from telegram.ext import AIORateLimiter, Application, ApplicationBuilder

from bip.core.errors import TelegramError

logger = structlog.get_logger(__name__)


class TelegramBot:
    """Outbound-first Telegram bot. Interactivity opt-in."""

    def __init__(self, token: str, channel_id: str) -> None:
        if not token:
            raise TelegramError("telegram_bot_token is empty — set TELEGRAM_BOT_TOKEN in .env")
        if not channel_id:
            raise TelegramError("telegram_channel_id is empty — set TELEGRAM_CHANNEL_ID in .env")

        self._channel_id = channel_id
        self._app: Application = (
            ApplicationBuilder()
            .token(token)
            .rate_limiter(AIORateLimiter(max_retries=3))
            .build()
        )
        self._polling_enabled = False

    async def start(self) -> None:
        await self._app.initialize()
        await self._app.start()
        logger.info("telegram_bot_started", channel_id=self._channel_id)

    async def shutdown(self) -> None:
        # Order matters: stop the Updater (if running) BEFORE stopping the
        # Application, otherwise pending updates may panic on a half-torn
        # event loop.
        if self._polling_enabled and self._app.updater:
            try:
                await self._app.updater.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("telegram_updater_stop_failed", err=str(exc))
            self._polling_enabled = False
        await self._app.stop()
        await self._app.shutdown()
        logger.info("telegram_bot_stopped")

    @property
    def bot(self) -> Bot:
        return self._app.bot

    @property
    def app(self) -> Application:
        return self._app

    @property
    def channel_id(self) -> str:
        return self._channel_id

    async def send_html(
        self,
        text: str,
        *,
        chat_id: str | int | None = None,
        reply_to_message_id: int | None = None,
        reply_markup: object | None = None,
        disable_notification: bool = False,
    ) -> int:
        """Send an HTML-parsed message. Returns the sent message_id.

        ``chat_id`` defaults to the bot's primary channel. ``reply_markup``
        accepts an InlineKeyboardMarkup or None. ``reply_to_message_id``
        threads outcomes under the original pick alert.
        """
        target = chat_id if chat_id is not None else self._channel_id
        msg = await self.bot.send_message(
            chat_id=int(target),
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_to_message_id=reply_to_message_id,
            reply_markup=reply_markup,
            disable_notification=disable_notification,
        )
        logger.info(
            "telegram_send_success",
            channel_id=str(target), length=len(text),
            message_id=msg.message_id,
        )
        return msg.message_id

    async def edit_html(
        self,
        *,
        chat_id: str | int,
        message_id: int,
        text: str,
        reply_markup: object | None = None,
    ) -> None:
        """Edit an existing message's text (HTML parse mode)."""
        await self.bot.edit_message_text(
            chat_id=int(chat_id),
            message_id=message_id,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )
        logger.info(
            "telegram_edit_success",
            chat_id=str(chat_id), message_id=message_id,
        )

    # ── Interactivity opt-in ────────────────────────────────────────────

    async def enable_interactivity(
        self,
        *,
        state: object,                   # TelegramState
        db_path: Path,
        operator_user_id: int,
    ) -> None:
        """Register callback/command handlers and start filtered polling.

        Must be called AFTER ``start()``. Idempotent — safe to call once
        per bot lifetime.
        """
        if self._polling_enabled:
            return
        if operator_user_id <= 0:
            logger.warning(
                "interactivity_disabled_no_operator_user_id "
                "(set TELEGRAM_OPERATOR_USER_ID in .env)"
            )
            return
        # Local import — handlers.py imports TelegramState which would
        # create a circular path if top-leveled.
        from bip.core.telegram.handlers import register_handlers

        register_handlers(
            self._app, state=state, db_path=db_path,
            operator_user_id=operator_user_id,
        )
        await self._app.updater.start_polling(
            allowed_updates=["callback_query", "message"],
            drop_pending_updates=True,
        )
        self._polling_enabled = True
        logger.info(
            "telegram_interactivity_enabled",
            operator_user_id=operator_user_id,
        )
