"""Protocols for v4 pipeline dependencies.

Defined as typing.Protocol so the orchestrator + workers can be tested
with lightweight mocks without importing real Supabase / Anthropic /
Telegram clients. The real implementations (supabase-py Client,
ClaudeValidator, TelegramSender) are structural-typing-compatible with
these protocols.

Per memory `project_data_blocker`: tests run against mocks; real
integrations land when data collection is unblocked.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupabaseClientProtocol(Protocol):
    """Minimal subset of supabase-py's Client used by v4 pipeline.

    The real `supabase.Client` is fully compatible — its `.table()` returns
    a chainable query builder that ultimately exposes `.execute()`.
    """

    def table(self, name: str) -> Any: ...


@runtime_checkable
class ClaudeValidatorProtocol(Protocol):
    """Minimal contract for Claude Role C validation.

    `validate(pick_summary, curated_signals)` returns either a
    ClaudeVerdict (from bip.core.claude.validator) or None when the
    upstream Claude API is unreachable. The DeliveryWorker treats None
    as a SKIP — see worker.py for the policy.
    """

    async def validate(self, pick_summary: str, curated_signals: str) -> Any: ...


@runtime_checkable
class TelegramSenderProtocol(Protocol):
    """Minimal Telegram sender contract.

    Real TelegramSender in src/bip/core/telegram/ implements this. The
    DeliveryWorker only needs an async send_pick(text) method; the
    payload formatting lives upstream (DeliveryWorker builds the text).
    """

    async def send_pick(self, *, text: str) -> Any: ...


__all__ = [
    "ClaudeValidatorProtocol",
    "SupabaseClientProtocol",
    "TelegramSenderProtocol",
]
