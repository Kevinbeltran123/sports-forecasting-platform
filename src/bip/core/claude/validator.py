"""Claude Role C validator (CLAUDE-01).

D-05/D-06: AsyncAnthropic + strict tool_use guarantees verdict schema conformance via
grammar-constrained sampling. tool_choice is FORCED to validate_pick — Pitfall 3
forbids ever branching it (would invalidate prompt cache).

D-07 retry semantics: hand-rolled 2-attempt loop with 60s sleep. Returns None after
two failures so caller (PickEngine) can persist status=filtered, reason_code='claude_api_unavailable'.
PATTERNS.md drift risk #11: NEVER wrap this in @tenacity.retry — wait-60s-then-retry differs
from exponential backoff. SDK-level retry (max_retries=2) handles 429/500/529 internally.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from anthropic import APIError, AsyncAnthropic, RateLimitError
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ClaudeVerdict(BaseModel):
    """D-06: tool_use response schema."""

    # D-01 broadens to allow SKIPPED — emitted ONLY by PickEngine local construction
    # when Claude API is down (claude_failure_mode='skip'). The Claude API tool_use
    # response itself never produces SKIPPED (VALIDATE_PICK_TOOL.input_schema.enum
    # remains CONFIRM/FLAG/REJECT only).
    verdict: str = Field(..., pattern="^(CONFIRM|FLAG|REJECT|SKIPPED)$")
    reason_code: str
    reasoning: str
    summary: str = Field(..., max_length=120)


VALIDATE_PICK_TOOL: dict[str, Any] = {
    "name": "validate_pick",
    "description": (
        "Emit the validator verdict for the proposed pick. "
        "verdict=CONFIRM means safe to send. "
        "verdict=FLAG means send with warning. "
        "verdict=REJECT means do not send."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["CONFIRM", "FLAG", "REJECT"]},
            "reason_code": {"type": "string"},
            "reasoning": {"type": "string"},
            "summary": {"type": "string", "maxLength": 120},
        },
        "required": ["verdict", "reason_code", "reasoning", "summary"],
        "additionalProperties": False,
    },
}

_TOOL_CHOICE: dict[str, str] = {"type": "tool", "name": "validate_pick"}


_SYSTEM_INSTRUCTIONS = (
    "You are Role C, the pick validator for a sports-betting pipeline. "
    "Validate the proposed pick against the historical learnings provided. "
    "Output ONLY via the validate_pick tool. Reason in English."
)


class ClaudeValidator:
    """Async wrapper around AsyncAnthropic for Role C validation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        learnings_text: str,
        learnings_sha: str,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key, max_retries=2)
        self._model = model
        self._learnings_text = learnings_text
        self._learnings_sha = learnings_sha

    def _build_system_blocks(self) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": _SYSTEM_INSTRUCTIONS},
            {
                "type": "text",
                "text": f"<!-- learnings_sha={self._learnings_sha} -->\n\n{self._learnings_text}",
                "cache_control": {"type": "ephemeral"},
            },
        ]

    async def validate(
        self,
        pick_summary: str,
        curated_signals: str,
    ) -> ClaudeVerdict | None:
        for attempt in range(2):
            try:
                msg = await self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=self._build_system_blocks(),
                    tools=[VALIDATE_PICK_TOOL],
                    tool_choice=_TOOL_CHOICE,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"PICK:\n{pick_summary}\n\n"
                            f"CURATED SIGNALS:\n{curated_signals}\n\n"
                            "Validate against learnings and emit your verdict."
                        ),
                    }],
                )
                tool_block = next(c for c in msg.content if c.type == "tool_use")
                logger.info(
                    "claude_validator_success",
                    attempt=attempt,
                    cache_read=getattr(msg.usage, "cache_read_input_tokens", 0),
                    cache_write=getattr(msg.usage, "cache_creation_input_tokens", 0),
                    learnings_sha=self._learnings_sha,
                )
                return ClaudeVerdict(**tool_block.input)
            except (RateLimitError, APIError) as exc:
                logger.warning(
                    "claude_validator_failed",
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                if attempt == 0:
                    await asyncio.sleep(60)

        logger.error(
            "claude_validator_unavailable",
            attempts=2,
            learnings_sha=self._learnings_sha,
        )
        return None
