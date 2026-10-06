"""GREEN tests for ClaudeValidator (CLAUDE-01 — D-06 strict tool_use, D-07 double-retry)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_canned_message(verdict: str = "CONFIRM"):
    """Mimic anthropic Message with content[0] as tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "verdict": verdict,
        "reason_code": "ok",
        "reasoning": "Looks fine.",
        "summary": "Strong edge against weak defense * good form",
    }
    msg = MagicMock()
    msg.content = [tool_block]
    msg.usage.cache_read_input_tokens = 3500
    msg.usage.cache_creation_input_tokens = 0
    return msg


class TestToolSchema:
    def test_tool_schema_strict(self):
        from bip.core.claude.validator import VALIDATE_PICK_TOOL
        assert VALIDATE_PICK_TOOL["strict"] is True
        schema = VALIDATE_PICK_TOOL["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"verdict", "reason_code", "reasoning", "summary"}
        assert schema["properties"]["verdict"]["enum"] == ["CONFIRM", "FLAG", "REJECT"]
        assert schema["properties"]["summary"].get("maxLength") == 120

    def test_verdict_enum(self):
        from pydantic import ValidationError
        from bip.core.claude.validator import ClaudeVerdict
        ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="x", summary="y")
        ClaudeVerdict(verdict="FLAG", reason_code="ok", reasoning="x", summary="y")
        ClaudeVerdict(verdict="REJECT", reason_code="ok", reasoning="x", summary="y")
        with pytest.raises(ValidationError):
            ClaudeVerdict(verdict="MAYBE", reason_code="ok", reasoning="x", summary="y")

    def test_summary_max_120_chars(self):
        from pydantic import ValidationError
        from bip.core.claude.validator import ClaudeVerdict
        with pytest.raises(ValidationError):
            ClaudeVerdict(verdict="CONFIRM", reason_code="ok", reasoning="x", summary="x" * 121)


class TestRetry:
    @pytest.mark.asyncio
    async def test_failure_double_retry(self, monkeypatch):
        """D-07: APIError twice → asyncio.sleep(60) called once → returns None."""
        from anthropic import APIError
        from bip.core.claude.validator import ClaudeValidator

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        validator = ClaudeValidator(
            api_key="test", model="claude-sonnet-4-6",
            learnings_text="x" * 10000, learnings_sha="deadbeef" * 5,
        )

        api_err = APIError("simulated", request=MagicMock(), body=None)
        validator._client = MagicMock()
        validator._client.messages.create = AsyncMock(side_effect=[api_err, api_err])

        monkeypatch.setattr("bip.core.claude.validator.asyncio.sleep", fake_sleep)

        result = await validator.validate(pick_summary="test pick", curated_signals="signals")

        assert result is None
        assert sleep_calls == [60]
        assert validator._client.messages.create.await_count == 2

    @pytest.mark.asyncio
    async def test_first_attempt_success_no_sleep(self, monkeypatch):
        from bip.core.claude.validator import ClaudeValidator, ClaudeVerdict

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        validator = ClaudeValidator(
            api_key="test", model="claude-sonnet-4-6",
            learnings_text="x" * 10000, learnings_sha="abc",
        )
        validator._client = MagicMock()
        validator._client.messages.create = AsyncMock(return_value=_build_canned_message("CONFIRM"))

        monkeypatch.setattr("bip.core.claude.validator.asyncio.sleep", fake_sleep)

        result = await validator.validate(pick_summary="x", curated_signals="y")

        assert isinstance(result, ClaudeVerdict)
        assert result.verdict == "CONFIRM"
        assert sleep_calls == []


class TestPromptCaching:
    def test_system_has_cache_control(self):
        """Pitfall 4: learnings system block must have cache_control={'type': 'ephemeral'}."""
        from bip.core.claude.validator import ClaudeValidator

        v = ClaudeValidator(
            api_key="test", model="claude-sonnet-4-6",
            learnings_text="x" * 10000, learnings_sha="abc",
        )
        blocks = v._build_system_blocks()
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "text"
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}
        assert "abc" in blocks[1]["text"]

    def test_tool_choice_constant(self):
        """Pitfall 3: tool_choice is a module constant — never branched."""
        import bip.core.claude.validator as v
        src = open(v.__file__).read()
        occurrences = src.count("tool_choice=")
        assert occurrences <= 2
