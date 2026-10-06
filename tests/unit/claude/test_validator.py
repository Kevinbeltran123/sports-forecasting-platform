"""ClaudeVerdict Literal SKIPPED test (BLOCKER 4 — D-01).

Phase 3 ClaudeValidator tool_use response uses Literal['CONFIRM','FLAG','REJECT'];
D-01 (skip mode) adds 'SKIPPED' to the accepted set on local construction.
"""
from __future__ import annotations


def test_claude_verdict_accepts_skipped():
    """D-01: PickEngine constructs ClaudeVerdict(verdict='SKIPPED', ...) when Claude API is down.

    The model MUST accept this value. The API itself never emits SKIPPED — it's
    constructed locally in src/bip/core/picks/engine.py:claude_failure_mode='skip'.
    """
    from bip.core.claude.validator import ClaudeVerdict
    v = ClaudeVerdict(
        verdict="SKIPPED",
        reason_code="claude_api_unavailable",
        reasoning="",
        summary="",
    )
    assert v.verdict == "SKIPPED"
