"""Tests for WC2026 pattern alert formatting + dispatch."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_pattern_alerts import (
    LIVE_ALERT_RULES,
    format_pattern_pick_alert,
    send_pattern_pick_alert,
    should_alert_live,
)
from bip.evaluation.live.engine_v3.wc2026_pattern_runner import PatternPick


def _pick(rule_id: str = "A1") -> PatternPick:
    return PatternPick(
        fixture_id="wc2026_grp_000",
        timestamp_utc=datetime(2026, 6, 11, 0, 50, tzinfo=timezone.utc),
        rule_id=rule_id,
        home_team="Mexico",
        away_team="South Africa",
        tournament_slug="world_cup_2026",
        phase="group",
        market_family=MarketFamily.GOALS,
        direction="under",
        decimal_odds=1.25,
        max_kelly=0.25,
        reason="HT 0-0 + U2.5 odds 1.250 ≤ 1.30",
        breakeven_odds=1.239,
    )


class TestFormat:
    def test_includes_rule_id(self):
        out = format_pattern_pick_alert(_pick("A1"))
        assert "PATTERN RULE A1" in out

    def test_no_emojis(self):
        out = format_pattern_pick_alert(_pick("A1"))
        # No emoji range — project convention.
        for ch in out:
            assert ord(ch) < 0x1F000, f"emoji-range char {ch!r} ({hex(ord(ch))})"

    def test_teams_escaped(self):
        p = _pick("A1")
        # Inject HTML-unsafe team name
        p_unsafe = PatternPick(**{**p.__dict__, "home_team": "Brazil <script>"})
        out = format_pattern_pick_alert(p_unsafe)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_includes_breakeven_when_present(self):
        out = format_pattern_pick_alert(_pick("A1"))
        assert "Breakeven" in out
        assert "1.239" in out

    def test_omits_breakeven_when_absent(self):
        base = _pick("A2")
        no_be = PatternPick(**{**base.__dict__, "breakeven_odds": None})
        out = format_pattern_pick_alert(no_be)
        assert "Breakeven" not in out

    def test_includes_kelly_percentage(self):
        out = format_pattern_pick_alert(_pick("A1"))
        assert "25%" in out


class TestShouldAlertLive:
    def test_a1_is_live(self):
        assert should_alert_live(_pick("A1")) is True

    @pytest.mark.parametrize("rule_id", ["A2", "A3", "A4"])
    def test_shadow_rules_default_silent(self, rule_id):
        assert should_alert_live(_pick(rule_id)) is False

    @pytest.mark.parametrize("rule_id", ["A2", "A3"])
    def test_force_promotes_shadow(self, rule_id):
        assert should_alert_live(_pick(rule_id), force=True) is True

    def test_live_alert_rules_frozen(self):
        assert isinstance(LIVE_ALERT_RULES, frozenset)
        assert LIVE_ALERT_RULES == {"A1"}


class _FakeBot:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)


class TestSendPatternPickAlert:
    @pytest.mark.asyncio
    async def test_sends_for_a1(self):
        bot = _FakeBot()
        sent = await send_pattern_pick_alert(bot, _pick("A1"), chat_id=42)
        assert sent is True
        assert len(bot.sent) == 1
        msg = bot.sent[0]
        assert msg["chat_id"] == 42
        assert msg["parse_mode"] == "HTML"
        assert "PATTERN RULE A1" in msg["text"]

    @pytest.mark.asyncio
    async def test_skips_a2_without_force(self):
        bot = _FakeBot()
        sent = await send_pattern_pick_alert(bot, _pick("A2"), chat_id=42)
        assert sent is False
        assert bot.sent == []

    @pytest.mark.asyncio
    async def test_force_sends_a3(self):
        bot = _FakeBot()
        sent = await send_pattern_pick_alert(
            bot, _pick("A3"), chat_id=42, force=True
        )
        assert sent is True
        assert "PATTERN RULE A3" in bot.sent[0]["text"]

    @pytest.mark.asyncio
    async def test_passes_disable_notification(self):
        bot = _FakeBot()
        await send_pattern_pick_alert(
            bot, _pick("A1"), chat_id=42, disable_notification=True
        )
        assert bot.sent[0]["disable_notification"] is True
