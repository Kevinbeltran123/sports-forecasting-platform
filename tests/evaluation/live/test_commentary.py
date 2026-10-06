"""Tests for commentary-narrative event extraction and cool-off gating."""

from __future__ import annotations

import pytest

from bip.evaluation.live.commentary import (
    CommentaryEvent,
    CommentaryEventType,
    DEFAULT_COOLOFF_MINUTES,
    extract_events,
    recent_event,
)


def _c(text: str, minute: int = 30, extra: int = 0, important: bool = False) -> dict:
    """Build a Sportmonks comment dict for testing."""
    return {
        "comment": text, "minute": minute, "extra_minute": extra,
        "is_goal": False, "is_important": important,
    }


class TestExtraction:
    def test_var_check(self):
        evs = extract_events([_c("VAR is checking a possible foul", minute=42)])
        assert len(evs) == 1
        assert evs[0].event_type == CommentaryEventType.VAR_CHECK
        assert evs[0].minute == 42

    def test_red_card_variants(self):
        comments = [
            _c("Jorrit Hendrix receives a red card.", minute=66),
            _c("Player sent off after dangerous tackle.", minute=78),
            _c("Second yellow for Vinicius — straight off.", minute=82),
        ]
        evs = extract_events(comments)
        assert len(evs) == 3
        assert all(e.event_type == CommentaryEventType.RED_CARD for e in evs)

    def test_injury_delay(self):
        evs = extract_events([
            _c("There is a delay in the match due to an injury to Ben White.", minute=15),
            _c("Player receives treatment from medical staff.", minute=22),
        ])
        assert len(evs) == 2
        assert all(e.event_type == CommentaryEventType.INJURY_DELAY for e in evs)

    def test_penalty_awarded(self):
        evs = extract_events([
            _c("Penalty awarded to Real Madrid.", minute=55),
            _c("VAR decision confirms a penalty awarded to Asier Villalibre.", minute=72),
        ])
        # Both match penalty_awarded; VAR_CHECK is also in the second text
        # but PENALTY_AWARDED has higher priority (listed first in _PATTERNS).
        assert len(evs) == 2
        assert all(e.event_type == CommentaryEventType.PENALTY_AWARDED for e in evs)

    def test_goal_disallowed(self):
        evs = extract_events([
            _c("VAR review results in a disallowed goal due to offside.", minute=44),
            _c("Goal ruled out for handball.", minute=51),
        ])
        assert len(evs) == 2
        assert all(e.event_type == CommentaryEventType.GOAL_DISALLOWED for e in evs)

    def test_yellow_card_does_not_match(self):
        # Yellow cards are too frequent (68% of Day-1 fixtures) — not gated.
        evs = extract_events([_c("Yellow card for player.", minute=30)])
        assert evs == []

    def test_routine_comment_not_extracted(self):
        # Most narration is shot/foul/corner — should produce no events.
        comments = [
            _c("Player took a right-footed shot from outside the box.", minute=20),
            _c("Corner kick for the home team.", minute=30),
            _c("Foul committed by midfielder.", minute=45),
        ]
        assert extract_events(comments) == []

    def test_extra_minute_used_in_absolute(self):
        evs = extract_events([_c("VAR is checking.", minute=45, extra=3)])
        assert evs[0].absolute_minute == 48

    def test_events_sorted_by_minute(self):
        comments = [
            _c("Player injury delay 5", minute=78),
            _c("VAR is checking handball", minute=22),
            _c("Penalty awarded", minute=50),
        ]
        evs = extract_events(comments)
        minutes = [e.absolute_minute for e in evs]
        assert minutes == sorted(minutes)

    def test_missing_minute_skipped(self):
        evs = extract_events([{"comment": "VAR is checking", "minute": None}])
        assert evs == []

    def test_priority_disallowed_over_var(self):
        # "VAR review results in a disallowed goal" matches GOAL_DISALLOWED
        # which is checked first in _PATTERNS; should NOT classify as VAR_CHECK.
        evs = extract_events([
            _c("VAR review results in a disallowed goal due to offside.", minute=22),
        ])
        assert evs[0].event_type == CommentaryEventType.GOAL_DISALLOWED


class TestRecentEvent:
    def test_event_within_window(self):
        ev = CommentaryEvent(
            minute=30, extra_minute=0,
            event_type=CommentaryEventType.VAR_CHECK,
            is_important=True, text="VAR",
        )
        # 3-min VAR cool-off: events at min 30 cover [30, 33]
        assert recent_event([ev], current_minute=31) is not None
        assert recent_event([ev], current_minute=33) is not None
        assert recent_event([ev], current_minute=34) is None

    def test_event_after_window(self):
        ev = CommentaryEvent(
            minute=20, extra_minute=0,
            event_type=CommentaryEventType.INJURY_DELAY,
            is_important=False, text="Injury",
        )
        # 1-min injury cool-off: event at 20 covers [20, 21]
        assert recent_event([ev], current_minute=22) is None

    def test_most_recent_wins(self):
        evs = [
            CommentaryEvent(20, 0, CommentaryEventType.VAR_CHECK, False, ""),
            CommentaryEvent(22, 0, CommentaryEventType.INJURY_DELAY, False, ""),
        ]
        # At min 22, both VAR (3-min @ 20→[20,23]) and INJURY (1-min @ 22→[22,23]) active
        ev = recent_event(evs, current_minute=22)
        assert ev is not None
        # Most recent wins
        assert ev.absolute_minute == 22
        assert ev.event_type == CommentaryEventType.INJURY_DELAY

    def test_custom_cooloff_dict(self):
        ev = CommentaryEvent(
            minute=10, extra_minute=0,
            event_type=CommentaryEventType.VAR_CHECK,
            is_important=False, text="VAR",
        )
        # Custom: 10-min VAR cool-off
        custom = {CommentaryEventType.VAR_CHECK: 10}
        assert recent_event([ev], current_minute=15, cooloff=custom) is not None
        # Default 3-min would NOT cover minute 15
        assert recent_event([ev], current_minute=15) is None

    def test_zero_cooloff_disables_type(self):
        ev = CommentaryEvent(
            minute=10, extra_minute=0,
            event_type=CommentaryEventType.VAR_CHECK,
            is_important=False, text="",
        )
        custom = {CommentaryEventType.VAR_CHECK: 0}
        assert recent_event([ev], current_minute=10, cooloff=custom) is None

    def test_default_cooloff_values(self):
        # Spot-check the published defaults
        assert DEFAULT_COOLOFF_MINUTES[CommentaryEventType.VAR_CHECK] == 3
        assert DEFAULT_COOLOFF_MINUTES[CommentaryEventType.RED_CARD] == 2
        assert DEFAULT_COOLOFF_MINUTES[CommentaryEventType.INJURY_DELAY] == 1
