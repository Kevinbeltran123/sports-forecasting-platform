"""Commentary-narrative event extraction for live-match adversarial gating.

Day-1 (2026-05-10) Phase 1 coverage audit confirmed that ~94% of fixtures
emit Sportmonks `raw.comments` with structured per-event narrative. The
predictor consumes structured events (goals, cards, subs) but ignores
commentary text — yet commentary contains uncertainty-creating events
that affect betting outcomes:

  • VAR_CHECK         — review in progress; market in flux
  • GOAL_DISALLOWED   — VAR ruling against a goal; opposite-side bias
  • RED_CARD          — sometimes appears here before structured events
  • PENALTY_AWARDED   — game-state shift; high goal probability
  • INJURY_DELAY      — flow disruption; possession/pressure stats stale

This module:
  • Parses `raw.comments` into a structured `CommentaryEvent` list
  • Filters out high-frequency noise tokens (yellow cards, fouls, corners)
    that don't shift betting outcomes meaningfully
  • Exposes `recent_event` query: was there a {VAR, RED, INJURY, ...}
    event within the last N minutes of game time?

`ValueDetector` consumes this via a new `enforce_commentary_cooloff`
gate that drops picks emitted within a configurable cool-off window
of disruptive commentary events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class CommentaryEventType(str, Enum):
    """Types of events extracted from commentary narrative."""

    VAR_CHECK = "var_check"
    GOAL_DISALLOWED = "goal_disallowed"
    RED_CARD = "red_card"
    PENALTY_AWARDED = "penalty_awarded"
    INJURY_DELAY = "injury_delay"


# Compiled regex patterns. Order matters: more-specific patterns first.
# GOAL_DISALLOWED includes "VAR review" text so we check it BEFORE VAR_CHECK.
_PATTERNS: list[tuple[CommentaryEventType, re.Pattern[str]]] = [
    (
        CommentaryEventType.GOAL_DISALLOWED,
        re.compile(r"disallowed|ruled out", re.I),
    ),
    (
        CommentaryEventType.PENALTY_AWARDED,
        re.compile(r"penalty (awarded|scored)|awarded a penalty", re.I),
    ),
    (
        CommentaryEventType.RED_CARD,
        re.compile(r"\bred card\b|sent off|second yellow", re.I),
    ),
    (
        CommentaryEventType.INJURY_DELAY,
        re.compile(r"\binjury|\btreatment\b|delay in the match", re.I),
    ),
    (
        CommentaryEventType.VAR_CHECK,
        re.compile(r"\bVAR\b", re.I),
    ),
]


@dataclass(frozen=True)
class CommentaryEvent:
    """A single uncertainty-creating event extracted from commentary."""

    minute: int
    extra_minute: int  # 0 when no stoppage offset
    event_type: CommentaryEventType
    is_important: bool
    text: str  # original snippet, for audit

    @property
    def absolute_minute(self) -> int:
        """Effective game minute accounting for stoppage offset."""
        return self.minute + (self.extra_minute or 0)


def extract_events(comments: Iterable[dict]) -> list[CommentaryEvent]:
    """Parse Sportmonks comment dicts into typed CommentaryEvents.

    Each comment dict has keys: comment (text), minute, extra_minute,
    is_goal, is_important, order. We scan the text against the patterns
    in priority order; first match wins. Non-matching comments are
    dropped (they're routine "shot/foul/corner" narration).
    """
    events: list[CommentaryEvent] = []
    for c in comments:
        text = c.get("comment") or ""
        if not text:
            continue
        for ev_type, pat in _PATTERNS:
            if pat.search(text):
                minute = c.get("minute")
                if minute is None:
                    continue
                events.append(
                    CommentaryEvent(
                        minute=int(minute),
                        extra_minute=int(c.get("extra_minute") or 0),
                        event_type=ev_type,
                        is_important=bool(c.get("is_important", False)),
                        text=text,
                    )
                )
                break  # first match wins
    # Sort ascending by absolute game minute (some comments arrive out of order)
    events.sort(key=lambda e: e.absolute_minute)
    return events


# Default cool-off windows per event type (minutes of GAME time).
# Calibrated to be conservatively short so we don't lose too many picks
# during a typical 10-15-second VAR check, but long enough to ride out
# the actual disruption window.
DEFAULT_COOLOFF_MINUTES: dict[CommentaryEventType, int] = {
    CommentaryEventType.VAR_CHECK: 3,
    CommentaryEventType.GOAL_DISALLOWED: 3,
    CommentaryEventType.RED_CARD: 2,
    CommentaryEventType.PENALTY_AWARDED: 2,
    CommentaryEventType.INJURY_DELAY: 1,
}


def recent_event(
    events: Iterable[CommentaryEvent],
    *,
    current_minute: int,
    cooloff: dict[CommentaryEventType, int] | None = None,
) -> CommentaryEvent | None:
    """Return the most recent event whose cool-off window covers ``current_minute``.

    If no event is within its window, returns None.
    Used by ValueDetector to gate picks during commentary disruptions.
    """
    if cooloff is None:
        cooloff = DEFAULT_COOLOFF_MINUTES
    most_recent: CommentaryEvent | None = None
    for e in events:
        window = cooloff.get(e.event_type, 0)
        if window <= 0:
            continue
        if e.absolute_minute <= current_minute <= e.absolute_minute + window:
            if most_recent is None or e.absolute_minute > most_recent.absolute_minute:
                most_recent = e
    return most_recent
