"""Live observer for v3 picks — see what v3 fires in real time without
touching v2's Telegram path.

The dual-write runtime calls ``observer.on_pick(...)`` for each allowed
ShadowPick after the pipeline returns successfully. Observers are
SIDE-EFFECT ONLY — they never raise, never block (best-effort), and
never modify pick state. They exist purely so the operator can WATCH
v3's behaviour during a jornada without flipping any production
switches.

Why an Observer Protocol instead of just printing inside dual_write:

- Lets the operator swap stdout for a log file, a second Telegram
  channel, or a websocket later without touching the runtime.
- Keeps the runtime free of UX decisions (formatting, coloring,
  filtering).
- Easy to wire ZERO observers in test paths.

Wiring (watch.py at startup)::

    from bip.evaluation.live.engine_v3.runtime.live_observer import (
        StdoutObserver,
        build_observer_from_env,
    )

    observer = build_observer_from_env()  # respects V3_LIVE_OBSERVE
    runtime = DualWriteRuntime.from_paths(observer=observer)

Activating::

    export V3_LIVE_OBSERVE=true
    # optional: V3_LIVE_OBSERVE_LOG=/path/to/v3_live.log (stdout by default)

Deactivating: ``unset V3_LIVE_OBSERVE`` (or set to false). Next watch
iteration picks up the change at startup; for a running process you
need to restart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Protocol, TextIO

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.mispricing_window import WindowResult
from bip.evaluation.live.engine_v3.pipeline import ShadowPick


# ──────────────────────────────────────────────────────────────────────
# ANSI helpers — colored terminal output
# ──────────────────────────────────────────────────────────────────────


_RESET = "0"
_BOLD = "1"
_DIM = "2"
_RED = "31"
_GREEN = "32"
_YELLOW = "33"
_BLUE = "34"
_MAGENTA = "35"
_CYAN = "36"
_WHITE = "37"
_GREY = "90"


def _ansi(*codes: str) -> str:
    """Build an ANSI escape sequence. Respects NO_COLOR env convention
    (https://no-color.org): when NO_COLOR is set + non-empty, returns
    empty string so terminals without color support stay clean.
    """
    if os.getenv("NO_COLOR"):
        return ""
    if not codes:
        return ""
    return f"\033[{';'.join(codes)}m"


def _direction_color(direction: str) -> str:
    """Bias color: green for 'something happens', red for 'no-action',
    blue for outcome picks. Helps the operator spot 'Under/No' picks
    (the old v2 bias) at a glance vs over/yes (v3 should diversify)."""
    d = direction.lower()
    if d in {"over", "yes"}:
        return _ansi(_GREEN)
    if d in {"under", "no"}:
        return _ansi(_RED)
    return _ansi(_BLUE)


def _edge_color(edge_pct: float) -> str:
    """Edge gradient: green for healthy, bold-green for high, RED+bold
    for SUSPICIOUSLY high (>15% almost always means mispricing in our
    favor OR calibration error against us — operator should audit)."""
    if edge_pct < 0:
        return _ansi(_DIM, _RED)
    if edge_pct < 3:
        return _ansi(_DIM)
    if edge_pct < 7:
        return _ansi(_GREEN)
    if edge_pct < 15:
        return _ansi(_BOLD, _GREEN)
    return _ansi(_BOLD, _RED)  # suspicious — audit


def _mes_color(mes: float) -> str:
    """MES gradient: yellow at threshold, green when generous."""
    if mes < 0.6:
        return _ansi(_RED)
    if mes < 0.75:
        return _ansi(_YELLOW)
    return _ansi(_BOLD, _GREEN)


_WINDOW_GLYPHS: dict[str, tuple[str, str]] = {
    "hot":        ("●", _ansi(_BOLD, _RED)),
    "optimal":    ("◯", _ansi(_BOLD, _GREEN)),
    "warm":       ("◇", _ansi(_YELLOW)),
    "cold":       ("✕", _ansi(_DIM)),
    "indefinite": ("?", _ansi(_DIM)),
}


def _format_window_label(window: WindowResult | None) -> str:
    if window is None:
        return f"{_ansi(_DIM)}---{_ansi(_RESET)}"
    label_str = (
        window.label.value
        if hasattr(window.label, "value")
        else str(window.label)
    )
    sym, color = _WINDOW_GLYPHS.get(label_str.lower(), ("?", _ansi(_DIM)))
    return f"{color}{sym} {label_str.upper()}{_ansi(_RESET)}"


def _humanize_bet(pick: ShadowPick, gsv: GameStateVector) -> str:
    """Render a human-readable bet descriptor from family + market_id + direction.

    Examples (input → output):
      goals + match_goals_over_2.5     + over    → "GOALS  Over 2.5"
      goals + first_half_goals_under_1.5 + under → "GOALS 1H Under 1.5"
      corners + match_corners_over_9.5 + over    → "CORNERS Over 9.5"
      cards + cards_over_4.5 + over              → "CARDS Over 4.5"
      next_goal + next_goal_home + home          → "NEXT GOAL → Home"
      btts + btts_yes + yes                      → "BTTS Yes"

    Falls back to ``family/direction`` (the old format) if the
    market_id doesn't follow a recognized pattern.
    """
    import re

    market_id = pick.candidate.market_id
    family = pick.full_thesis.prediction.family.value
    direction = pick.full_thesis.prediction.direction

    # Try to pull a numeric line from the market_id (e.g. 9.5, 2.5, 0.5)
    m = re.search(r"(\d+\.?\d*)", market_id)
    line = m.group(1) if m else None

    # Period tag: 1H / 2H / FT
    mid_lower = market_id.lower()
    if "first_half" in mid_lower or "1h_" in mid_lower or "_1h" in mid_lower:
        period_tag = " 1H"
    elif "second_half" in mid_lower or "2h_" in mid_lower or "_2h" in mid_lower:
        period_tag = " 2H"
    else:
        period_tag = ""

    family_label = {
        "goals":          "GOALS",
        "corners":        "CORNERS",
        "cards":          "CARDS",
        "btts":           "BTTS",
        "next_goal":      "NEXT GOAL",
        "next_corner":    "NEXT CORNER",
        "props":          "PROPS",
        "result_1x2":     "1X2",
        "double_chance":  "DBLCHC",
        "asian_handicap": "AH",
        "race_to_x":      "RACE",
    }.get(family, family.upper())

    direction_lower = direction.lower()
    if family in ("next_goal", "next_corner", "result_1x2"):
        # Direction encodes the team / outcome, not a line.
        team_label = {
            "home": "Home",
            "away": "Away",
            "draw": "Draw",
            "no":   "No more",
        }.get(direction_lower, direction.capitalize())
        return f"{family_label}{period_tag} → {team_label}"

    direction_word = {
        "over":  "Over",
        "under": "Under",
        "yes":   "Yes",
        "no":    "No",
    }.get(direction_lower, direction.capitalize())

    if line is not None:
        return f"{family_label}{period_tag} {direction_word} {line}"
    return f"{family_label}{period_tag} {direction_word}"


def _team_label_for_pick(pick: ShadowPick, gsv: GameStateVector) -> str:
    """For next-goal / 1X2 picks, render which team the direction maps to.

    Returns an empty string for picks where the team is not relevant
    (over/under/yes/no). Used in the formatter as a trailing parenthetical
    so the operator never has to guess "home = which team".
    """
    direction = pick.full_thesis.prediction.direction.lower()
    family = pick.full_thesis.prediction.family.value
    if family not in ("next_goal", "next_corner", "result_1x2"):
        return ""
    if direction == "home":
        return f"team={gsv.home_team_id}"
    if direction == "away":
        return f"team={gsv.away_team_id}"
    return ""


# ──────────────────────────────────────────────────────────────────────
# Protocol — anything that records a pick callback satisfies this.
# ──────────────────────────────────────────────────────────────────────


class LivePickObserver(Protocol):
    """Side-effect-only callback for each v3 allowed pick.

    Implementations MUST NOT raise — the runtime calls them inside a
    best-effort try/except and a thrown observer would only spam logs.
    Implementations MUST NOT block — they run inside asyncio.to_thread,
    so I/O is fine, but anything synchronous + slow (e.g., a
    requests.post with no timeout) eats the v3 latency budget.
    """

    def on_pick(
        self,
        pick: ShadowPick,
        gsv: GameStateVector,
        window: WindowResult | None,
    ) -> None: ...


# ──────────────────────────────────────────────────────────────────────
# Pick formatter — THIS IS THE OPERATOR-OWNED UX SURFACE
# ──────────────────────────────────────────────────────────────────────


def format_v3_pick(
    pick: ShadowPick,
    gsv: GameStateVector,
    window: WindowResult | None,
) -> str:
    """Return ONE line describing the v3 pick, ready to print.

    THIS FUNCTION IS THE OPERATOR'S UX SURFACE. The default below is a
    starting point — customize it for what you want to see scrolling
    during a live jornada. Trade-offs to consider:

    - **Conciseness**: a busy jornada produces 5-20 picks/min. If the
      line is too long, you lose the ability to scan.
    - **Information density**: fixture_id alone is meaningless; you
      probably want minute + market direction + family at minimum.
    - **Visual cues**: terminal color codes (ANSI) help spot tier or
      window labels quickly. Run ``echo -e "\\e[31mRED\\e[0m"`` to
      test if your terminal supports them.
    - **Audit value**: include the archetype + layer so you can spot
      "rule" vs "pattern" theses without grep'ing the parquet later.

    Available fields:

    - ``pick.fixture_id`` (int)
    - ``pick.timestamp_utc`` (datetime)
    - ``pick.full_thesis.archetype`` (ThesisArchetype enum)
    - ``pick.full_thesis.source.layer`` (str: "rule" or "pattern")
    - ``pick.full_thesis.prediction.family`` (MarketFamily enum)
    - ``pick.full_thesis.prediction.direction`` (str: over/under/yes/no/...)
    - ``pick.candidate.market_id`` (str)
    - ``pick.candidate.fair_prob`` (float)
    - ``pick.candidate.mes.score`` (float, the MES total)
    - ``pick.candidate.mes.base_edge`` (float)
    - ``gsv.time.minute`` (int)
    - ``gsv.score.home_goals`` / ``gsv.score.away_goals`` (int)
    - ``gsv.tactical.game_phase`` (str: "cagey_closed"/"open_attacking"/etc)
    - ``window.label`` (str: "HOT"/"OPTIMAL"/"WARM"/"COLD"/"INDEFINITE")
    """
    # Operator-customized aesthetic colored output. ANSI 16-color palette
    # (works on any modern terminal). To disable colors set NO_COLOR=1 in
    # environment — _ansi() returns empty strings under that convention.
    layer = pick.full_thesis.source.layer
    layer_tag = (
        f"{_ansi(_BOLD, _CYAN)}[R]{_ansi(_RESET)}"
        if layer == "rule"
        else f"{_ansi(_BOLD, _MAGENTA)}[P]{_ansi(_RESET)}"
    )
    minute = gsv.time.minute
    score = f"{gsv.score.home_goals}-{gsv.score.away_goals}"
    archetype = pick.full_thesis.archetype.value
    direction = pick.full_thesis.prediction.direction
    edge_pct = pick.candidate.mes.base_edge * 100
    mes = pick.candidate.mes.score

    dir_color = _direction_color(direction)
    edge_color = _edge_color(edge_pct)
    mes_color = _mes_color(mes)
    win_str = _format_window_label(window)
    bet_desc = _humanize_bet(pick, gsv)
    team_tag = _team_label_for_pick(pick, gsv)
    team_str = (
        f"  {_ansi(_DIM)}{team_tag}{_ansi(_RESET)}" if team_tag else ""
    )

    return (
        f"{layer_tag} "
        f"{_ansi(_DIM)}fid={pick.fixture_id}{_ansi(_RESET)} "
        f"{_ansi(_BOLD, _YELLOW)}{minute:>3d}'{_ansi(_RESET)} "
        f"{_ansi(_BOLD)}{score}{_ansi(_RESET)}  "
        f"{_ansi(_WHITE)}{archetype:<28}{_ansi(_RESET)}"
        f"{_ansi(_DIM)} → {_ansi(_RESET)}"
        f"{dir_color}{bet_desc}{_ansi(_RESET)}  "
        f"edge {edge_color}{edge_pct:+6.2f}%{_ansi(_RESET)}  "
        f"MES {mes_color}{mes:.2f}{_ansi(_RESET)}  "
        f"{win_str}"
        f"{team_str}"
    )


# ──────────────────────────────────────────────────────────────────────
# Built-in observers
# ──────────────────────────────────────────────────────────────────────


class NullObserver:
    """No-op observer. Default when V3_LIVE_OBSERVE is unset."""

    def on_pick(
        self,
        pick: ShadowPick,
        gsv: GameStateVector,
        window: WindowResult | None,
    ) -> None:
        return None


class StdoutObserver:
    """Writes each pick to a file-like stream (stdout by default).

    Pass ``stream`` to redirect to a file or alternate sink. The
    observer flushes after each line so picks appear without
    waiting for buffer fill.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        prefix: str = "v3:",
        formatter: Any = format_v3_pick,
    ) -> None:
        self._stream = stream or sys.stdout
        self._prefix = prefix
        self._format = formatter

    def on_pick(
        self,
        pick: ShadowPick,
        gsv: GameStateVector,
        window: WindowResult | None,
    ) -> None:
        try:
            line = self._format(pick, gsv, window)
            print(f"{self._prefix} {line}", file=self._stream, flush=True)
        except Exception:  # noqa: BLE001
            # Observers MUST NEVER throw into the runtime. Silently
            # swallow a formatter bug rather than risk the v3 dual-write
            # loop logging spam every iteration.
            return None


class FileObserver(StdoutObserver):
    """Same as StdoutObserver but writes to a file with line-buffered IO.

    The file is opened in append mode and stays open for the process
    lifetime. Mid-jornada rotations are not supported (kill the watch
    process to rotate).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        prefix: str = "v3:",
        formatter: Any = format_v3_pick,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append mode — picks appear in tail -f in real time.
        stream = path.open("a", encoding="utf-8", buffering=1)
        super().__init__(stream=stream, prefix=prefix, formatter=formatter)
        self._path = path

    @property
    def path(self) -> Path:
        return self._path


# ──────────────────────────────────────────────────────────────────────
# Env-driven factory
# ──────────────────────────────────────────────────────────────────────


_TRUE_VALUES = {"1", "true", "yes", "on"}


def build_observer_from_env() -> LivePickObserver:
    """Construct the observer dictated by environment variables.

    - ``V3_LIVE_OBSERVE`` unset / falsey → ``NullObserver``
    - ``V3_LIVE_OBSERVE=true`` and ``V3_LIVE_OBSERVE_LOG`` unset →
      ``StdoutObserver`` (prints to the watch.py terminal)
    - ``V3_LIVE_OBSERVE=true`` and ``V3_LIVE_OBSERVE_LOG=/path/to/v3.log``
      → ``FileObserver`` writing to that path; ``tail -f`` it from a
      second terminal.
    """
    enabled = os.getenv("V3_LIVE_OBSERVE", "").strip().lower() in _TRUE_VALUES
    if not enabled:
        return NullObserver()
    log_path = os.getenv("V3_LIVE_OBSERVE_LOG", "").strip()
    if log_path:
        return FileObserver(log_path)
    return StdoutObserver()


__all__ = [
    "FileObserver",
    "LivePickObserver",
    "NullObserver",
    "StdoutObserver",
    "build_observer_from_env",
    "format_v3_pick",
]
