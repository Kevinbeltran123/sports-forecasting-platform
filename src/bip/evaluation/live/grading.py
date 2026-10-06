"""Pick grading — single source of truth for win/loss determination.

Both ``analyze_jornada`` (post-jornada graded picks DB) and ``backtest``
(snapshot replay) need to determine whether a given pick won or lost
against a final match outcome. Before this module they had separate
implementations:

  - ``analyze_jornada._grade_pick``: full coverage of all markets
  - ``backtest._outcome_for_pick``: missing draw_no_bet, btts_second_half,
    cards_*, team_to_score_first, home/away_clean_sheet — silently
    returned 0/0 for those markets and inflated the "graded" count
    while reporting 0% win rate

This module unifies them. ``grade_pick(market, selection, outcome)``
returns ``True`` (won), ``False`` (lost), or ``None`` (ungradable —
e.g., draw on draw-no-bet, market we don't know how to grade).

The ``FinalOutcome`` dataclass carries every signal needed to grade
any market we emit, including ``total_cards`` and ``total_corners``
which come from match events (not score state).

This module has NO live-data dependencies — it operates on a
``FinalOutcome`` snapshot that the caller derived from a finished
fixture. That keeps the grader pure + trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.live.match_state import LiveMatchState
from bip.sports.football.sportmonks.schemas import Fixture


@dataclass(frozen=True)
class FinalOutcome:
    """Final match state used to grade every emitted pick.

    Constructed once per fixture from the FT-state snapshot. The
    ``goal_events`` timeline is preserved so we can answer
    "team_to_score_first" exactly (rather than the best-effort
    inference both prior implementations used).
    """

    fixture_id: int
    league_id: int | None
    home_goals: int
    away_goals: int
    home_goals_first_half: int
    away_goals_first_half: int
    total_cards: int
    total_corners: int
    is_finished: bool
    # (minute, scoring_team_id) — sorted ascending. Empty when no goals.
    goal_events: tuple[tuple[int, int], ...] = ()
    home_team_id: int = 0
    away_team_id: int = 0

    @property
    def fulltime_result(self) -> str:
        if self.home_goals > self.away_goals:
            return "home"
        if self.home_goals == self.away_goals:
            return "draw"
        return "away"

    @property
    def first_half_result(self) -> str:
        if self.home_goals_first_half > self.away_goals_first_half:
            return "home"
        if self.home_goals_first_half == self.away_goals_first_half:
            return "draw"
        return "away"

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def total_goals_first_half(self) -> int:
        return self.home_goals_first_half + self.away_goals_first_half

    @property
    def total_goals_second_half(self) -> int:
        return self.total_goals - self.total_goals_first_half

    @property
    def home_goals_second_half(self) -> int:
        return self.home_goals - self.home_goals_first_half

    @property
    def away_goals_second_half(self) -> int:
        return self.away_goals - self.away_goals_first_half

    @property
    def btts(self) -> bool:
        return self.home_goals > 0 and self.away_goals > 0

    @property
    def btts_first_half(self) -> bool:
        return (
            self.home_goals_first_half > 0
            and self.away_goals_first_half > 0
        )

    @property
    def btts_second_half(self) -> bool:
        return (
            self.home_goals_second_half > 0
            and self.away_goals_second_half > 0
        )

    @property
    def first_scorer(self) -> str:
        """Team that scored first, or "none" if 0-0.

        Uses the ``goal_events`` timeline when available — the previous
        analyze_jornada implementation could only infer this from final
        score (and got "none" wrong when both teams scored).
        """
        if not self.goal_events:
            return "none"
        first_minute, first_team = self.goal_events[0]
        if first_team == self.home_team_id:
            return "home"
        if first_team == self.away_team_id:
            return "away"
        return "none"


def derive_final_outcome_from_state(
    state: LiveMatchState, fixture: Fixture | None = None,
) -> FinalOutcome | None:
    """Build a FinalOutcome from a finished LiveMatchState.

    Returns None when the state isn't yet final. ``fixture`` (optional)
    is used for league_id when available — falls back to ``state.league_id``.
    """
    if not state.is_finished:
        return None
    home_1h = sum(
        1 for minute, team in state.goal_events
        if minute <= 45 and team == state.home_team_id
    )
    away_1h = sum(
        1 for minute, team in state.goal_events
        if minute <= 45 and team == state.away_team_id
    )
    total_cards = (
        len(state.yellow_card_events) + len(state.red_card_events)
    )
    total_corners = state.home_corners + state.away_corners
    league_id = state.league_id
    if league_id is None and fixture is not None:
        league_id = fixture.league_id
    return FinalOutcome(
        fixture_id=state.fixture_id,
        league_id=league_id,
        home_goals=state.home_goals,
        away_goals=state.away_goals,
        home_goals_first_half=home_1h,
        away_goals_first_half=away_1h,
        total_cards=total_cards,
        total_corners=total_corners,
        is_finished=True,
        goal_events=tuple(state.goal_events),
        home_team_id=state.home_team_id,
        away_team_id=state.away_team_id,
    )


def _parse_total_line(market: str, prefix: str) -> float | None:
    """Extract numeric line from market keys like 'ou_2_5' → 2.5."""
    if not market.startswith(prefix):
        return None
    rest = market[len(prefix):]
    parts = rest.split("_")
    if len(parts) < 2:
        return None
    try:
        return float(f"{parts[0]}.{parts[1]}")
    except ValueError:
        return None


def grade_pick(
    market: str, selection: str, outcome: FinalOutcome,
) -> bool | None:
    """Return True (won), False (lost), or None (ungradable / void).

    Covers every market the predictor emits. Add a new branch here
    when a new market is added — the cascade-driven design ensures
    callers (analyze_jornada, backtest) stay in sync.
    """
    # ── 1X2-family ─────────────────────────────────────────────────────
    if market == "fulltime_result":
        return selection == outcome.fulltime_result
    if market == "first_half_result":
        return selection == outcome.first_half_result
    if market == "double_chance":
        ft = outcome.fulltime_result
        return (
            (selection == "1x" and ft in ("home", "draw"))
            or (selection == "x2" and ft in ("draw", "away"))
            or (selection == "12" and ft in ("home", "away"))
        )
    if market == "draw_no_bet":
        ft = outcome.fulltime_result
        if ft == "draw":
            return None  # void (stake refunded)
        return selection == ft
    if market == "htft":
        # selection format e.g. "home_home", "draw_away"
        ht = outcome.first_half_result
        ft = outcome.fulltime_result
        return selection == f"{ht}_{ft}"

    # ── BTTS family ────────────────────────────────────────────────────
    if market == "btts":
        return (selection == "yes") == outcome.btts
    if market == "btts_first_half":
        return (selection == "yes") == outcome.btts_first_half
    if market == "btts_second_half":
        return (selection == "yes") == outcome.btts_second_half

    # ── Clean sheets ───────────────────────────────────────────────────
    if market in ("home_clean_sheet", "away_clean_sheet"):
        opp_goals = (
            outcome.away_goals if market == "home_clean_sheet"
            else outcome.home_goals
        )
        return (selection == "yes") == (opp_goals == 0)

    # ── First scorer (3-way: home / away / none) ───────────────────────
    if market == "team_to_score_first":
        return selection == outcome.first_scorer

    # ── Match goals OU ─────────────────────────────────────────────────
    # Bug discovered 2026-05-10 audit: previous code used
    # ``total < line + 1`` for the under arm, which double-marked picks
    # where total = ceil(line) as both over AND under winners. For a
    # 2.5 line with 3 goals: over wins (3 > 2.5) AND under wins
    # (3 < 3.5). Real-world: under should LOSE with 3 goals on 2.5 line.
    # Fix: ``total < line`` is the correct half-line under semantics.
    line = _parse_total_line(market, "ou_")
    if line is not None:
        if selection == "over":
            return outcome.total_goals > line
        if selection == "under":
            return outcome.total_goals < line
        return None

    line = _parse_total_line(market, "first_half_ou_")
    if line is not None:
        if selection == "over":
            return outcome.total_goals_first_half > line
        if selection == "under":
            return outcome.total_goals_first_half < line
        return None

    # ── Per-team goals OU 1.5 ──────────────────────────────────────────
    if market in ("home_ou_1_5", "away_ou_1_5"):
        team_goals = (
            outcome.home_goals if market == "home_ou_1_5"
            else outcome.away_goals
        )
        if selection == "over":
            return team_goals > 1.5
        if selection == "under":
            return team_goals < 1.5
        return None

    line = _parse_total_line(market, "corners_total_")
    if line is not None:
        if selection == "over":
            return outcome.total_corners > line
        if selection == "under":
            return outcome.total_corners < line
        return None

    line = _parse_total_line(market, "cards_total_")
    if line is not None:
        if selection == "over":
            return outcome.total_cards > line
        if selection == "under":
            return outcome.total_cards < line
        return None

    # Unknown market — caller should not have emitted it
    return None
