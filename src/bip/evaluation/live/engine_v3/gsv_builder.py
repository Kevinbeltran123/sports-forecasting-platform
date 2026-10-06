"""GSV builder — projects ``LiveMatchState`` + priors + market snapshot
into a typed ``GameStateVector``.

Architectural role: the builder is the **single point** where loose
Sportmonks data crystallizes into structured causal signals. Any
downstream consumer (hypothesis generator, market selector, predictor)
reads ONLY the GSV — never raw Sportmonks types. This is what makes the
v3 pipeline auditable and testable independently of the API surface.

Uses the **advanced stats** that Sportmonks Pro exposes:
- ``BIG_CHANCES_CREATED`` / ``BIG_CHANCES_MISSED`` — primary xG proxy
- ``SHOTS_INSIDEBOX`` — territorial domination signal
- ``DANGEROUS_ATTACKS`` — pressure proxy in absence of xG feed
- ``KEY_PASSES`` — chance creation, lags shot-rate by 30-60s
- ``TOTAL_CROSSES`` / ``ACCURATE_CROSSES`` — set-piece intensity
- ``FOULS`` + ``YELLOW_CARDS`` + ``ref_card_rate_prior`` — cards regime

The tactical phase enums (``parking_bus``/``controlling``/``pressing``/
``chasing``/``collapsing``) are inferred from a small decision-tree
over these signals — not learned. They are deliberately **lossy** per
sec 4 of the design doc: enums beat dense vectors for auditability,
which is the whole point of v3.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

from bip.evaluation.live.engine_v3.gsv import (
    CardsState,
    CornerState,
    CriticalEvent,
    FlowState,
    GamePhase,
    GameStateVector,
    MarketSnapshot,
    NumericalState,
    PreMatchPriors,
    PressingIntensity,
    RosterState,
    ScoreState,
    SubstitutionEvent,
    TacticalState,
    TeamPhase,
    Tempo,
    TimeState,
    XGState,
)
from bip.evaluation.live.match_state import LiveMatchState
from bip.sports.football.sportmonks.types import StatType

# Minimum implied-probability gap between home and away (from team-named
# bookmaker markets) above which we trust the market favourite. A gap
# below this means the market sees a coin-flip; we then fall back to
# the prior-λ signal. Day-4 (2026-05-13, n=31) empirical anchor: the
# narrowest non-coin-flip gap observed was ~6 pp (Espanyol vs Athletic
# Club: 39.3% home vs 39.3% away — exactly even, so falls back).
_MARKET_DOM_MIN_PROB_GAP = 0.04

# Shadow gap for dominant-team identification (2026-05-14 forensic, task 5a).
# Computed alongside the enforced 0.04 gap to collect data on whether
# a narrower threshold (0.025) would resolve more near-coin-flip markets.
# The enforced dominant_team_id uses _MARKET_DOM_MIN_PROB_GAP (0.04).
# The shadow value uses this lower gap and is stored in gsv.shadow_dominant_team_id.
# One week of shadow data will determine if 0.025 is a sound policy.
_MARKET_DOM_MIN_PROB_GAP_SHADOW = 0.025

# elo_diff threshold below which we treat the elo signal as "absent".
# Production observed value across 31 Day-4 fixtures: 0.0 (never
# populated). Kept as a forward-compatibility hook for when the feed
# is wired up — see internal research notes
_ELO_DIFF_USABLE_MIN = 25.0

# ML-lambda gap threshold: when |lambda_home - lambda_away| < this,
# the ML λ signal is considered a coin-flip and we defer to the
# bookmaker market signal (tier 2). This guards against near-equal λ
# values producing an arbitrary dominant team assignment.
# Default 0.15 goals gap (~10% relative for a 1.5λ game) — configurable
# via the ``ml_lambda_min_gap`` parameter of ``_choose_dominant_team_id``.
_ML_LAMBDA_MIN_GAP: float = 0.15

# Minimum token length to count as a match between a market-id team
# fragment and the GSV team name. 3 catches the smallest meaningful
# tokens (e.g., "ofi", "psg") while filtering filler like "fc", "de".
_NAME_TOKEN_MIN_LEN = 3

# Expected xG-diff conditional on goal-diff: empirical anchor used to
# derive ``xg.xg_vs_score_divergence``. Pulled from a top-5-league
# rolling sample (2024-2025) — see internal research notes §3.2. The
# magnitude of divergence above this baseline is what archetype #5
# (regression to xG) fires on.
_EXPECTED_XG_DIFF_FOR_SCORE: dict[int, float] = {
    -3: -2.4,
    -2: -1.6,
    -1: -0.8,
    0: 0.0,
    1: 0.8,
    2: 1.6,
    3: 2.4,
}


def _expected_xg_diff(goal_diff: int) -> float:
    """Bounded linear ≈ 0.8 × goal_diff, clipped at ±3."""
    if goal_diff >= 3:
        return _EXPECTED_XG_DIFF_FOR_SCORE[3]
    if goal_diff <= -3:
        return _EXPECTED_XG_DIFF_FOR_SCORE[-3]
    return _EXPECTED_XG_DIFF_FOR_SCORE.get(goal_diff, 0.8 * goal_diff)


def _live_xg_advanced(stats: dict[int, float]) -> float:
    """Compute live xG using Sportmonks advanced stats.

    The hierarchy (best signal first):
    1. ``BIG_CHANCES_CREATED`` × 0.28 (each big chance ≈ a tap-in xG)
    2. ``SHOTS_INSIDEBOX`` × 0.10
    3. ``SHOTS_ON_TARGET`` outside box ≈ 0.05 (= total OT − inside box ot)
    4. ``DANGEROUS_ATTACKS`` × 0.003 as a tail signal

    The weights are conservative anchors — the real calibration happens
    in the conditional predictor downstream, where these proxies are
    blended with Sportmonks' own ``xg`` prediction when emitted.
    """
    bc = stats.get(StatType.BIG_CHANCES_CREATED, 0.0)
    sib = stats.get(StatType.SHOTS_INSIDEBOX, 0.0)
    sot = stats.get(StatType.SHOTS_ON_TARGET, 0.0)
    sout = max(0.0, sot - sib * 0.6)  # rough fraction of OT that came from outside box
    da = stats.get(StatType.DANGEROUS_ATTACKS, 0.0)
    return 0.28 * bc + 0.10 * sib + 0.05 * sout + 0.003 * da


def _pressing_intensity(state: LiveMatchState, side: str) -> PressingIntensity:
    """Derive a 3-level pressing-intensity enum from advanced stats.

    Two signal paths, blended:

    1. Rolling-window (preferred when trends are available):
       - ``DANGEROUS_ATTACKS`` rate in the last 10 minutes
       - ``KEY_PASSES`` rate in the same window
       - average pressure (when emitted)
    2. Match-level fallback (when no trends emitted by the league):
       - Per-minute aggregate rates of the same stats, normalised so a
         well-pressing team registers as ``mid``/``high`` regardless of
         trend availability.

    The fallback prevents the predictor from collapsing the entire match
    to ``low``-pressure ``controlling`` phase when a league happens not
    to emit trend records — a real gap encountered in lower-tier
    competitions covered by Sportmonks.
    """
    da10 = state.dangerous_attacks_in_last_window(side, window=10)
    kp10 = state.key_passes_in_last_window(side, window=10)
    avg_pressure = (
        state.home_pressure_avg if side == "home" else state.away_pressure_avg
    )
    window_signal = da10 + 2.0 * kp10 + 0.3 * avg_pressure

    # Fallback: per-minute rate × 10 (so 10 minutes of typical play)
    if state.minute > 0:
        stats = state.home_stats if side == "home" else state.away_stats
        da_total = stats.get(StatType.DANGEROUS_ATTACKS, 0.0)
        kp_total = stats.get(StatType.KEY_PASSES, 0.0)
        rate_signal = (da_total + 2.0 * kp_total) * 10.0 / max(1, state.minute)
    else:
        rate_signal = 0.0

    # Take the stronger signal — we want the conservative case where
    # both signals agree on "low" to actually land at "low".
    signal = max(window_signal, rate_signal)
    if signal < 8:
        return "low"
    if signal < 20:
        return "mid"
    return "high"


def _team_phase(
    state: LiveMatchState,
    side: str,
    *,
    own_goals: int,
    opp_goals: int,
    is_dominant: bool,
) -> TeamPhase:
    """Map a side's situational signals to a 5-level enum.

    The decision tree, in priority order:
    - score-state already collapsing (down 2+ with <15m) → ``collapsing``
    - trailing AND high pressing → ``chasing``
    - leading AND killing-clock detector AND late game → ``parking_bus``
    - leading or controlling with high possession → ``controlling``
    - high pressing intensity → ``pressing``
    - default → ``controlling``
    """
    minute = state.minute
    pi = _pressing_intensity(state, side)
    trailing = own_goals < opp_goals
    diff = own_goals - opp_goals

    if diff <= -2 and minute >= 75:
        return "collapsing"
    if trailing and pi == "high":
        return "chasing"
    if (diff >= 1 and state.is_killing_clock(side)) or (
        diff >= 1 and minute >= 75 and pi == "low" and is_dominant
    ):
        return "parking_bus"
    if pi == "high":
        return "pressing"
    return "controlling"


def _game_phase(home: TeamPhase, away: TeamPhase, minute: int) -> GamePhase:
    """Compose a global game-phase enum from each side's phase.

    Definitions:
    - ``desperate``: a side is collapsing OR chasing while we're past 80'
    - ``cruise``: leader is parking_bus AND past 75'
    - ``cagey_closed``: both controlling AND past 70' (low-tempo late stalemate)
    - ``cagey_open``: both controlling AND before minute 30 (slow start)
    - ``open_attacking``: anything else
    """
    if "collapsing" in (home, away):
        return "desperate"
    if (home == "chasing" or away == "chasing") and minute >= 80:
        return "desperate"
    if "parking_bus" in (home, away) and minute >= 75:
        return "cruise"
    if home == "controlling" and away == "controlling":
        if minute >= 70:
            return "cagey_closed"
        if minute < 30:
            return "cagey_open"
    return "open_attacking"


def _tempo(state: LiveMatchState) -> Tempo:
    """Tempo = total shots rate. Anchored to ~26 shots/match = 0.29/min."""
    total = state.home_shots_total + state.away_shots_total
    if state.minute <= 0:
        return "medium"
    rate = total / state.minute
    if rate < 0.20:
        return "low"
    if rate < 0.40:
        return "medium"
    return "high"


def _xg_per_min_last_15(state: LiveMatchState, side: str) -> float:
    """Approximate live xG rate over the last 15 minutes using
    big_chances + inside-box shots from the trends stream."""
    bc15 = state.stat_in_last_window(side, StatType.BIG_CHANCES_CREATED, window=15)
    sib15 = state.stat_in_last_window(side, StatType.SHOTS_INSIDEBOX, window=15)
    window = min(15, max(1, state.minute))
    return (0.28 * bc15 + 0.10 * sib15) / window


def _role_signal_for_sub(state: LiveMatchState, sub_minute: int, team_id: int) -> str:
    """Best-effort defensive-vs-offensive sub inference.

    Without per-player position data, we approximate: a sub in the last
    20 minutes by the leading team when goal_diff>=1 is more likely
    defensive (Archetype #4). A sub by the trailing team is more likely
    offensive. This is a placeholder until we wire player-position data.
    """
    is_home = team_id == state.home_team_id
    own = state.home_goals if is_home else state.away_goals
    opp = state.away_goals if is_home else state.home_goals
    if own > opp and sub_minute >= 60:
        return "defensive"
    if own < opp and sub_minute >= 60:
        return "offensive"
    return "unknown"


def _normalize_team_name(name: str) -> set[str]:
    """Strip accents, lower-case, split on non-alnum, drop short fillers.

    Returns the set of meaningful tokens. Used to fuzzy-match a team
    name from the GSV (e.g., ``"Manchester City"``) against a market-
    id fragment (e.g., ``"man_city"``) since Sportmonks uses different
    abbreviations across endpoints.
    """
    if not name:
        return set()
    no_accents = unicodedata.normalize("NFKD", name).encode(
        "ascii", "ignore"
    ).decode("ascii")
    tokens = re.split(r"[^a-z0-9]+", no_accents.lower())
    return {t for t in tokens if len(t) >= _NAME_TOKEN_MIN_LEN}


def _name_matches(market_fragment: str, team_name: str) -> bool:
    """True iff the market-id fragment and team name share at least one
    meaningful (≥3 char) token. Common cases:

    - ``"man_city"`` vs ``"Manchester City"`` → {"city"} overlap ✓
    - ``"crystal_palace"`` vs ``"Crystal Palace"`` → exact set match ✓
    - ``"man_city"`` vs ``"Crystal Palace"`` → ∅ ✗
    - ``"barcelona"`` vs ``"FC Barcelona"`` → {"barcelona"} ✓
    """
    return bool(
        _normalize_team_name(market_fragment) & _normalize_team_name(team_name)
    )


def _market_implied_win_probs(
    state: LiveMatchState, markets: MarketSnapshot,
) -> tuple[float | None, float | None]:
    """Sum implied P(team wins) from team-named BTTS-x-result markets.

    The Sportmonks "result + BTTS" market exposes 6 unambiguously
    team-named outcomes per fixture:
        result___both_teams_to_score_<home>_/_yes
        result___both_teams_to_score_<home>_/_no
        result___both_teams_to_score_<away>_/_yes
        result___both_teams_to_score_<away>_/_no
        result___both_teams_to_score_draw_/_yes
        result___both_teams_to_score_draw_/_no

    P(team wins) = 1/yes_odds + 1/no_odds (raw, with overround — relative
    ordering survives un-vigging). These are the **load-bearing**
    bookmaker signal: market-ids embed team names, so there's no
    home/away convention ambiguity.

    We deliberately ignore the numeric ``fulltime_result_1/_2`` lines:
    Day-4 (2026-05-13, n=31) observed 8 fixtures where those lines
    disagreed with the team-named markets (most egregiously Palace-
    City: ``fulltime_result_1=1.30`` implied Palace ~77% to win, while
    team-named markets put City at ~80% — a clean inversion). The
    numeric labels do NOT reliably map to home/away in the Sportmonks
    odds feed.
    """
    home_p: float | None = None
    away_p: float | None = None
    for mid, line in markets.lines.items():
        if not mid.startswith("result___both_teams_to_score_"):
            continue
        # Strip the prefix and the "_/_yes" / "_/_no" suffix.
        try:
            tail = mid.split("result___both_teams_to_score_", 1)[1]
            team_part = tail.split("_/_", 1)[0]
        except IndexError:
            continue
        if "draw" in team_part:
            continue
        dec = line.side_a_decimal
        if not dec or dec <= 1.0:
            continue
        implied = 1.0 / dec
        if _name_matches(team_part, state.home_team_name or ""):
            home_p = (home_p or 0.0) + implied
        elif _name_matches(team_part, state.away_team_name or ""):
            away_p = (away_p or 0.0) + implied
    return home_p, away_p


def _market_dominant_team_id(
    state: LiveMatchState, markets: MarketSnapshot,
    *,
    min_prob_gap: float = _MARKET_DOM_MIN_PROB_GAP,
) -> int | None:
    """Return the bookmaker favourite's team_id from team-named markets.

    Returns ``None`` when:
    - No team-named markets present (only numeric lines available).
    - Only one side has team-named markets (incomplete data).
    - The gap in implied P(win) is below ``min_prob_gap``
      (market sees a coin-flip). Defaults to the enforced 0.04 gap.

    Pass ``min_prob_gap=_MARKET_DOM_MIN_PROB_GAP_SHADOW`` to compute the
    shadow dominant team id at the 0.025 threshold.
    """
    home_p, away_p = _market_implied_win_probs(state, markets)
    if home_p is None or away_p is None:
        return None
    if abs(home_p - away_p) < min_prob_gap:
        return None
    return state.home_team_id if home_p > away_p else state.away_team_id


def _choose_dominant_team_id(
    state: LiveMatchState,
    priors: PreMatchPriors,
    markets: MarketSnapshot,
    *,
    ml_lambda_min_gap: float = _ML_LAMBDA_MIN_GAP,
) -> int:
    """Decide the dominant team — ELO > ML-λ > bookmaker market > Sportmonks-λ.

    Precedence (top wins):

    1. Strong elo_diff (≥ ``_ELO_DIFF_USABLE_MIN``). Forward-compatible
       hook; production observed value is always 0.0 today.

    2. ML-lambda tier (NEW — Wave-3, Task 3.3): when
       ``PreMatchPriors.lambda_home_prematch / away`` were populated from
       the ML lambda store (penaltyblog Dixon-Coles fit) AND the gap
       |lh - la| >= ``ml_lambda_min_gap``, use the higher-λ team.
       When the gap is below the threshold (coin-flip λ), skip this tier
       and defer to the bookmaker market (tier 3).

       Rationale: ML-λ is positioned ABOVE the bookmaker market signal
       because (a) it is pre-match and unaffected by the live-odds
       noise observed on Day-4 (7 fixtures where market disagreed with
       the structural form signal), and (b) the Dixon-Coles fit uses the
       full season history, not just the current live snapshot. However
       it sits BELOW elo_diff so a strong elo advantage always wins.

       Detection: the lambda store populates the same PreMatchPriors
       fields as the Sportmonks type_id-240 derivation. We cannot
       distinguish them by field value. The ML-λ tier fires on ANY
       decisive λ gap (|lh - la| >= threshold) regardless of source —
       if the Sportmonks-derived λ disagrees with the market, that gap
       is still meaningful signal. If you need to suppress the ML tier,
       set ``ml_lambda_min_gap=float('inf')``.

    3. Team-named bookmaker market favourite (BTTS-x-result outcomes
       summed). The canonical live-market signal.

    4. Higher-λ from priors (Sportmonks type_id-240 / ML-λ coin-flip
       fallback). Used only when tiers 1-3 all produce None/coin-flip.

    Empirical Day-4 (2026-05-13, n=31) calibration:
    - 8 fixtures had numeric ``fulltime_result_1/2`` *disagreeing* with
      the team-named markets — that signal is structurally unreliable
      and is no longer consulted.
    - 7 fixtures had the priors λ *disagreeing* with the team-named
      markets (most prominently Palace vs Man City: λ said Palace was
      favourite, market clearly said City). The ML-λ tier with a decisive
      gap would have correctly overridden the coin-flip Sportmonks λ in
      those cases, but defers to market when λ is itself a coin-flip.
    """
    elo = priors.elo_diff
    if abs(elo) >= _ELO_DIFF_USABLE_MIN:
        return state.home_team_id if elo > 0 else state.away_team_id

    # Tier 2: ML-lambda gap (above market, below elo)
    lh, la = priors.lambda_home_prematch, priors.lambda_away_prematch
    if abs(lh - la) >= ml_lambda_min_gap:
        return state.home_team_id if lh > la else state.away_team_id

    # Tier 3: bookmaker market
    market_pick = _market_dominant_team_id(state, markets)
    if market_pick is not None:
        return market_pick

    # Tier 4: Sportmonks-λ coin-flip fallback (also covers ML-λ coin-flip)
    return state.home_team_id if lh >= la else state.away_team_id


def _ref_period(minute: int, is_finished: bool, is_half_time: bool) -> str:
    if is_finished:
        return "FT"
    if is_half_time:
        return "HT"
    if minute == 0:
        return "NS"
    if minute <= 45:
        return "1H"
    if minute <= 90:
        return "2H"
    if minute <= 105:
        return "ET1"
    if minute <= 120:
        return "ET2"
    return "PEN"


class GSVBuilder:
    """Stateful builder: tracks ``state_version`` per fixture so callers
    can detect frame transitions without re-keying timestamps."""

    def __init__(self) -> None:
        self._versions: dict[int, int] = {}

    def build(
        self,
        state: LiveMatchState,
        priors: PreMatchPriors,
        markets: MarketSnapshot,
        *,
        dominant_team_id: int | None = None,
        last_critical_event: CriticalEvent | None = None,
        last_critical_event_age_sec: float | None = None,
        ref_card_rate_prior: float = 0.0,
        now_utc: datetime | None = None,
    ) -> GameStateVector:
        """Construct a single GSV frame from current snapshots.

        Arguments:
            dominant_team_id: pre-match favorite ID. If ``None``, the
                builder picks the side with the higher prior ``λ`` from
                ``priors``.
            last_critical_event: most recent goal / red / key sub seen.
                Drives ``has_recent_critical_event`` (no-bet rule #3).
            ref_card_rate_prior: referee's career cards-per-game baseline.
                Drives archetype #6.
        """
        ts = now_utc or datetime.now(timezone.utc)

        # Pick the dominant team — explicit arg wins; else use the
        # priors/market/elo decision (see ``_choose_dominant_team_id``).
        if dominant_team_id is None:
            dominant_team_id = _choose_dominant_team_id(state, priors, markets)

        # Shadow dominant-team at 0.025 gap (task 5a). This is SEPARATE from the
        # enforced selection above and does NOT affect dominant_losing or any other
        # derived field. Stored in gsv.shadow_dominant_team_id for offline analysis.
        shadow_dominant_team_id = _market_dominant_team_id(
            state, markets, min_prob_gap=_MARKET_DOM_MIN_PROB_GAP_SHADOW,
        )

        # Score state — dominant_losing is the load-bearing predicate.
        leader_goals = state.home_goals if dominant_team_id == state.home_team_id else state.away_goals
        follower_goals = state.away_goals if dominant_team_id == state.home_team_id else state.home_goals
        dominant_losing = leader_goals < follower_goals
        last_goal_minute = state.goal_events[-1][0] if state.goal_events else None
        last_goal_team_id = state.goal_events[-1][1] if state.goal_events else None
        minutes_since_last_goal = (
            float(state.minute - last_goal_minute) if last_goal_minute is not None else float(state.minute)
        )

        score = ScoreState(
            home_goals=state.home_goals,
            away_goals=state.away_goals,
            goal_diff=state.home_goals - state.away_goals,
            dominant_team_id=dominant_team_id,
            dominant_losing=dominant_losing,
            last_goal_minute=last_goal_minute,
            last_goal_team_id=last_goal_team_id,
            last_goal_xg=None,
            minutes_since_last_goal=minutes_since_last_goal,
        )

        time = TimeState(
            minute=state.minute,
            period=_ref_period(state.minute, state.is_finished, state.is_half_time),
            time_remaining_half=max(0.0, (45 - state.minute) if state.minute < 45 else (90 - state.minute)),
            time_remaining_match=float(state.remaining_minutes),
        )

        numerical = NumericalState(
            home_players=max(0, 11 - sum(1 for _m, t in state.red_card_events if t == state.home_team_id)),
            away_players=max(0, 11 - sum(1 for _m, t in state.red_card_events if t == state.away_team_id)),
            numerical_advantage=(
                sum(1 for _m, t in state.red_card_events if t == state.away_team_id)
                - sum(1 for _m, t in state.red_card_events if t == state.home_team_id)
            ),
            red_cards_home=sum(1 for _m, t in state.red_card_events if t == state.home_team_id),
            red_cards_away=sum(1 for _m, t in state.red_card_events if t == state.away_team_id),
        )

        # xG — prefer advanced-stat computation; xg_vs_score_divergence is derived.
        home_xg = _live_xg_advanced(state.home_stats)
        away_xg = _live_xg_advanced(state.away_stats)
        xg_diff = home_xg - away_xg
        expected = _expected_xg_diff(score.goal_diff)
        xg = XGState(
            home_xg_total=home_xg,
            away_xg_total=away_xg,
            xg_diff=xg_diff,
            xg_per_min_home_last_15=_xg_per_min_last_15(state, "home"),
            xg_per_min_away_last_15=_xg_per_min_last_15(state, "away"),
            xg_vs_score_divergence=xg_diff - expected,
            shots_total=(state.home_shots_total, state.away_shots_total),
            shots_on_target=(
                int(state.home_stats.get(StatType.SHOTS_ON_TARGET, 0)),
                int(state.away_stats.get(StatType.SHOTS_ON_TARGET, 0)),
            ),
            shots_in_box=(
                int(state.home_stats.get(StatType.SHOTS_INSIDEBOX, 0)),
                int(state.away_stats.get(StatType.SHOTS_INSIDEBOX, 0)),
            ),
            big_chances=(
                int(state.home_stats.get(StatType.BIG_CHANCES_CREATED, 0)),
                int(state.away_stats.get(StatType.BIG_CHANCES_CREATED, 0)),
            ),
        )

        flow = FlowState(
            possession_home_5min=state.home_possession,
            possession_home_match=state.home_possession,
            attacks_last_10min=(
                state.stat_in_last_window("home", StatType.ATTACKS, window=10),
                state.stat_in_last_window("away", StatType.ATTACKS, window=10),
            ),
            dangerous_attacks_last_10min=(
                state.dangerous_attacks_in_last_window("home", window=10),
                state.dangerous_attacks_in_last_window("away", window=10),
            ),
            attack_zone_dominant=None,
            pressing_intensity=(
                _pressing_intensity(state, "home")
                if home_xg + away_xg < 0.01 or home_xg >= away_xg
                else _pressing_intensity(state, "away")
            ),
        )

        corners = CornerState(
            corners_home=state.home_corners,
            corners_away=state.away_corners,
            corner_rate_last_15min=float(
                state.corners_in_last_window("home", window=15)
                + state.corners_in_last_window("away", window=15)
            ) / 15.0,
        )

        cards = CardsState(
            yellows=(state.yellow_card_count_home, state.yellow_card_count_away),
            reds=(numerical.red_cards_home, numerical.red_cards_away),
            card_rate_last_15min=float(
                state.yellow_cards_in_last_window("home", window=15)
                + state.yellow_cards_in_last_window("away", window=15)
            ) / 15.0,
            ref_card_rate_prior=ref_card_rate_prior,
        )

        recent_subs = [
            SubstitutionEvent(
                minute=m,
                team_id=t,
                player_in_id=p or None,
                role_signal=_role_signal_for_sub(state, m, t),  # type: ignore[arg-type]
            )
            for m, t, p in state.substitution_events
            if state.minute - m <= 5
        ]
        roster = RosterState(
            subs_used=(state.substitutions_home, state.substitutions_away),
            subs_remaining=(max(0, 5 - state.substitutions_home), max(0, 5 - state.substitutions_away)),
            recent_subs_5min=recent_subs,
        )

        home_phase = _team_phase(
            state, "home",
            own_goals=state.home_goals, opp_goals=state.away_goals,
            is_dominant=dominant_team_id == state.home_team_id,
        )
        away_phase = _team_phase(
            state, "away",
            own_goals=state.away_goals, opp_goals=state.home_goals,
            is_dominant=dominant_team_id == state.away_team_id,
        )
        tactical = TacticalState(
            home_phase=home_phase,
            away_phase=away_phase,
            game_phase=_game_phase(home_phase, away_phase, state.minute),
            tempo=_tempo(state),
        )

        version = self._versions.get(state.fixture_id, 0) + 1
        self._versions[state.fixture_id] = version

        return GameStateVector(
            fixture_id=state.fixture_id,
            state_version=version,
            timestamp_utc=ts,
            home_team_id=state.home_team_id,
            away_team_id=state.away_team_id,
            home_team_name=state.home_team_name,
            away_team_name=state.away_team_name,
            score=score,
            time=time,
            numerical=numerical,
            xg=xg,
            flow=flow,
            corners=corners,
            cards=cards,
            roster=roster,
            tactical=tactical,
            priors=priors,
            markets=markets,
            last_critical_event=last_critical_event,
            last_critical_event_age_sec=last_critical_event_age_sec,
            shadow_dominant_team_id=shadow_dominant_team_id,
        )


__all__ = ["GSVBuilder"]
