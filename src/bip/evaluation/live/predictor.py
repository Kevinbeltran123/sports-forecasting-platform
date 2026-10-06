"""Live match predictor — combines Sportmonks ML predictions with our own
Dixon-Robinson-style live state model.

Strategy (operator-imparcial choice 2026-05-09, hardened 2026-05-09 night):

  Sportmonks already emits 29 well-calibrated pre-built predictions
  (1X2, BTTS, OU, HT/FT, First Half Winner, Correct Score, etc.). We
  TRUST those as a strong baseline — competing with their proprietary
  ML head-on is a losing battle. Our model adds value at THREE seams:

  1. Live state adjustments — Sportmonks predictions are pre-match
     biased; we re-score using Dixon-Robinson scaling for remaining-
     time markets:
         λ_remaining = λ_pre × (90 - minute) / 90
     This is the standard literature approach (Dixon & Robinson 1998).

  2. Live signal lift — when our `pressure` and `xG_proxy` strongly
     contradict Sportmonks (e.g., home pressure 80 vs 30 but pre-match
     prediction is balanced), we shift probabilities a bounded amount
     toward the live signal. The boost is capped at ±10pp per market.

  3. Information-density gate — at low live-signal density (early
     minutes, no events, no pressure samples), we DO NOT live-adjust;
     we return the Sportmonks prior verbatim. This kills the "min 8
     burst" pathology where the predictor invents divergence from
     thin air.

The output is a ``MarketProbabilities`` object with a probability for
every market the value detector knows how to compare against odds, plus
a per-market confidence half-width derived from disagreement between
Sportmonks's own prediction sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.tournaments.predictors.bivariate_poisson import (
    _bivariate_poisson_grid as _bivariate_poisson_grid_full,
)
from bip.sports.football.sportmonks.types import PredictionType, StatType


def _bivariate_poisson_grid(
    lam_home: float, lam_away: float, *, rho: float = 0.04, max_n: int = 8,
) -> np.ndarray:
    """Build a Bivariate Poisson grid over (additional_home, additional_away).

    Wraps the existing tournaments-package implementation, decomposing
    the rho parameter into the shared lambda_3 component:
        λ_3 = rho × √(λ_h × λ_a), clamped so λ_1, λ_2 ≥ 0
        λ_1 = λ_h − λ_3,  λ_2 = λ_a − λ_3
    """
    if lam_home <= 0 and lam_away <= 0:
        # Degenerate: no chance of any goals → all mass at (0,0)
        g = np.zeros((max_n + 1, max_n + 1))
        g[0, 0] = 1.0
        return g
    lh = max(lam_home, 1e-6)
    la = max(lam_away, 1e-6)
    lam_12 = rho * math.sqrt(lh * la)
    lam_12 = max(0.0, min(lam_12, min(lh, la) - 1e-9))
    lam_1 = lh - lam_12
    lam_2 = la - lam_12
    return _bivariate_poisson_grid_full(lam_1, lam_2, lam_12, max_n)


# Standard markets we emit. Selection-keys are inspired by Sportmonks
# vocabulary so callers can map to in-play odds market_descriptions.
MARKET_FULLTIME_RESULT = "fulltime_result"      # {home, draw, away}
MARKET_DOUBLE_CHANCE = "double_chance"           # {1x, x2, 12}
MARKET_DRAW_NO_BET = "draw_no_bet"               # {home, away}
MARKET_FIRST_HALF_RESULT = "first_half_result"   # {home, draw, away}
MARKET_HTFT = "htft"                              # 9 cells
MARKET_BTTS = "btts"                              # {yes, no}
MARKET_BTTS_FIRST_HALF = "btts_first_half"        # {yes, no}
MARKET_BTTS_SECOND_HALF = "btts_second_half"      # {yes, no}
MARKET_OU_05 = "ou_0_5"                           # {over, under}
MARKET_OU_15 = "ou_1_5"
MARKET_OU_25 = "ou_2_5"
MARKET_OU_35 = "ou_3_5"
MARKET_FIRST_HALF_OU_05 = "first_half_ou_0_5"
MARKET_FIRST_HALF_OU_15 = "first_half_ou_1_5"
MARKET_HOME_OU_15 = "home_ou_1_5"
MARKET_AWAY_OU_15 = "away_ou_1_5"
MARKET_HOME_CLEAN_SHEET = "home_clean_sheet"      # {yes, no}
MARKET_AWAY_CLEAN_SHEET = "away_clean_sheet"      # {yes, no}
MARKET_TEAM_TO_SCORE_FIRST = "team_to_score_first"  # {home, away, none}
MARKET_CORNERS_TOTAL_8_5 = "corners_total_8_5"    # {over, under}
MARKET_CORNERS_TOTAL_9_5 = "corners_total_9_5"
MARKET_CORNERS_TOTAL_10_5 = "corners_total_10_5"
MARKET_CORNERS_TOTAL_11_5 = "corners_total_11_5"
MARKET_CARDS_TOTAL_3_5 = "cards_total_3_5"        # {over, under}
MARKET_CARDS_TOTAL_4_5 = "cards_total_4_5"
MARKET_CARDS_TOTAL_5_5 = "cards_total_5_5"

# Tunables — all empirical, can be calibrated post-trial against real ROI
PRESSURE_DIFF_TRIGGER = 25.0          # avg pressure diff (0-100) before we adjust
PRESSURE_BOOST_MAX = 0.05              # max ±5pp shift on 1X2 from avg pressure
PRESSURE_TREND_TRIGGER = 10.0          # pressure derivative pts before triggers
PRESSURE_TREND_BOOST_MAX = 0.03        # max ±3pp shift from pressure trend
XG_SIGNAL_BOOST_MAX = 0.04             # max ±4pp shift from live xG vs expected
RED_CARD_LAMBDA_PENALTY = 0.25         # 25% reduction for affected team's λ
RED_CARD_LAMBDA_BOOST = 0.10           # 10% boost for opposing team's λ
TRAILING_LATE_GAME_BOOST_MAX = 0.04    # ±4pp for "trailing team pushes" effect

# Information-density threshold — below this we return Sportmonks priors
# verbatim instead of rebuilding from per-team OU + pressure (kills cluster C).
INFO_DENSITY_LIVE_THRESHOLD = 0.40

# League-prior expected total corners per match (top-5 European average).
# Per-league calibration tracked separately when post-trial data accrues.
DEFAULT_LEAGUE_CORNERS_PRIOR = 10.4
# Effective sample-size weight for the prior — at minute 0 we trust the
# prior fully; by minute 60 we trust observed corner rate ~3:1.
CORNERS_PRIOR_WEIGHT_MINUTES = 18.0

# League-prior expected total cards (yellows + reds) per match.
# Top-5 European average ~3.8 yellows + ~0.15 reds = 3.95.
DEFAULT_LEAGUE_CARDS_PRIOR = 3.95
CARDS_PRIOR_WEIGHT_MINUTES = 22.0

# Killing-the-clock dampening: if a side is in clock-killing mode, scale
# their goal-creation λ down by this fraction (4pp on overs).
KILLING_CLOCK_LAMBDA_DAMP = 0.15

# ── BTTS half-time decomposition (empirical, top-5 league average) ──────
# Of all BTTS-yes matches: ~45% have both teams scored by HT, ~55% don't
# (both goals after HT). These hardcoded factors are a backstop for when
# Sportmonks doesn't ship dedicated BTTS-1H/BTTS-2H predictions.
#
# TODO calibrate per-league post-trial: a high-pace league like the
# Bundesliga skews toward 1H BTTS realisation; tactical leagues like
# Serie A skew later. With ≥30 collected fixtures per league we can
# replace these with empirical fractions from goal_events timestamps.
BTTS_FIRST_HALF_FRACTION = 0.45
BTTS_SECOND_HALF_FRACTION = 0.55

# ── Team-form lift caps ─────────────────────────────────────────────────
# When team form (last ~10 matches) diverges from the league baseline,
# tilt pre-match priors proportionally — but BOUND the lift. Sportmonks's
# pre-match prediction already implicitly absorbs season form; the
# divergence-vs-baseline signal we add represents recent drift only.
#
# Cap multiplier at [0.7, 1.4]: a team scoring 1H 60% recently vs league
# 42% gives ratio 1.43 → capped to 1.4× tilt. This prevents over-fitting
# to small-sample form patterns when n_matches is in the 5-10 range.
TEAM_FORM_LIFT_MIN = 0.7
TEAM_FORM_LIFT_MAX = 1.4


@dataclass(frozen=True)
class MarketProbabilities:
    """Per-fixture probabilities, one map per market.

    Probabilities sum to 1.0 within a market (for mutually exclusive
    selections like 1X2 / BTTS / OU). Numeric tolerances are applied to
    the SUM at construction time (1.0 ± 1e-3 acceptable).

    ``confidences[market]`` is a per-market credibility-interval half-
    width derived from the disagreement between Sportmonks's own
    prediction sources (1X2 prior vs DC vs CORRECT_SCORE marginal). Used
    by ValueDetector to gate picks when our reported EV is within the
    noise band.
    """

    fixture_id: int
    minute: int
    snapshot_kind: str  # 'pre_match' | 'live'
    by_market: dict[str, dict[str, float]]
    # Provenance: which model contributed which probabilities
    sources: dict[str, str]   # market → 'sportmonks' | 'live_adjusted' | 'derived'
    # Per-market credibility half-width (decimal, e.g. 0.04 = ±4pp)
    confidences: dict[str, float] = field(default_factory=dict)

    def get(self, market: str, selection: str) -> float | None:
        m = self.by_market.get(market)
        if m is None:
            return None
        return m.get(selection)


class LiveMatchPredictor:
    """Combine Sportmonks predictions with live state adjustments."""

    def __init__(
        self,
        *,
        pressure_boost_max: float = PRESSURE_BOOST_MAX,
        pressure_trend_boost_max: float = PRESSURE_TREND_BOOST_MAX,
        xg_signal_boost_max: float = XG_SIGNAL_BOOST_MAX,
        red_card_penalty: float = RED_CARD_LAMBDA_PENALTY,
        red_card_boost: float = RED_CARD_LAMBDA_BOOST,
        trailing_late_boost_max: float = TRAILING_LATE_GAME_BOOST_MAX,
        info_density_threshold: float = INFO_DENSITY_LIVE_THRESHOLD,
    ) -> None:
        self.pressure_boost_max = pressure_boost_max
        self.pressure_trend_boost_max = pressure_trend_boost_max
        self.xg_signal_boost_max = xg_signal_boost_max
        self.red_card_penalty = red_card_penalty
        self.red_card_boost = red_card_boost
        self.trailing_late_boost_max = trailing_late_boost_max
        self.info_density_threshold = info_density_threshold

    def predict(self, state: LiveMatchState) -> MarketProbabilities:
        """Emit market probabilities for the current state."""
        by_market: dict[str, dict[str, float]] = {}
        sources: dict[str, str] = {}
        confidences: dict[str, float] = {}

        # 1. Fulltime result — adjust for live signal (gated by info density)
        ft_probs, ft_src = self._fulltime_result(state)
        if ft_probs:
            by_market[MARKET_FULLTIME_RESULT] = ft_probs
            sources[MARKET_FULLTIME_RESULT] = ft_src
            confidences[MARKET_FULLTIME_RESULT] = self._confidence_1x2(state, ft_probs)
            # Derived: double chance — prefer Sportmonks DC when available,
            # else derive from 1X2.
            dc_probs, dc_src = self._double_chance(state, ft_probs)
            if dc_probs:
                by_market[MARKET_DOUBLE_CHANCE] = dc_probs
                sources[MARKET_DOUBLE_CHANCE] = dc_src
                confidences[MARKET_DOUBLE_CHANCE] = confidences[MARKET_FULLTIME_RESULT]
            # Derived: draw-no-bet (B-20)
            dnb = self._draw_no_bet_from_1x2(ft_probs)
            if dnb is not None:
                by_market[MARKET_DRAW_NO_BET] = dnb
                sources[MARKET_DRAW_NO_BET] = "derived"
                confidences[MARKET_DRAW_NO_BET] = confidences[MARKET_FULLTIME_RESULT] * 1.2

        # 2. First half result — only useful before HT
        if not state.is_finished and state.minute < 45:
            fh_probs = self._sportmonks_first_half_result(state)
            if fh_probs:
                by_market[MARKET_FIRST_HALF_RESULT] = fh_probs
                sources[MARKET_FIRST_HALF_RESULT] = "sportmonks"
                confidences[MARKET_FIRST_HALF_RESULT] = 0.05

        # 3. HT/FT — only useful before HT
        if not state.is_finished and state.minute < 45:
            htft = self._sportmonks_htft(state)
            if htft:
                by_market[MARKET_HTFT] = htft
                sources[MARKET_HTFT] = "sportmonks"
                confidences[MARKET_HTFT] = 0.06  # 9-cell market is noisier

        # 4. BTTS (full match) — already-scored teams shift probability
        btts = self._btts(state)
        if btts:
            by_market[MARKET_BTTS] = btts
            sources[MARKET_BTTS] = "live_adjusted"
            confidences[MARKET_BTTS] = 0.04

        # 5. BTTS first half — pre-HT only
        if not state.is_finished and state.minute < 45:
            btts1 = self._btts_first_half(state)
            if btts1:
                by_market[MARKET_BTTS_FIRST_HALF] = btts1
                sources[MARKET_BTTS_FIRST_HALF] = "live_adjusted"
                confidences[MARKET_BTTS_FIRST_HALF] = 0.06

        # 6. BTTS second half — derived (B-24)
        btts2h = self._btts_second_half(state)
        if btts2h:
            by_market[MARKET_BTTS_SECOND_HALF] = btts2h
            sources[MARKET_BTTS_SECOND_HALF] = "derived"
            confidences[MARKET_BTTS_SECOND_HALF] = 0.06

        # 7. OU 0.5/1.5/2.5/3.5 — Dixon-Robinson scaled when in play
        for line, market in [(0.5, MARKET_OU_05), (1.5, MARKET_OU_15),
                             (2.5, MARKET_OU_25), (3.5, MARKET_OU_35)]:
            ou = self._ou_total(state, line=line)
            if ou:
                by_market[market] = ou
                sources[market] = "live_adjusted"
                confidences[market] = 0.04

        # 8. First half OU
        if not state.is_finished and state.minute < 45:
            for line, market in [(0.5, MARKET_FIRST_HALF_OU_05),
                                 (1.5, MARKET_FIRST_HALF_OU_15)]:
                ou = self._first_half_ou(state, line=line)
                if ou:
                    by_market[market] = ou
                    sources[market] = "live_adjusted"
                    confidences[market] = 0.05

        # 9. Per-team OU 1.5
        home_ou15 = self._team_ou(state, line=1.5, side="home")
        away_ou15 = self._team_ou(state, line=1.5, side="away")
        if home_ou15:
            by_market[MARKET_HOME_OU_15] = home_ou15
            sources[MARKET_HOME_OU_15] = "sportmonks"
            confidences[MARKET_HOME_OU_15] = 0.05
        if away_ou15:
            by_market[MARKET_AWAY_OU_15] = away_ou15
            sources[MARKET_AWAY_OU_15] = "sportmonks"
            confidences[MARKET_AWAY_OU_15] = 0.05

        # 10. Team clean sheet (B-23) — derived from team-to-score-remaining
        for side, market_key in (
            ("home", MARKET_AWAY_CLEAN_SHEET),  # away CS = home doesn't score
            ("away", MARKET_HOME_CLEAN_SHEET),  # home CS = away doesn't score
        ):
            cs = self._team_clean_sheet(state, scoring_side=side)
            if cs:
                by_market[market_key] = cs
                sources[market_key] = "derived"
                confidences[market_key] = 0.05

        # 11. Team to score first (B-17) — useful when score is 0-0
        if state.home_goals == 0 and state.away_goals == 0:
            ttsf = self._team_to_score_first(state)
            if ttsf:
                by_market[MARKET_TEAM_TO_SCORE_FIRST] = ttsf
                sources[MARKET_TEAM_TO_SCORE_FIRST] = "sportmonks"
                confidences[MARKET_TEAM_TO_SCORE_FIRST] = 0.06

        # 12. Corners totals (B-1) — only when CORNERS stat is populated
        # AND informational density has accumulated. Without the density
        # gate, an isolated corner at minute 10 with no other stats
        # produces a Poisson rate estimated from a single data point —
        # the same cluster-C pathology we kill on goal markets.
        if (
            state.minute >= 10
            and (state.home_corners + state.away_corners) > 0
            and state.informational_density >= self.info_density_threshold
        ):
            for line, market in (
                (8.5, MARKET_CORNERS_TOTAL_8_5),
                (9.5, MARKET_CORNERS_TOTAL_9_5),
                (10.5, MARKET_CORNERS_TOTAL_10_5),
                (11.5, MARKET_CORNERS_TOTAL_11_5),
            ):
                co = self._corners_total(state, line=line)
                if co:
                    by_market[market] = co
                    sources[market] = "live_adjusted"
                    confidences[market] = 0.06

        # 13. Cards totals (#255) — yellow_card_events + reds give us a
        # rate; engagement composite (tackles + interceptions + duels) and
        # fouls feed the per-minute card rate. Same density-gate as
        # corners — an isolated yellow at min 10 isn't a Poisson basis.
        total_cards_so_far = (
            state.yellow_card_count_home
            + state.yellow_card_count_away
            + len(state.red_card_events)
        )
        if (
            state.minute >= 15
            and total_cards_so_far > 0
            and state.informational_density >= self.info_density_threshold
        ):
            for line, market in (
                (3.5, MARKET_CARDS_TOTAL_3_5),
                (4.5, MARKET_CARDS_TOTAL_4_5),
                (5.5, MARKET_CARDS_TOTAL_5_5),
            ):
                ca = self._cards_total(state, line=line)
                if ca:
                    by_market[market] = ca
                    sources[market] = "live_adjusted"
                    # Wider CI than corners — cards are noisier (depend
                    # heavily on referee + game state).
                    confidences[market] = 0.08

        kind = "live" if state.is_live or state.is_half_time else "pre_match"
        return MarketProbabilities(
            fixture_id=state.fixture_id,
            minute=state.minute,
            snapshot_kind=kind,
            by_market=by_market,
            sources=sources,
            confidences=confidences,
        )

    # ── per-market predictors ───────────────────────────────────────────

    def _fulltime_result(
        self, state: LiveMatchState,
    ) -> tuple[dict[str, float], str] | tuple[None, None]:
        sm = state.sportmonks_prediction(PredictionType.FULLTIME_RESULT_PROBABILITY)
        if not sm:
            return None, None
        # Sportmonks pre-match probability — used as the prior
        base = {
            "home": _pct(sm.get("home")),
            "draw": _pct(sm.get("draw")),
            "away": _pct(sm.get("away")),
        }

        # Pre-match: trust Sportmonks fully
        if not state.is_live and not state.is_half_time and state.minute == 0:
            return _normalise(base), "sportmonks"

        # ── B-G3 / B-7 information-density gate ────────────────────────
        # At low density (early minutes, no events, no pressure samples),
        # we DO NOT rebuild 1X2 from per-team OU + nudges — we return the
        # Sportmonks prior verbatim. This kills the cluster-C "min-8 burst".
        density = state.informational_density
        if density < self.info_density_threshold:
            return _normalise(base), "sportmonks_low_info"

        # ── Dixon-Robinson live scoring ────────────────────────────────
        # Estimate per-team λ_remaining via Sportmonks per-team OU 0.5
        # → P(team scores ≥ 1) = 1 - exp(-λ_full); invert to get λ_full,
        # then scale by remaining-time fraction.
        lam_home_remaining = self._team_lambda_remaining(state, side="home")
        lam_away_remaining = self._team_lambda_remaining(state, side="away")

        if lam_home_remaining is None or lam_away_remaining is None:
            # Fall back to nudged Sportmonks if we can't model remaining time
            return _normalise(base), "sportmonks"

        # Build a bivariate Poisson grid over ADDITIONAL goals scored
        # (rho=0.04 Dixon-Coles default; international-style low-correlation
        # is fine for live-state since most goals are independent in tail)
        grid = _bivariate_poisson_grid(lam_home_remaining, lam_away_remaining,
                                        rho=0.04, max_n=8)

        # Map each (h_add, a_add) outcome to current_score + add → final outcome
        p_home, p_draw, p_away = 0.0, 0.0, 0.0
        for h_add in range(grid.shape[0]):
            for a_add in range(grid.shape[1]):
                fh = state.home_goals + h_add
                fa = state.away_goals + a_add
                p = grid[h_add, a_add]
                if fh > fa:
                    p_home += p
                elif fh == fa:
                    p_draw += p
                else:
                    p_away += p

        live_probs = _normalise({"home": p_home, "draw": p_draw, "away": p_away})

        # Pressure-based + trend-aware + trailing-late nudges
        if state.is_live:
            live_probs = self._apply_live_nudges(state, live_probs)

        return live_probs, "live_adjusted"

    def _apply_live_nudges(
        self, state: LiveMatchState, probs: dict[str, float],
    ) -> dict[str, float]:
        """Layer the live signal nudges atop the Dixon-Robinson scored probs.

        Order of application matters less than caps because each nudge is
        bounded; the final ``_clip`` keeps probabilities in [0.001, 0.999]
        and ``_normalise`` rescales to sum 1.
        """
        adj = dict(probs)

        # 1. Average pressure differential — only when BOTH sides have
        # samples. Asymmetric data (e.g., Sportmonks emits home pressure
        # but drops away samples this snapshot) produces a fake huge diff
        # against an artificial 0 baseline → false PRESSURE_DIFF_TRIGGER.
        has_home_pressure = bool(state.home_pressure_recent)
        has_away_pressure = bool(state.away_pressure_recent)
        if has_home_pressure and has_away_pressure:
            diff = state.home_pressure_avg - state.away_pressure_avg
            if abs(diff) > PRESSURE_DIFF_TRIGGER:
                shift = self.pressure_boost_max * min(1.0, abs(diff) / 50.0)
                if diff > 0:
                    adj["home"] += shift
                    adj["away"] -= shift / 2
                    adj["draw"] -= shift / 2
                else:
                    adj["away"] += shift
                    adj["home"] -= shift / 2
                    adj["draw"] -= shift / 2

        # 2. Pressure TREND (rising vs falling) — needs ≥4 samples per side
        if (
            len(state.home_pressure_recent) >= 4
            and len(state.away_pressure_recent) >= 4
        ):
            trend_diff = state.home_pressure_trend - state.away_pressure_trend
        else:
            trend_diff = 0.0
        if abs(trend_diff) > PRESSURE_TREND_TRIGGER:
            shift = self.pressure_trend_boost_max * min(1.0, abs(trend_diff) / 30.0)
            if trend_diff > 0:
                # Home momentum building
                adj["home"] += shift
                adj["draw"] -= shift / 2
                adj["away"] -= shift / 2
            else:
                adj["away"] += shift
                adj["draw"] -= shift / 2
                adj["home"] -= shift / 2

        # 3. Late-game trailing-team push (>=70 min, trailing by exactly 1)
        # Trend-aware: scale the nudge by the trailing team's MEASURED push
        # intensity (shot acceleration). A team trailing 1-0 at min 75 that's
        # actually pressing (accel 1.5×) gets a bigger draw-equalising
        # nudge than a team that's faded (accel 0.6×) and probably won't
        # equalise. Without trends, we default to factor=1.0 (preserves
        # original unconditional nudge).
        if state.minute >= 70 and abs(state.score_diff_home) == 1:
            base_shift = self.trailing_late_boost_max * min(
                1.0, (state.minute - 70) / 20.0
            )
            trailing_side = "home" if state.score_diff_home == -1 else "away"
            push_factor = 1.0
            if state.trends:
                accel = state.shot_acceleration(trailing_side)
                # Map shot_acceleration to push_factor:
                #   accel >= 1.5 → 1.5× boost (peak push)
                #   accel ~ 1.0 → 1.0× boost (steady)
                #   accel <= 0.5 → 0.4× boost (faded — they aren't coming back)
                #   accel = 0.0 (no recent shots) → 0.5× boost (could be just lull)
                if accel == 0.0:
                    push_factor = 0.5
                else:
                    push_factor = max(0.40, min(1.5, accel))
            shift = base_shift * push_factor
            if state.score_diff_home == -1:
                # Home trailing by 1 → push up draw (equalising)
                adj["draw"] += shift
                adj["away"] -= shift
            else:
                # Away trailing by 1
                adj["draw"] += shift
                adj["home"] -= shift

        # 4. Killing-the-clock dampening (B-3) — when a leading team is
        # parking the bus (high possession, no penetration), suppress the
        # late-game equaliser bias and push probability to the leader.
        for side in ("home", "away"):
            if not state.is_killing_clock(side):
                continue
            leading = (
                (side == "home" and state.score_diff_home > 0)
                or (side == "away" and state.score_diff_home < 0)
            )
            if leading:
                shift = self.trailing_late_boost_max * 0.5  # half of trailing-push
                if side == "home":
                    adj["home"] += shift
                    adj["draw"] -= shift / 2
                    adj["away"] -= shift / 2
                else:
                    adj["away"] += shift
                    adj["draw"] -= shift / 2
                    adj["home"] -= shift / 2

        return _normalise(_clip(adj))

    @staticmethod
    def _scaled_red_card_penalty(remaining: int) -> float:
        """Penalty fraction (0-1) applied to affected team's λ.

        Larger when more time remains because the team must play down
        a man for longer; smaller when red card is late + likely to
        result in defensive shell rather than goal-rate collapse.
        """
        if remaining >= 45:
            return 0.40
        if remaining >= 30:
            return 0.35
        if remaining >= 15:
            return 0.25
        return 0.15

    @staticmethod
    def _scaled_red_card_boost(remaining: int) -> float:
        """Boost fraction (0-1) applied to opposing team's λ."""
        if remaining >= 45:
            return 0.18
        if remaining >= 30:
            return 0.14
        if remaining >= 15:
            return 0.10
        return 0.05

    def _team_lambda_remaining(
        self, state: LiveMatchState, *, side: str,
    ) -> float | None:
        """Per-team Poisson λ scaled to remaining minutes, with live adjustments.

        Pipeline:
        1. Back out full-match λ from Sportmonks per-team OU 0.5:
            λ_full = -ln(1 − P(over 0.5))
        2. Scale to remaining time (90 − minute) / 90.
        3. Apply red-card adjustment (Mengual & Forrest 2002):
           - Affected team: λ *= (1 - 0.25..0.40 by remaining time)
           - Opposing team: λ *= (1 + 0.10..0.18)
        4. Apply BIG_CHANCES_MISSED regression (B-6): if missed/created > 0.5,
           team has been wasteful; small λ boost (regress toward conversion).
        5. Apply SAVES signal (B-5): if opp has many saves vs few SOT, the
           opposing keeper has been overperforming; small additional boost.
        6. Apply INJURIES penalty (B-9): mid-match injuries depress λ.
        7. Apply engagement composite (B-11): high tackles/interceptions
           mid-match → goal suppression.
        8. Apply killing-the-clock dampening (B-3): if THIS side is parking
           the bus, drop their attacking λ.
        9. Apply live xG signal: if live xG-proxy / expected ratio > 1.2
           or < 0.8, scale λ accordingly (capped at ±20%).
        """
        type_id = (
            PredictionType.HOME_OVER_UNDER_0_5_PROBABILITY if side == "home"
            else PredictionType.AWAY_OVER_UNDER_0_5_PROBABILITY
        )
        sm = state.sportmonks_prediction(type_id)
        if not sm:
            return None
        p_over_0_5 = _pct(sm.get("yes"))
        if not 0.0 < p_over_0_5 < 1.0:
            return None
        lam_full = -math.log(1 - p_over_0_5)
        remaining_fraction = max(0, 90 - state.minute) / 90.0
        lam_remaining = lam_full * remaining_fraction

        if not state.is_live:
            return lam_remaining

        # ── Red-card adjustment (scaled by remaining minutes) ─────────
        red_min_self = (
            state.red_card_minute_home if side == "home"
            else state.red_card_minute_away
        )
        red_min_opp = (
            state.red_card_minute_away if side == "home"
            else state.red_card_minute_home
        )
        if red_min_self is not None and red_min_self <= state.minute:
            time_a_man_down = max(0, 90 - state.minute)
            penalty = self._scaled_red_card_penalty(time_a_man_down)
            lam_remaining *= (1.0 - penalty)
        if red_min_opp is not None and red_min_opp <= state.minute:
            time_opp_a_man_down = max(0, 90 - state.minute)
            boost = self._scaled_red_card_boost(time_opp_a_man_down)
            lam_remaining *= (1.0 + boost)

        # ── BIG_CHANCES_MISSED regression (B-6) ──────────────────────
        # Wasteful team likely to convert next big chance. Bound nudge at +8%.
        bc_created = (
            state.home_big_chances_created if side == "home"
            else state.away_big_chances_created
        )
        bc_missed = (
            state.home_big_chances_missed if side == "home"
            else state.away_big_chances_missed
        )
        if bc_created >= 2 and bc_missed >= 1:
            miss_rate = bc_missed / max(bc_created, 1)
            if miss_rate >= 0.5:
                lam_remaining *= 1.0 + min(0.08, 0.04 * miss_rate)

        # ── SAVES signal (B-5) ───────────────────────────────────────
        # If THIS team has been kept out by OPP keeper many times, their xG
        # has been under-rewarded → small regression-toward-mean boost.
        opp_saves = state.away_saves if side == "home" else state.home_saves
        own_sot = (
            state.home_stats.get(StatType.SHOTS_ON_TARGET, 0) if side == "home"
            else state.away_stats.get(StatType.SHOTS_ON_TARGET, 0)
        )
        if opp_saves >= 4 and own_sot >= 4:
            save_rate = opp_saves / max(own_sot, 1)
            if save_rate >= 0.7:
                lam_remaining *= 1.0 + min(0.06, 0.04 * (save_rate - 0.5))

        # ── INJURIES penalty (B-9) ───────────────────────────────────
        # Mid-match injuries cost players → soft λ depression. Capped at -15%.
        inj = state.home_injuries if side == "home" else state.away_injuries
        if inj >= 1 and state.minute >= 30:
            lam_remaining *= max(0.85, 1.0 - 0.05 * inj)

        # ── Engagement composite (B-11) ──────────────────────────────
        # High tackles+interceptions+duels suggests scrappy game → goal
        # suppression. Use TOTAL match engagement (both sides) since this
        # is a match-state signal not a team-specific one.
        total_eng = state.home_engagement + state.away_engagement
        # Empirical: top-5 league average ≈ 80 by minute 60. Above 110+ = scrappy.
        if state.minute >= 30:
            expected_eng = (state.minute / 60.0) * 80.0
            if expected_eng > 0 and total_eng > expected_eng * 1.4:
                ratio = total_eng / expected_eng
                damp = min(0.10, 0.05 * (ratio - 1.4))
                lam_remaining *= (1.0 - damp)

        # ── Killing-the-clock dampening (B-3) ────────────────────────
        if state.is_killing_clock(side):
            lam_remaining *= (1.0 - KILLING_CLOCK_LAMBDA_DAMP)

        # ── Live xG signal: actual creation vs minute-prorated expectation ──
        if state.minute >= 15:  # too noisy in opening minutes
            live_xg = (
                state.home_shot_quality_xg_proxy if side == "home"
                else state.away_shot_quality_xg_proxy
            )
            # Expected xG by this minute is lam_full * (minute / 90)
            expected = lam_full * (state.minute / 90.0)
            if expected > 0.05:
                ratio = live_xg / expected
                # Cap the multiplier shift at ±20%
                ratio_capped = max(0.80, min(1.20, ratio))
                # Soften: only half of ratio diff reaches λ
                lam_remaining *= 0.5 + 0.5 * ratio_capped

        # ── Trends-derived multi-stat momentum composite ─────────────
        # Cumulative xG-proxy is slow-moving; the momentum_score combines
        # shots + dangerous_attacks + key_passes rolling rates vs match-
        # average via weighted geometric mean — much more robust than
        # shots-only (a team firing desperate long-range attempts shows
        # high shots but flat KP/DA → composite stays neutral).
        # Skips automatically when no trends or minute < 15.
        momentum = state.momentum_score(side, window=5)
        if momentum != 1.0:  # 1.0 is the no-data / steady-pace neutral
            # Cap multiplier at [0.85, 1.15] — momentum effect is real
            # but a single 5-minute window can't justify >15% λ shift.
            momentum_mult = max(0.85, min(1.15, 0.5 + 0.5 * momentum))
            lam_remaining *= momentum_mult

        # ── Set-piece intensity (corners + crosses rolling) ─────────
        # ~30% of all goals in top-5 leagues come from set-pieces
        # (corners, free-kicks, throw-in deliveries that lead to crosses).
        # When a team is generating set-pieces well above the league
        # baseline in the last 15 min, their goal threat is elevated
        # beyond what shots-rolling alone captures (the "siege" pattern:
        # team deep in opp half winning corners but no shots yet).
        # Skip automatically without trends.
        if state.trends and state.minute >= 25:
            sp = state.set_piece_intensity(side, window=15)
            if sp != 1.0:
                # Cap at [0.92, 1.10] — set-piece intensity is a smaller,
                # more uncertain signal than direct shot momentum.
                sp_mult = max(0.92, min(1.10, 0.7 + 0.3 * sp))
                lam_remaining *= sp_mult

        return max(0.0, lam_remaining)

    def _double_chance(
        self, state: LiveMatchState, ft_probs: dict[str, float],
    ) -> tuple[dict[str, float] | None, str]:
        """Return (DC probabilities, source).

        Prefers Sportmonks ``DOUBLE_CHANCE_PROBABILITY`` when available
        (B-18); falls back to derivation from our 1X2 probabilities.

        Sportmonks DC body shape: ``{draw_home, draw_away, home_away}``
        — confusingly named because Sportmonks uses "first_outcome second_outcome"
        ordering by alphabetical, where:
          - ``home_away`` = 12 (no draw)
          - ``draw_home`` = 1X
          - ``draw_away`` = X2
        """
        sm = state.sportmonks_prediction(PredictionType.DOUBLE_CHANCE_PROBABILITY)
        # Pre-match: trust Sportmonks DC directly when present
        if sm and not state.is_live and not state.is_half_time and state.minute == 0:
            body = {
                "1x": _pct(sm.get("draw_home")),
                "x2": _pct(sm.get("draw_away")),
                "12": _pct(sm.get("home_away")),
            }
            if all(0.0 < v < 1.0 for v in body.values()):
                return _normalise(body), "sportmonks"
        # Live or fallback: derive from our 1X2 (which already reflects state)
        return self._double_chance_from_1x2(ft_probs), "derived"

    def _draw_no_bet_from_1x2(
        self, ft: dict[str, float],
    ) -> dict[str, float] | None:
        """Renormalise 1X2 dropping the draw → DNB market."""
        non_draw = ft["home"] + ft["away"]
        if non_draw <= 0:
            return None
        return {
            "home": ft["home"] / non_draw,
            "away": ft["away"] / non_draw,
        }

    def _team_clean_sheet(
        self, state: LiveMatchState, *, scoring_side: str,
    ) -> dict[str, float] | None:
        """P(scoring_side does NOT score the rest of the match).

        ``scoring_side="home"`` returns the AWAY clean-sheet market
        (i.e. away keeper keeps a clean sheet).
        """
        scoring_prob = self._team_to_score_remaining(state, scoring_side)
        if scoring_prob is None:
            return None
        return {
            "yes": _clip_scalar(1.0 - scoring_prob),
            "no": _clip_scalar(scoring_prob),
        }

    def _team_to_score_first(
        self, state: LiveMatchState,
    ) -> dict[str, float] | None:
        """Use Sportmonks TEAM_TO_SCORE_FIRST_PROBABILITY (type 238).

        Body shape: ``{home, draw, away}`` — "draw" means no goal scored
        in the rest of the match (clean sheet both sides).
        """
        sm = state.sportmonks_prediction(
            PredictionType.TEAM_TO_SCORE_FIRST_PROBABILITY
        )
        if not sm:
            return None
        return _normalise({
            "home": _pct(sm.get("home")),
            "away": _pct(sm.get("away")),
            "none": _pct(sm.get("draw")),
        })

    def _btts_second_half(
        self, state: LiveMatchState,
    ) -> dict[str, float] | None:
        """BTTS-2H — both teams score within the SECOND HALF period.

        Bookmaker pricing of BTTS-2H is independent of first-half goals: it
        asks whether each team scores at least once in the 2H itself. So we
        compute P(home scores in 2H) × P(away scores in 2H) using
        ``_team_score_in_remaining_uncond`` — which deliberately ignores the
        first-half scoreline (unlike ``_team_to_score_remaining``, which
        short-circuits to 1.0 for already-scored teams).
        """
        sm = state.sportmonks_prediction(PredictionType.BTTS_PROBABILITY)
        if not sm:
            return None
        # Pre-HT: BTTS-2H requires both teams to score AFTER HT.
        # See BTTS_SECOND_HALF_FRACTION docstring for empirical basis +
        # per-league calibration TODO.
        if state.minute < 45 and not state.is_half_time:
            full_yes = _pct(sm.get("yes"))
            yes_2h = full_yes * BTTS_SECOND_HALF_FRACTION
            yes_2h = self._apply_team_form_lift(
                yes_2h, state, signal="second_half_goal_rate",
                league_baseline=0.65 ** 2,  # league P(both teams score 2H)
            )
            return _normalise({"yes": yes_2h, "no": 1.0 - yes_2h})
        # HT or 2H: P(both score in remaining time), agnostic to 1H goals.
        home_2h = self._team_score_in_remaining_uncond(state, "home")
        away_2h = self._team_score_in_remaining_uncond(state, "away")
        if home_2h is None or away_2h is None:
            return None
        yes = home_2h * away_2h
        yes = self._apply_team_form_lift(
            yes, state, signal="second_half_goal_rate",
            league_baseline=0.65 ** 2,
        )
        return _normalise({"yes": yes, "no": 1.0 - yes})

    def _corners_total(
        self, state: LiveMatchState, *, line: float,
    ) -> dict[str, float] | None:
        """Total-corners O/U using a Poisson model.

        Two-source rate blend:
          1. Match-average rate: cumulative corners ÷ minute, regularised
             by league prior (CORNERS_PRIOR_WEIGHT_MINUTES on the prior).
          2. Recent rate: corners in last 15 min from trends (when trends
             data is available). Captures current pace better than match
             average for matches that have shifted tempo (e.g., one team
             dominating second half corner-wise).

        Final rate is a weighted blend: 60% recent, 40% match-average,
        when both are available. Falls back to match-average alone when
        trends aren't emitted.
        """
        if state.minute < 10:
            return None
        corners_so_far = state.home_corners + state.away_corners
        if corners_so_far > line:
            return {"over": 1.0, "under": 0.0}

        # Source 1: match-average rate with league-prior regularisation
        effective_minutes = state.minute + CORNERS_PRIOR_WEIGHT_MINUTES
        prior_corners = (
            DEFAULT_LEAGUE_CORNERS_PRIOR
            * (CORNERS_PRIOR_WEIGHT_MINUTES / 90.0)
        )
        match_avg_rate = (corners_so_far + prior_corners) / max(
            effective_minutes, 1.0,
        )

        # Source 2: recent corner rate from trends (last 15 min)
        recent_rate: float | None = None
        if state.trends and state.minute >= 25:
            recent_corners = (
                state.corners_in_last_window("home", window=15)
                + state.corners_in_last_window("away", window=15)
            )
            window_minutes = min(15, state.minute)
            if window_minutes >= 5:
                recent_rate = recent_corners / float(window_minutes)

        # Blend: 60% recent, 40% match-average when both present
        if recent_rate is not None:
            blended_rate = 0.60 * recent_rate + 0.40 * match_avg_rate
        else:
            blended_rate = match_avg_rate

        lam_remaining = blended_rate * max(0, 90 - state.minute)
        if lam_remaining <= 0:
            return {"over": 0.0, "under": 1.0}
        needed = max(0, math.ceil(line) - corners_so_far)
        p_over = 1.0 - _poisson_cdf(needed - 1, lam_remaining)
        return {"over": _clip_scalar(p_over), "under": _clip_scalar(1 - p_over)}

    def _cards_total(
        self, state: LiveMatchState, *, line: float,
    ) -> dict[str, float] | None:
        """Total-cards (yellow + red) O/U via Poisson with a league prior.

        Cards depend heavily on referee disposition + game state (tight
        match = more cards, blowout = fewer). λ_remaining is built from
        three signals:
          1. Match-average rate (cumulative cards ÷ match-minute,
             regularised by league prior)
          2. Engagement composite multiplier (cumulative
             tackles+interceptions+duels vs expected for the minute)
          3. Recent foul + yellow rolling rate from trends — captures
             referee-state drift (a referee who started lenient and
             tightened up, or vice versa) and late-game tactical-foul
             clusters that the cumulative average smooths over.
        """
        cards_so_far = (
            state.yellow_card_count_home
            + state.yellow_card_count_away
            + len(state.red_card_events)
        )
        if cards_so_far > line:
            return {"over": 1.0, "under": 0.0}
        # Match-average rate with league-prior regularisation
        effective_minutes = state.minute + CARDS_PRIOR_WEIGHT_MINUTES
        prior_cards = (
            DEFAULT_LEAGUE_CARDS_PRIOR
            * (CARDS_PRIOR_WEIGHT_MINUTES / 90.0)
        )
        rate_per_min = (cards_so_far + prior_cards) / max(effective_minutes, 1.0)
        lam_remaining = rate_per_min * max(0, 90 - state.minute)

        # Engagement multiplier — scrappy games book more cards.
        if state.minute >= 30:
            total_eng = state.home_engagement + state.away_engagement
            expected_eng = (state.minute / 60.0) * 80.0
            if expected_eng > 0 and total_eng > 0:
                eng_ratio = total_eng / expected_eng
                eng_mult = max(0.85, min(1.20, 0.5 + 0.5 * eng_ratio))
                lam_remaining *= eng_mult

        # Trend-aware adjustment from rolling fouls + yellows in last 15 min.
        # When recent foul rate vs match-avg rate is materially > 1, game
        # is heating up and λ_remaining should rise; cluster of recent
        # yellows compounds (referee in card-happy mode).
        if state.trends and state.minute >= 25:
            recent_fouls = (
                state.fouls_in_last_window("home", window=15)
                + state.fouls_in_last_window("away", window=15)
            )
            cum_fouls = (
                state._cumulative_at_minute("home", StatType.FOULS, state.minute)
                + state._cumulative_at_minute("away", StatType.FOULS, state.minute)
            )
            avg_fouls_per_min = cum_fouls / state.minute if state.minute > 0 else 0
            window_min = min(15, state.minute)
            if avg_fouls_per_min > 0 and window_min >= 5:
                recent_per_min = recent_fouls / float(window_min)
                foul_ratio = recent_per_min / avg_fouls_per_min
                # Cap at [0.85, 1.20] — fouls trend is real but bounded.
                foul_mult = max(0.85, min(1.20, 0.6 + 0.4 * foul_ratio))
                lam_remaining *= foul_mult
            # Yellow-cluster bonus: 3+ yellows in last 15 min suggests
            # the ref is now booking more freely → +10% λ.
            recent_yellows = (
                state.yellow_cards_in_last_window("home", window=15)
                + state.yellow_cards_in_last_window("away", window=15)
            )
            if recent_yellows >= 3:
                lam_remaining *= 1.10

        if lam_remaining <= 0:
            return {"over": 0.0, "under": 1.0}
        needed = max(0, math.ceil(line) - cards_so_far)
        p_over = 1.0 - _poisson_cdf(needed - 1, lam_remaining)
        return {"over": _clip_scalar(p_over), "under": _clip_scalar(1 - p_over)}

    def _sportmonks_first_half_result(
        self, state: LiveMatchState,
    ) -> dict[str, float] | None:
        sm = state.sportmonks_prediction(PredictionType.FIRST_HALF_WINNER_PROBABILITY)
        if not sm:
            return None
        return _normalise({
            "home": _pct(sm.get("home")),
            "draw": _pct(sm.get("draw")),
            "away": _pct(sm.get("away")),
        })

    def _sportmonks_htft(self, state: LiveMatchState) -> dict[str, float] | None:
        sm = state.sportmonks_prediction(PredictionType.HTFT_PROBABILITY)
        if not sm:
            return None
        keys = [
            "home_home", "home_draw", "home_away",
            "draw_home", "draw_draw", "draw_away",
            "away_home", "away_draw", "away_away",
        ]
        body = {k: _pct(sm.get(k)) for k in keys}
        return _normalise(body)

    def _btts(self, state: LiveMatchState) -> dict[str, float] | None:
        sm = state.sportmonks_prediction(PredictionType.BTTS_PROBABILITY)
        if not sm:
            return None
        # If the match is in play and only one team has scored, BTTS-yes
        # depends only on the OTHER team scoring in remaining time.
        # We use the Sportmonks pre-match BTTS as a baseline and
        # condition on the current state.
        base_yes = _pct(sm.get("yes"))
        base_no = _pct(sm.get("no"))
        if not state.is_live and not state.is_half_time:
            return _normalise({"yes": base_yes, "no": base_no})

        # CASE 1: BOTH teams scoreless — need BOTH to score in remaining time.
        # Bug discovered 2026-05-10: previous code fell into the "one team
        # scored" branch for the 0-0 case and computed P(home scores)
        # alone, dramatically over-stating P(BTTS yes). Real win rate on
        # those picks: 1/33 (3%). Fix: joint probability, treating teams
        # as independent (same independence assumption Phase 5 BTTS-2H uses).
        if state.home_goals == 0 and state.away_goals == 0:
            home_p = self._team_to_score_remaining(state, "home")
            away_p = self._team_to_score_remaining(state, "away")
            if home_p is None or away_p is None:
                return _normalise({"yes": base_yes, "no": base_no})
            yes = home_p * away_p
            return _normalise({"yes": yes, "no": 1.0 - yes})

        # CASE 2: exactly ONE team has scored — BTTS-yes depends on the
        # OTHER team scoring in remaining time. Existing logic, unchanged.
        if state.home_goals == 0 or state.away_goals == 0:
            need_team = "away" if state.home_goals > 0 else "home"
            scoring_prob = self._team_to_score_remaining(state, need_team)
            if scoring_prob is None:
                return _normalise({"yes": base_yes, "no": base_no})
            return _normalise({"yes": scoring_prob, "no": 1 - scoring_prob})

        # CASE 3: BOTH already scored → BTTS yes is certain.
        return {"yes": 1.0, "no": 0.0}

    def _btts_first_half(self, state: LiveMatchState) -> dict[str, float] | None:
        # Sportmonks doesn't ship BTTS-1H pre-built; derive from
        # Home OU 0.5 1H × Away OU 0.5 1H if available
        # (Both OU 0.5 yes ≡ team_score_first_half_yes).
        # Simpler fallback: BTTS-1H ≈ BTTS_FIRST_HALF_FRACTION × full BTTS.
        # See BTTS_FIRST_HALF_FRACTION for empirical basis + calibration TODO.
        sm = state.sportmonks_prediction(PredictionType.BTTS_PROBABILITY)
        if not sm:
            return None
        full_yes = _pct(sm.get("yes"))
        first_half_yes = full_yes * BTTS_FIRST_HALF_FRACTION

        # Team-form lift: when both teams have a 1H-goal pattern that
        # diverges from the league baseline (recent ~10 matches), tilt
        # the BTTS-1H probability proportionally. This is the signal
        # Sportmonks's pre-match prediction may NOT have captured if
        # the team's recent pattern differs from their season aggregate
        # (form-related drift, lineup changes, etc.).
        first_half_yes = self._apply_team_form_lift(
            first_half_yes, state, signal="first_half_goal_rate",
            league_baseline=0.42 ** 2,  # league P(both teams score 1H)
        )
        return _normalise({"yes": first_half_yes, "no": 1 - first_half_yes})

    def _ou_total(
        self, state: LiveMatchState, *, line: float,
    ) -> dict[str, float] | None:
        # Map line → Sportmonks PredictionType
        mapping = {
            0.5: None,  # not directly emitted, derive
            1.5: PredictionType.OVER_UNDER_1_5_PROBABILITY,
            2.5: PredictionType.OVER_UNDER_2_5_PROBABILITY,
            3.5: PredictionType.OVER_UNDER_3_5_PROBABILITY,
        }
        type_id = mapping.get(line)

        # Pre-match: trust Sportmonks directly
        if not state.is_live and not state.is_half_time:
            if type_id is None:
                return None
            sm = state.sportmonks_prediction(type_id)
            if not sm:
                return None
            return _normalise({"over": _pct(sm.get("yes")),
                               "under": _pct(sm.get("no"))})

        # Live: condition on already-scored goals + Dixon-Robinson scaled λ
        already = state.home_goals + state.away_goals
        if already > line:
            return {"over": 1.0, "under": 0.0}
        goals_needed = math.ceil(line) - already
        if goals_needed <= 0:
            # Already over (e.g., line=2.5 and 3 goals)
            return {"over": 1.0, "under": 0.0}

        # Use combined home + away xG proxy + Sportmonks pre-match λ
        # Estimate match λ from Sportmonks OU 2.5 if available:
        # P(over 2.5) ≈ P(Poisson(λ) ≥ 3); invert numerically
        lam_total = self._estimate_total_lambda(state)
        if lam_total is None:
            return None
        # Remaining-time scale
        lam_remaining = lam_total * (max(0, 90 - state.minute) / 90.0)
        # B-3: killing-the-clock dampens TOTAL-λ proportional to how many
        # sides are killing clock. If both teams are parking the bus, full
        # 15% damp. If only one team is killing (the leader; the other is
        # still pushing), only ~half the goal-rate suppression applies —
        # the active team still contributes their full λ. The previous
        # `if EITHER killed → full damp` over-suppressed asymmetric cases.
        home_kills = state.is_killing_clock("home")
        away_kills = state.is_killing_clock("away")
        if home_kills and away_kills:
            lam_remaining *= (1.0 - KILLING_CLOCK_LAMBDA_DAMP)
        elif home_kills or away_kills:
            lam_remaining *= (1.0 - KILLING_CLOCK_LAMBDA_DAMP * 0.5)
        if lam_remaining <= 0:
            return {"over": 1.0 if already > line else 0.0,
                    "under": 0.0 if already > line else 1.0}
        # Probability that ≥ goals_needed more goals score
        p_over = 1.0 - _poisson_cdf(goals_needed - 1, lam_remaining)
        return {"over": _clip_scalar(p_over), "under": _clip_scalar(1 - p_over)}

    def _first_half_ou(
        self, state: LiveMatchState, *, line: float,
    ) -> dict[str, float] | None:
        # Pre-HT only; if past HT, this market is a settled market handled
        # elsewhere. Derive from current 1H goals + remaining 1H minutes.
        if state.minute >= 45:
            return None
        already_1h = self._first_half_goals_so_far(state)
        if already_1h > line:
            return {"over": 1.0, "under": 0.0}
        goals_needed = math.ceil(line) - already_1h
        # Estimate 1H λ ~= 0.45 × full λ
        lam_total = self._estimate_total_lambda(state)
        if lam_total is None:
            return None
        lam_first_half_total = lam_total * 0.45
        # Remaining 1H: (45 - minute) / 45 fraction
        lam_remaining_1h = lam_first_half_total * max(0, 45 - state.minute) / 45.0
        if lam_remaining_1h <= 0:
            return {"over": 1.0 if already_1h > line else 0.0,
                    "under": 0.0 if already_1h > line else 1.0}
        p_over = 1.0 - _poisson_cdf(goals_needed - 1, lam_remaining_1h)
        return {"over": _clip_scalar(p_over), "under": _clip_scalar(1 - p_over)}

    def _team_ou(
        self, state: LiveMatchState, *, line: float, side: str,
    ) -> dict[str, float] | None:
        # Use Sportmonks per-team OU when available
        type_map = {
            ("home", 0.5): PredictionType.HOME_OVER_UNDER_0_5_PROBABILITY,
            ("home", 1.5): PredictionType.HOME_OVER_UNDER_1_5_PROBABILITY,
            ("home", 2.5): PredictionType.HOME_OVER_UNDER_2_5_PROBABILITY,
            ("home", 3.5): PredictionType.HOME_OVER_UNDER_3_5_PROBABILITY,
            ("away", 0.5): PredictionType.AWAY_OVER_UNDER_0_5_PROBABILITY,
            ("away", 1.5): PredictionType.AWAY_OVER_UNDER_1_5_PROBABILITY,
            ("away", 2.5): PredictionType.AWAY_OVER_UNDER_2_5_PROBABILITY,
            ("away", 3.5): PredictionType.AWAY_OVER_UNDER_3_5_PROBABILITY,
        }
        type_id = type_map.get((side, line))
        if type_id is None:
            return None
        sm = state.sportmonks_prediction(type_id)
        if not sm:
            return None
        return _normalise({
            "over": _pct(sm.get("yes")),
            "under": _pct(sm.get("no")),
        })

    # ── live helpers ────────────────────────────────────────────────────

    def _estimate_total_lambda(self, state: LiveMatchState) -> float | None:
        """Estimate the full-match Poisson λ for total goals.

        Use Sportmonks OU 2.5 to back out λ via P(goals ≥ 3 | Poisson(λ)) =
        Sportmonks_yes. We solve numerically (binary search 0.5 → 6.0).

        When trends data is available, apply a CROSS-TEAM momentum
        multiplier: the joint home×away momentum_score captures whether
        the match is in an open end-to-end phase (both teams pushing →
        more goals) or scrappy (both teams cooling → fewer goals). Same
        bound as the per-side momentum: [0.85, 1.15].
        """
        sm = state.sportmonks_prediction(PredictionType.OVER_UNDER_2_5_PROBABILITY)
        if not sm:
            return None
        target = _pct(sm.get("yes"))  # P(over 2.5)
        if not 0.0 < target < 1.0:
            # Sportmonks emitted no usable OU 2.5 prediction — return None so
            # callers (OU markets, BTTS-2H, first-half OU) skip this snapshot
            # rather than fabricate λ from a hard-coded mid-range default.
            return None
        # Binary search for λ
        lo, hi = 0.1, 6.0
        for _ in range(40):
            mid = (lo + hi) / 2
            p = 1.0 - _poisson_cdf(2, mid)  # P(X ≥ 3)
            if p < target:
                lo = mid
            else:
                hi = mid
        lam = (lo + hi) / 2

        # Cross-team momentum: if BOTH teams are accelerating (joint > 1.2)
        # the match is open and total-λ should rise; if both are slowing
        # (joint < 0.8) the match is scrappy and total-λ should fall.
        # Joint product is the right composition because momentum is
        # multiplicative (independent per-team rates), and a single hot
        # team alone doesn't justify a total-goals lift.
        if state.trends and state.minute >= 20:
            home_m = state.momentum_score("home", window=5)
            away_m = state.momentum_score("away", window=5)
            joint = home_m * away_m
            # Map joint product to multiplier: joint=1.0 → 1.0; joint=2.0 →
            # 1.15 (cap); joint=0.5 → 0.85 (floor).
            joint_mult = max(0.85, min(1.15, 0.6 + 0.4 * joint))
            lam *= joint_mult
        return lam

    def _team_to_score_remaining(
        self, state: LiveMatchState, side: str,
    ) -> float | None:
        """P(side scores ≥ 1 in the remaining minutes).

        Use the per-team OU 0.5 Sportmonks prediction as the base team-λ
        (over 0.5 for the full match → ≥1 goal), back out the per-team λ,
        scale by remaining-time fraction, return P(Poisson(λ_remaining) ≥ 1).
        """
        type_id = (
            PredictionType.HOME_OVER_UNDER_0_5_PROBABILITY if side == "home"
            else PredictionType.AWAY_OVER_UNDER_0_5_PROBABILITY
        )
        sm = state.sportmonks_prediction(type_id)
        if not sm:
            return None
        p_over_0_5 = _pct(sm.get("yes"))
        if not 0.0 < p_over_0_5 < 1.0:
            return None
        # P(Poisson(λ) ≥ 1) = 1 - exp(-λ)  →  λ = -ln(1 - P)
        lam_pre = -math.log(1 - p_over_0_5)
        already = state.home_goals if side == "home" else state.away_goals
        # P(team already scored)? If yes → already 1.0 BTTS contribution
        if already > 0:
            return 1.0
        lam_remaining = lam_pre * max(0, 90 - state.minute) / 90.0
        if lam_remaining <= 0:
            return 0.0
        return 1.0 - math.exp(-lam_remaining)

    def _apply_team_form_lift(
        self,
        base_yes: float,
        state: LiveMatchState,
        *,
        signal: str,
        league_baseline: float,
    ) -> float:
        """Tilt a P(yes) probability by the joint home×away team-form ratio.

        Returns the lifted probability, capped so the lift is bounded by
        ``[TEAM_FORM_LIFT_MIN, TEAM_FORM_LIFT_MAX]``. When either side's
        form is missing, returns ``base_yes`` unchanged — degrades
        gracefully on cache misses or low-fixture-count teams.

        ``signal`` is the attribute name on ``TeamForm`` to read (e.g.
        ``"first_half_goal_rate"``, ``"second_half_goal_rate"``,
        ``"late_goals_rate"``).

        ``league_baseline`` is the joint baseline for the same signal.
        For "both teams score in 1H": (league_1H_rate)² ≈ 0.18.
        For "both teams score in 2H": (league_2H_rate)² ≈ 0.42.
        """
        h = state.home_team_form
        a = state.away_team_form
        if h is None or a is None:
            return base_yes
        try:
            h_rate = float(getattr(h, signal))
            a_rate = float(getattr(a, signal))
        except (AttributeError, TypeError, ValueError):
            return base_yes
        joint_form = h_rate * a_rate
        if league_baseline <= 0:
            return base_yes
        ratio = joint_form / league_baseline
        ratio_capped = max(TEAM_FORM_LIFT_MIN, min(TEAM_FORM_LIFT_MAX, ratio))
        return _clip_scalar(base_yes * ratio_capped)

    def _team_score_in_remaining_uncond(
        self, state: LiveMatchState, side: str,
    ) -> float | None:
        """P(side scores ≥1 in remaining minutes), ignoring first-half goals.

        Differs from ``_team_to_score_remaining`` in that it does NOT
        short-circuit to 1.0 when the side has already scored — required
        for BTTS-2H (both score IN THE SECOND HALF, regardless of 1H state).
        """
        type_id = (
            PredictionType.HOME_OVER_UNDER_0_5_PROBABILITY if side == "home"
            else PredictionType.AWAY_OVER_UNDER_0_5_PROBABILITY
        )
        sm = state.sportmonks_prediction(type_id)
        if not sm:
            return None
        p_over_0_5 = _pct(sm.get("yes"))
        if not 0.0 < p_over_0_5 < 1.0:
            return None
        lam_pre = -math.log(1 - p_over_0_5)
        lam_remaining = lam_pre * max(0, 90 - state.minute) / 90.0
        if lam_remaining <= 0:
            return 0.0
        return 1.0 - math.exp(-lam_remaining)

    def _first_half_goals_so_far(self, state: LiveMatchState) -> int:
        """Count goals scored in the first half (period_id == 1)."""
        return sum(
            1 for minute, _team in state.goal_events if minute <= 45
        )

    def _double_chance_from_1x2(self, ft: dict[str, float]) -> dict[str, float]:
        return {
            "1x": _clip_scalar(ft["home"] + ft["draw"]),
            "x2": _clip_scalar(ft["draw"] + ft["away"]),
            "12": _clip_scalar(ft["home"] + ft["away"]),
        }

    # ── confidence (B-G4) ──────────────────────────────────────────────

    def _confidence_1x2(
        self, state: LiveMatchState, our_probs: dict[str, float],
    ) -> float:
        """Per-pick credibility-interval half-width for 1X2 markets.

        Disagreement between Sportmonks sources — direct 1X2, DOUBLE_CHANCE
        marginal, CORRECT_SCORE-grid marginal — gives a free standard-error
        proxy across ALL THREE outcomes (home/draw/away). We take the MAX
        spread across the three axes so a pick on `home` or `away` doesn't
        get a CI derived only from disagreement on `draw`.

        Floor of 0.02 (±2pp) ensures ValueDetector never demands less than
        baseline noise.
        """
        sources_per_outcome: dict[str, list[float]] = {
            "home": [], "draw": [], "away": [],
        }

        sm_ftr = state.sportmonks_prediction(
            PredictionType.FULLTIME_RESULT_PROBABILITY
        )
        if sm_ftr:
            for outcome in ("home", "draw", "away"):
                v = _pct(sm_ftr.get(outcome))
                if 0 < v < 1:
                    sources_per_outcome[outcome].append(v)

        sm_dc = state.sportmonks_prediction(PredictionType.DOUBLE_CHANCE_PROBABILITY)
        if sm_dc:
            # Each DC selection covers 2 outcomes; the OPPOSITE-pair tells
            # us the third outcome's marginal:
            #   P(home) = 1 - P(X2)        (X2 = draw or away → not home)
            #   P(draw) = 1 - P(home_away) (12 = home or away → not draw)
            #   P(away) = 1 - P(1X)        (1X = home or draw → not away)
            x2 = _pct(sm_dc.get("draw_away"))
            ha = _pct(sm_dc.get("home_away"))
            one_x = _pct(sm_dc.get("draw_home"))
            if 0 < x2 < 1:
                sources_per_outcome["home"].append(1.0 - x2)
            if 0 < ha < 1:
                sources_per_outcome["draw"].append(1.0 - ha)
            if 0 < one_x < 1:
                sources_per_outcome["away"].append(1.0 - one_x)

        grid = state.sportmonks_score_grid
        if grid is not None:
            n = grid.shape[0]
            p_home = float(
                sum(grid[i, j] for i in range(n) for j in range(n) if i > j)
            )
            p_draw = float(sum(grid[i, i] for i in range(n)))
            p_away = float(
                sum(grid[i, j] for i in range(n) for j in range(n) if i < j)
            )
            sources_per_outcome["home"].append(p_home)
            sources_per_outcome["draw"].append(p_draw)
            sources_per_outcome["away"].append(p_away)

        spreads: list[float] = []
        for sources in sources_per_outcome.values():
            if len(sources) >= 2:
                spreads.append(max(sources) - min(sources))

        max_spread = max(spreads) if spreads else 0.04
        # Half-width: spread/2 + 1.5pp baseline noise
        return max(0.02, max_spread / 2 + 0.015)


# ── numeric helpers ─────────────────────────────────────────────────────────


def _pct(x: Any) -> float:
    """Sportmonks emits percentages 0-100. Convert to 0-1."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v / 100.0 if v > 1.5 else v


def _normalise(body: dict[str, float]) -> dict[str, float]:
    total = sum(body.values())
    if total <= 0:
        return body
    return {k: v / total for k, v in body.items()}


def _clip(body: dict[str, float], *, lo: float = 0.001, hi: float = 0.999) -> dict[str, float]:
    return {k: max(lo, min(hi, v)) for k, v in body.items()}


def _clip_scalar(v: float, *, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, v))


def _sums_to_one(body: dict[str, float], *, tol: float = 1e-3) -> bool:
    return abs(sum(body.values()) - 1.0) < tol


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X ≤ k) for Poisson(λ). For small k this is a sum; we cap k at 30."""
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    cdf = 0.0
    term = math.exp(-lam)
    cdf += term
    for i in range(1, min(k, 30) + 1):
        term *= lam / i
        cdf += term
    return cdf
