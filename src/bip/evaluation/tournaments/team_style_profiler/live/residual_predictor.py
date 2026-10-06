"""Residual live predictor — re-model markets given current MatchState.

OPERATOR DECISION (locked 2026-05-24): re-model minute-by-minute using
residual λ adjusted by current state.

Method:
  1. Pre-match λ's are derived per cross_team_predictor.
  2. Residual λ = λ_pre_match × (minutes_remaining / 90).
     I.e., we assume the team will continue to score at the same average
     rate over the remaining time.
  3. FT_goals = current_goals + residual_goals, where residual_goals ~
     Bivariate Poisson(λ_residual_home, λ_residual_away).
  4. P(FT >= 2.5) = sum over (h_res, a_res) of P(current + (h_res, a_res) > 2.5).
  5. Similar for BTTS (need both teams to have >=1 goal at FT) and AH.

The residual model captures the "what can still happen" aspect cleanly.
For markets that resolve at FT (most of our markets), this is correct.
For live in-running markets (next goal, time to next event), we'd need
a different model — out of scope for Fase 6 MVP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    MarketPredictions,
)
from bip.evaluation.tournaments.team_style_profiler.live.match_state import (
    MatchState,
)
from bip.evaluation.tournaments.team_style_profiler.predictor_helpers import (
    bivariate_poisson_grid,
    p_btts,
    p_home_covers_ah,
    p_over_total,
    poisson_sf,
)


@dataclass(frozen=True)
class LiveMarketProb:
    """A live market probability snapshot."""

    market: str
    probability: float
    confidence: str
    rationale: str


@dataclass(frozen=True)
class LiveMarketPredictions:
    """Bundle of FT-resolving market probabilities recomputed live."""

    fixture_id: int
    minute: int
    home_goals: int
    away_goals: int
    lambda_residual_home: float
    lambda_residual_away: float

    over_2_5: LiveMarketProb
    over_3_5: LiveMarketProb
    under_2_5: LiveMarketProb
    btts_yes: LiveMarketProb
    btts_no: LiveMarketProb
    ah_home_minus_0_5: LiveMarketProb
    ah_home_minus_1_5: LiveMarketProb
    cards_over_3_5: LiveMarketProb | None
    cards_over_4_5: LiveMarketProb | None

    @property
    def all_markets(self) -> list[LiveMarketProb]:
        out: list[LiveMarketProb] = [
            self.over_2_5, self.over_3_5, self.under_2_5,
            self.btts_yes, self.btts_no,
            self.ah_home_minus_0_5, self.ah_home_minus_1_5,
        ]
        for m in [self.cards_over_3_5, self.cards_over_4_5]:
            if m is not None:
                out.append(m)
        return out


def predict_live_markets(
    pre_match: MarketPredictions,
    state: MatchState,
    pre_match_cards_lambda_total: float | None = None,
    rho: float = 0.0,
) -> LiveMarketPredictions:
    """Recompute markets given current match state.

    Args:
        pre_match: pre-match MarketPredictions (from cross_team_predictor).
        state: current MatchState.
        pre_match_cards_lambda_total: optional λ used for the pre-match
            cards prediction (sum of both teams' yellow_cards_per_match).
            If None, cards markets are skipped (we don't have access to the
            raw λ from MarketPredictions because cards weren't bivariate).
        rho: bivariate Poisson correlation. Default 0.0 matches lock_v1.

    Returns:
        LiveMarketPredictions with FT-resolving probabilities updated.
    """
    minutes_remaining = state.minutes_remaining
    # Residual time fraction
    if minutes_remaining <= 0 or state.is_finished:
        # Match is done — return degenerate probs based on current score
        final_total = state.home_goals + state.away_goals
        final_btts = (state.home_goals >= 1) and (state.away_goals >= 1)
        return _degenerate_predictions(state, pre_match)

    if state.is_pre_match:
        # Equivalent to the pre-match prediction; just lift the values.
        residual_frac = 1.0
    else:
        residual_frac = minutes_remaining / 90.0

    # Scale pre-match λ's by residual fraction
    lam_res_h = pre_match.lambda_home * residual_frac
    lam_res_a = pre_match.lambda_away * residual_frac
    lam_res_h = max(lam_res_h, 0.001)
    lam_res_a = max(lam_res_a, 0.001)

    # Build residual scoreline distribution
    grid_res = bivariate_poisson_grid(lam_res_h, lam_res_a, rho=rho, max_goals=10)

    cur_h, cur_a = state.home_goals, state.away_goals

    # --- Goals O/U (FT total) ---
    # FT total > line  <=>  current_total + res_total > line
    # residual matrix index (h_res, a_res) -> total contrib h_res + a_res
    cur_total = cur_h + cur_a

    def _ft_over(line: float) -> float:
        # We need P((cur_h + h_res) + (cur_a + a_res) > line) = P(h_res + a_res > line - cur_total)
        target = line - cur_total
        if target < 0:
            return 1.0  # already exceeded
        # Sum residual grid entries where (h+a) > target
        return p_over_total(grid_res, target)

    p_over_25 = _ft_over(2.5)
    p_over_35 = _ft_over(3.5)
    p_under_25 = 1.0 - p_over_25

    # --- BTTS ---
    # FT both teams >= 1: handle the 4 quadrants based on current score
    def _ft_btts() -> float:
        h_already = cur_h >= 1
        a_already = cur_a >= 1
        if h_already and a_already:
            return 1.0  # already BTTS
        # P(home_res >= 1 if home not yet scored) - need this for both teams
        # We use the marginal residual Poissons
        # P(home_total >= 1) = if already, 1.0 else (1 - P(home_res = 0))
        from bip.evaluation.tournaments.team_style_profiler.predictor_helpers import (
            poisson_pmf,
        )
        p_h_score = 1.0 if h_already else (1.0 - poisson_pmf(0, lam_res_h))
        p_a_score = 1.0 if a_already else (1.0 - poisson_pmf(0, lam_res_a))
        return p_h_score * p_a_score

    p_btts_yes = _ft_btts()
    p_btts_no = 1.0 - p_btts_yes

    # --- AH (home perspective) ---
    # FT (cur_h + h_res) - (cur_a + a_res) > line means (h_res - a_res) > line - (cur_h - cur_a)
    cur_diff = cur_h - cur_a

    def _ft_ah_home(line: float) -> float:
        # We want P(home covers AH line). predict_asian_handicap convention:
        # line < 0 (e.g. -0.5) means home needs to outscore by |line|+ε.
        # p_home_covers_ah(grid, threshold) computes P(h - a > threshold).
        # For FT covering: P((cur_h + h_res) - (cur_a + a_res) > -line)
        # i.e., (h_res - a_res) > -line - cur_diff
        threshold = -line - cur_diff
        return p_home_covers_ah(grid_res, threshold)

    p_ah_home_05 = _ft_ah_home(-0.5)
    p_ah_home_15 = _ft_ah_home(-1.5)

    # --- Cards ---
    cards_o35 = cards_o45 = None
    if pre_match_cards_lambda_total is not None:
        # Residual cards λ
        lam_cards_res = pre_match_cards_lambda_total * residual_frac
        cur_cards = state.total_yellow_cards
        # P(FT total > line) = P(cur_cards + res_cards > line) = P(res_cards > line - cur_cards)
        if lam_cards_res > 0.001:
            cards_o35 = LiveMarketProb(
                market="cards_O3.5",
                probability=poisson_sf(int(3.5 - cur_cards), lam_cards_res) if (3.5 - cur_cards) >= 0 else 1.0,
                confidence="MEDIUM",
                rationale=f"residual cards Poisson(λ={lam_cards_res:.2f}), cur={cur_cards}",
            )
            cards_o45 = LiveMarketProb(
                market="cards_O4.5",
                probability=poisson_sf(int(4.5 - cur_cards), lam_cards_res) if (4.5 - cur_cards) >= 0 else 1.0,
                confidence="MEDIUM",
                rationale=f"residual cards Poisson(λ={lam_cards_res:.2f}), cur={cur_cards}",
            )

    rationale = (
        f"residual @ min {state.minute}: "
        f"λ_res_h={lam_res_h:.2f} λ_res_a={lam_res_a:.2f}, "
        f"score {cur_h}-{cur_a}"
    )

    return LiveMarketPredictions(
        fixture_id=state.fixture_id,
        minute=state.minute,
        home_goals=cur_h,
        away_goals=cur_a,
        lambda_residual_home=lam_res_h,
        lambda_residual_away=lam_res_a,
        over_2_5=LiveMarketProb("O2.5", p_over_25, pre_match.over_2_5.confidence, rationale),
        over_3_5=LiveMarketProb("O3.5", p_over_35, pre_match.over_3_5.confidence, rationale),
        under_2_5=LiveMarketProb("U2.5", p_under_25, pre_match.under_2_5.confidence, rationale),
        btts_yes=LiveMarketProb("BTTS_yes", p_btts_yes, pre_match.btts_yes.confidence, rationale),
        btts_no=LiveMarketProb("BTTS_no", p_btts_no, pre_match.btts_no.confidence, rationale),
        ah_home_minus_0_5=LiveMarketProb(
            "AH_home_-0.5", p_ah_home_05,
            pre_match.ah_home_minus_0_5.confidence if pre_match.ah_home_minus_0_5 else "MEDIUM",
            rationale,
        ),
        ah_home_minus_1_5=LiveMarketProb(
            "AH_home_-1.5", p_ah_home_15,
            pre_match.ah_home_minus_1_5.confidence if pre_match.ah_home_minus_1_5 else "MEDIUM",
            rationale,
        ),
        cards_over_3_5=cards_o35,
        cards_over_4_5=cards_o45,
    )


def _degenerate_predictions(
    state: MatchState, pre_match: MarketPredictions
) -> LiveMarketPredictions:
    """Once match is finished, all probs degenerate to 0 or 1."""
    cur_h, cur_a = state.home_goals, state.away_goals
    total = cur_h + cur_a
    diff = cur_h - cur_a
    btts = (cur_h >= 1) and (cur_a >= 1)
    rationale = f"match finished; score {cur_h}-{cur_a}"
    one = lambda mk, p: LiveMarketProb(mk, p, "HIGH", rationale)
    return LiveMarketPredictions(
        fixture_id=state.fixture_id,
        minute=state.minute,
        home_goals=cur_h,
        away_goals=cur_a,
        lambda_residual_home=0.0,
        lambda_residual_away=0.0,
        over_2_5=one("O2.5", 1.0 if total > 2.5 else 0.0),
        over_3_5=one("O3.5", 1.0 if total > 3.5 else 0.0),
        under_2_5=one("U2.5", 1.0 if total < 2.5 else 0.0),
        btts_yes=one("BTTS_yes", 1.0 if btts else 0.0),
        btts_no=one("BTTS_no", 0.0 if btts else 1.0),
        ah_home_minus_0_5=one("AH_home_-0.5", 1.0 if diff > 0.5 else 0.0),
        ah_home_minus_1_5=one("AH_home_-1.5", 1.0 if diff > 1.5 else 0.0),
        cards_over_3_5=None,
        cards_over_4_5=None,
    )
