"""Conditional Predictor — section 3 of the v3 engine design notes.

For each ``(thesis, market, GSV)`` we want ``P(outcome | GSV, horizon)``.

Sec 3 architectural rule: one predictor PER MARKET FAMILY, not a
monolith. Goals/BTTS uses a time-decay hazard model conditioned on
state. Corners uses a Poisson rate model. Cards likewise. Next-X uses
a hazard-point model. Props use shrinkage from pre-match priors.

This Phase-1 scaffold ships **two families**: corners and goals-2H.
Both implement a common protocol so the Market Selector can call them
uniformly. The implementations are **deliberately simple** — Phase 2
swaps them for the real calibrated estimators per sec 5 (state
bucketing + hierarchical pooling + isotonic per minute-bucket).

The protocol is what unblocks the architecture: get the **types**
right now and the predictor implementations can swap in without
touching the selector or the gate.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.calibrator import IsotonicCalibrator


@dataclass(frozen=True)
class PredictionPoint:
    """Predicted probability + uncertainty band.

    ``ci_low`` / ``ci_high`` reflect the predictor's own confidence; the
    No-Bet gate rule #8 reads these to abort overly-uncertain calls."""

    p: float
    ci_low: float
    ci_high: float
    family: MarketFamily


class FamilyPredictor(Protocol):
    """Each market-family predictor implements this single method."""

    family: MarketFamily

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        ...


# ──────────────────────────────────────────────────────────────────────
# Corners family
# ──────────────────────────────────────────────────────────────────────


def _poisson_ge(lam: float, k: int) -> float:
    """P(X >= k) for Poisson(λ). Uses CDF complement."""
    if lam <= 0:
        return 0.0
    s = 0.0
    pmf = math.exp(-lam)
    for i in range(0, k):
        s += pmf
        pmf *= lam / (i + 1)
    return max(0.0, 1.0 - s)


def _parse_line(market_id: str) -> float | None:
    """Best-effort: pull the Over/Under line value from a market id like
    ``match_corners_over_10.5``. Returns None when ambiguous."""
    parts = market_id.replace("_", " ").split()
    for p in reversed(parts):
        try:
            v = float(p)
            return v
        except ValueError:
            continue
    return None


def _parse_side_a(market_id: str) -> str | None:
    """Identify which side of an Over/Under market is side_a.

    Returns ``"over"`` if the market_id encodes the Over side, ``"under"``
    if it encodes the Under side, ``None`` if the side is not a plain
    Over/Under (e.g. Asian goal-line handicaps like
    ``1st_half_goal_line_(0-0)_over_0.5`` carry a starting-score prefix
    that changes the resolution and must be skipped).
    """
    mid = market_id.lower()
    # Reject Asian goal-line / score-handicap variants that embed a
    # starting score "(X-Y)" before the over/under token. Resolving these
    # correctly requires the full handicap logic, which the Phase-1
    # Poisson predictor does not implement.
    import re
    if re.search(r"\([0-9]+-[0-9]+\)", mid):
        return None
    if "_over_" in mid or mid.endswith("_over") or mid.startswith("over_"):
        return "over"
    if "_under_" in mid or mid.endswith("_under") or mid.startswith("under_"):
        return "under"
    return None


# Map thesis direction labels to the over/under market side they support.
# - "over" → over markets only
# - "under" / "no" → under markets only
# - "home" / "away" (NEXT_GOAL theses): "another goal coming" → over only
_DIRECTION_TO_SIDE: dict[str, str] = {
    "over": "over",
    "under": "under",
    "no": "under",
    "yes": "over",
    "home": "over",
    "away": "over",
}


class CornersPredictor:
    """Poisson rate model for corners markets.

    The live rate is built from:
    - Pre-match expected corners (``priors.expected_corners_total``)
    - The observed live rate (``corners_to_minute / minute × 90``)
    - A blending weight from ``minute / (minute + 18)`` — taken from
      the operator's existing predictor's `CORNERS_PRIOR_WEIGHT_MINUTES`

    Then scaled forward over the remaining horizon and combined with
    the **thesis adjustment**: if a relevant archetype is active, the
    rate is nudged by the thesis ``magnitude_pp`` (interpreted as a
    fractional rate change since corners are continuous, not bounded).
    """

    family = MarketFamily.CORNERS

    def __init__(self, prior_blend_minutes: float = 18.0) -> None:
        self.prior_blend_minutes = prior_blend_minutes

    def _expected_total(self, gsv: GameStateVector) -> float:
        prior = gsv.priors.expected_corners_total
        observed = gsv.corners.corners_home + gsv.corners.corners_away
        minute = max(1, gsv.time.minute)
        rate_observed = observed / minute * 90.0
        w_minute = minute / (minute + self.prior_blend_minutes)
        return w_minute * rate_observed + (1.0 - w_minute) * prior

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        """Predict P(market side_a outcome) for a CORNERS-family market.

        Same alignment contract as Goals2HPredictor:
        - market_id must parse to a plain Over/Under (no handicap)
        - thesis.direction must match market side_a (mismatch → None)
        - resolved markets (already past line) → None
        """
        if thesis.prediction.family not in (MarketFamily.CORNERS, MarketFamily.NEXT_CORNER):
            return None
        # Defensive: refuse to score a market whose id maps to anything
        # other than CORNERS / NEXT_CORNER. Prevents "5.5 in
        # alternative_match_goals_over_5.5 parsed as a corners line"
        # type errors. The cost is one extra string-match per call.
        from bip.evaluation.live.engine_v3.market_selector import (
            family_for_market_id,
        )
        market_family = family_for_market_id(market_id)
        if market_family not in (MarketFamily.CORNERS, MarketFamily.NEXT_CORNER):
            return None
        line = _parse_line(market_id)
        side = _parse_side_a(market_id)
        if line is None or side is None:
            return None
        thesis_side = _DIRECTION_TO_SIDE.get(thesis.prediction.direction)
        if thesis_side is None or thesis_side != side:
            return None
        already = gsv.corners.corners_home + gsv.corners.corners_away
        target = int(math.ceil(line)) if line != int(line) else int(line) + 1
        if side == "over" and already >= target:
            return None
        if side == "under" and already > int(line):
            return None
        expected_total = self._expected_total(gsv)
        sign = 1.0 if side == "over" else -1.0
        adjusted = expected_total * (1.0 + sign * thesis.prediction.magnitude_pp)
        remaining = max(0, gsv.time.time_remaining_match)
        future_lam = max(0.0, adjusted * remaining / 90.0)
        need = max(0, target - already)
        p_over = _poisson_ge(future_lam, need)
        p = p_over if side == "over" else max(0.0, 1.0 - p_over)
        sigma = math.sqrt(future_lam) / max(1.0, target)
        return PredictionPoint(
            p=p,
            ci_low=max(0.0, p - sigma),
            ci_high=min(1.0, p + sigma),
            family=MarketFamily.CORNERS,
        )


# ──────────────────────────────────────────────────────────────────────
# Goals-2H family
# ──────────────────────────────────────────────────────────────────────


class Goals2HPredictor:
    """Goals predictor scoped to the second-half horizon.

    Uses a Poisson model on a state-conditional λ:
    - Base λ_2H ≈ (home λ + away λ) × time_remaining / 90
    - Conditioned on ``tactical.game_phase``:
        * cagey_closed → ×0.55
        * cruise → ×0.55
        * cagey_open → ×0.80
        * open_attacking → ×1.00
        * desperate → ×1.30
    - Then nudged by the thesis magnitude_pp (in the over/under direction)
    """

    family = MarketFamily.GOALS

    _PHASE_MULT: dict[str, float] = {
        "cagey_closed": 0.55,
        "cruise": 0.55,
        "cagey_open": 0.80,
        "open_attacking": 1.00,
        "desperate": 1.30,
    }

    def _lambda_remaining(self, gsv: GameStateVector) -> float:
        prior_lam = gsv.priors.lambda_home_prematch + gsv.priors.lambda_away_prematch
        remaining = max(0, gsv.time.time_remaining_match)
        base = prior_lam * remaining / 90.0
        return base * self._PHASE_MULT.get(gsv.tactical.game_phase, 1.0)

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        """Predict P(market side_a outcome) for a GOALS-family market.

        Returns the probability of side_a (the first decimal in the
        market line). compute_mes computes ``base_edge = fair_prob -
        1/side_a_decimal``, so fair_prob MUST be aligned with side_a or
        the edge sign is meaningless.

        Routing rules:
        - thesis.family must be GOALS or NEXT_GOAL.
        - market_id must parse to an Over/Under (no Asian handicap
          starting-score prefix). Anything else → None.
        - thesis.direction (mapped via _DIRECTION_TO_SIDE) must match
          the market's side_a. Mismatch → None: the bet is on the
          opposite side and the edge cannot be evaluated by this
          predictor.

        BTTS is intentionally NOT handled here — different semantic.
        """
        if thesis.prediction.family not in (MarketFamily.GOALS, MarketFamily.NEXT_GOAL):
            return None
        line = _parse_line(market_id)
        side = _parse_side_a(market_id)
        if line is None or side is None:
            return None
        thesis_side = _DIRECTION_TO_SIDE.get(thesis.prediction.direction)
        if thesis_side is None or thesis_side != side:
            return None
        already = gsv.score.home_goals + gsv.score.away_goals
        lam = self._lambda_remaining(gsv)
        # If the line is already settled (target ≤ already for over, or
        # already > line for under), the market is resolved — refuse to
        # produce a fair_prob (book would have closed it; if it hasn't,
        # we don't bet on resolved markets).
        target = int(math.ceil(line)) if line != int(line) else int(line) + 1
        if side == "over" and already >= target:
            return None
        if side == "under" and already > int(line):
            return None
        # Adjust λ by the thesis magnitude in the thesis's predicted direction.
        sign = 1.0 if side == "over" else -1.0
        lam = max(0.0, lam * (1.0 + sign * thesis.prediction.magnitude_pp))
        need = max(0, target - already)
        p_over = _poisson_ge(lam, need)
        p = p_over if side == "over" else max(0.0, 1.0 - p_over)
        sigma = math.sqrt(max(0.01, lam)) / 4.0
        return PredictionPoint(
            p=p, ci_low=max(0.0, p - sigma), ci_high=min(1.0, p + sigma),
            family=MarketFamily.GOALS,
        )


# ──────────────────────────────────────────────────────────────────────
# BTTS family (Phase 2)
# ──────────────────────────────────────────────────────────────────────


class BTTSPredictor:
    """Both-Teams-To-Score predictor — Phase 2 scaffold.

    Models P(home scores ≥1 in remaining) and P(away scores ≥1 in
    remaining) as two independent Poissons (the standard approximation —
    a future Phase-3 implementation can use a Bivariate Poisson with
    Dixon-Coles dependence). Then:

        P(BTTS yes) = P(home_remaining ≥ 1 ∨ home_already ≥ 1)
                      × P(away_remaining ≥ 1 ∨ away_already ≥ 1)

    Under independence, with the obvious "team has already scored ⇒
    probability 1.0" branch.

    Direction handling:
    - thesis.direction ∈ {"yes", "over", "home", "away"} → predict P(yes)
    - thesis.direction ∈ {"no", "under"} → predict P(no) = 1 - P(yes)

    Routing rules:
    - thesis.family must be BTTS or GOALS (a GOALS thesis can express on
      a BTTS market: e.g. cruise_mode → under goals → BTTS No).
    - market_id family must be BTTS — defensive check.
    """

    family = MarketFamily.BTTS

    _PHASE_MULT_PER_TEAM: dict[str, float] = {
        "cagey_closed": 0.55,
        "cruise": 0.55,
        "cagey_open": 0.80,
        "open_attacking": 1.00,
        "desperate": 1.30,
    }

    def _lambda_team_remaining(self, gsv: GameStateVector, team: str) -> float:
        if team == "home":
            base_lam = gsv.priors.lambda_home_prematch
        else:
            base_lam = gsv.priors.lambda_away_prematch
        remaining = max(0, gsv.time.time_remaining_match)
        phase_mult = self._PHASE_MULT_PER_TEAM.get(gsv.tactical.game_phase, 1.0)
        return base_lam * remaining / 90.0 * phase_mult

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        if thesis.prediction.family not in (MarketFamily.BTTS, MarketFamily.GOALS):
            return None
        from bip.evaluation.live.engine_v3.market_selector import (
            family_for_market_id,
        )
        if family_for_market_id(market_id) != MarketFamily.BTTS:
            return None

        mid = market_id.lower()
        # Determine side_a of the BTTS market.
        if "_no" in mid or mid.endswith("_no") or "_yes_no" in mid:
            side_a = "no"
        elif "_yes" in mid or mid.endswith("_yes") or "_no_yes" in mid:
            side_a = "yes"
        else:
            return None

        # Map thesis.direction to which BTTS outcome we want.
        d = thesis.prediction.direction
        if d in ("yes", "over", "home", "away"):
            thesis_side = "yes"
        elif d in ("no", "under"):
            thesis_side = "no"
        else:
            return None
        if thesis_side != side_a:
            # Bet would be on side_b, but compute_mes uses side_a_decimal.
            # Skip to keep fair_prob aligned with implied(side_a).
            return None

        # P(home scores ≥1 in match) = 1 if home_goals ≥ 1, else 1 - exp(-λ_home_rem)
        if gsv.score.home_goals >= 1:
            p_home_at_least_one = 1.0
        else:
            lam_h = self._lambda_team_remaining(gsv, "home")
            p_home_at_least_one = 1.0 - math.exp(-max(0.0, lam_h))
        if gsv.score.away_goals >= 1:
            p_away_at_least_one = 1.0
        else:
            lam_a = self._lambda_team_remaining(gsv, "away")
            p_away_at_least_one = 1.0 - math.exp(-max(0.0, lam_a))
        p_yes = p_home_at_least_one * p_away_at_least_one  # independence approx
        p = p_yes if side_a == "yes" else max(0.0, 1.0 - p_yes)
        # CI: a coarse band — Phase 3 will derive from Dixon-Coles posterior.
        sigma = 0.08
        return PredictionPoint(
            p=p,
            ci_low=max(0.0, p - sigma),
            ci_high=min(1.0, p + sigma),
            family=MarketFamily.BTTS,
        )


# ──────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────


class ConditionalPredictor:
    """Composite predictor that dispatches by family.

    Designed to grow: register additional family predictors with
    ``register(predictor)``.

    Phase 2 additions:
    - BTTSPredictor (the operator's Day-1+2 worst-calibrated family)
    - Optional ``IsotonicCalibrator`` applied to every raw ``p`` before
      returning. The calibrator post-corrects systematic miscalibration
      learned from v2 historical outcomes (sec 5.4).
    """

    def __init__(
        self,
        calibrator: "IsotonicCalibrator | None" = None,
    ) -> None:
        self._predictors: dict[MarketFamily, FamilyPredictor] = {}
        self._calibrator = calibrator

    def register(self, predictor: FamilyPredictor) -> None:
        self._predictors[predictor.family] = predictor

    @property
    def calibrator(self) -> "IsotonicCalibrator | None":
        return self._calibrator

    def set_calibrator(self, calibrator: "IsotonicCalibrator | None") -> None:
        self._calibrator = calibrator

    @classmethod
    def default(cls, calibrator: "IsotonicCalibrator | None" = None) -> ConditionalPredictor:
        # Importing locally to avoid a circular dep when
        # next_goal_predictor imports PredictionPoint.
        from bip.evaluation.live.engine_v3.next_goal_predictor import (
            NextGoalPredictor,
        )

        p = cls(calibrator=calibrator)
        p.register(CornersPredictor())
        p.register(Goals2HPredictor())
        p.register(BTTSPredictor())
        p.register(NextGoalPredictor())
        return p

    def predict(
        self, thesis: Thesis, market_id: str, gsv: GameStateVector
    ) -> PredictionPoint | None:
        """Dispatch using the thesis's predicted family.

        Routing order:
        - CORNERS / NEXT_CORNER → CornersPredictor (single instance)
        - CARDS → CardsPredictor (Phase 3) if registered
        - PROPS → PlayerPropsPredictor (Phase 3, shadow-mode) if registered
        - NEXT_GOAL → NextGoalPredictor (Phase 3 hazard) if registered,
          else fallback to Goals2HPredictor (line-based, no
          "P(any goal)" shortcut — see Goals2HPredictor.predict).
        - GOALS → Goals2HPredictor
        - BTTS → BTTSPredictor (Phase 3) if registered. Goals2HPredictor
          deliberately does NOT handle BTTS — different semantic.
        """
        family = thesis.prediction.family
        raw: PredictionPoint | None
        if family in (MarketFamily.CORNERS, MarketFamily.NEXT_CORNER):
            p = self._predictors.get(MarketFamily.CORNERS)
            raw = p.predict(thesis, market_id, gsv) if p else None
        elif family == MarketFamily.CARDS:
            p = self._predictors.get(MarketFamily.CARDS)
            raw = p.predict(thesis, market_id, gsv) if p else None
        elif family == MarketFamily.PROPS:
            p = self._predictors.get(MarketFamily.PROPS)
            raw = p.predict(thesis, market_id, gsv) if p else None
        elif family == MarketFamily.BTTS:
            p = self._predictors.get(MarketFamily.BTTS)
            raw = p.predict(thesis, market_id, gsv) if p else None
        elif family == MarketFamily.NEXT_GOAL:
            from bip.evaluation.live.engine_v3.market_selector import (
                family_for_market_id,
            )
            mkt_fam = family_for_market_id(market_id)
            if mkt_fam == MarketFamily.NEXT_GOAL:
                ng = self._predictors.get(MarketFamily.NEXT_GOAL)
                raw = ng.predict(thesis, market_id, gsv) if ng else None
            elif mkt_fam == MarketFamily.GOALS:
                # Legitimate fallback: a NEXT_GOAL thesis on a goals
                # over/under market — the Goals2H predictor handles the
                # line-based mapping correctly under the new contract.
                raw = self._predictors[MarketFamily.GOALS].predict(thesis, market_id, gsv)
            else:
                raw = None
        elif family == MarketFamily.GOALS:
            # GOALS theses prefer Goals2H; if the market id resolves to
            # BTTS (e.g. cruise_mode → under goals on a BTTS_no market),
            # route through BTTSPredictor instead so the predictor and
            # market semantics line up.
            from bip.evaluation.live.engine_v3.market_selector import (
                family_for_market_id,
            )
            mkt_fam = family_for_market_id(market_id)
            if mkt_fam == MarketFamily.BTTS and MarketFamily.BTTS in self._predictors:
                raw = self._predictors[MarketFamily.BTTS].predict(thesis, market_id, gsv)
            else:
                raw = self._predictors[MarketFamily.GOALS].predict(thesis, market_id, gsv)
        else:
            raw = None

        if raw is None:
            return None
        if self._calibrator is not None and self._calibrator.is_fitted:
            # Calibrate against the MARKET family (not the thesis family) —
            # calibration is keyed by the market the bet lands on.
            from bip.evaluation.live.engine_v3.market_selector import (
                family_for_market_id,
            )
            market_family = family_for_market_id(market_id) or raw.family
            p_cal = self._calibrator.transform(raw.p, market_family, gsv.time.minute)
            return PredictionPoint(
                p=p_cal,
                ci_low=max(0.0, p_cal - (raw.p - raw.ci_low)),
                ci_high=min(1.0, p_cal + (raw.ci_high - raw.p)),
                family=raw.family,
            )
        return raw


# ──────────────────────────────────────────────────────────────────────
# Fair-prob provider — bridge from predictor to selector
# ──────────────────────────────────────────────────────────────────────


def make_fair_prob_provider(predictor: ConditionalPredictor) -> Callable[
    [Thesis, str, GameStateVector], float | None
]:
    def provider(thesis: Thesis, market_id: str, gsv: GameStateVector) -> float | None:
        result = predictor.predict(thesis, market_id, gsv)
        if result is None:
            return None
        return result.p

    return provider


__all__ = [
    "BTTSPredictor",
    "ConditionalPredictor",
    "CornersPredictor",
    "FamilyPredictor",
    "Goals2HPredictor",
    "PredictionPoint",
    "make_fair_prob_provider",
]
