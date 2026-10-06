"""Value detector — Z-score gate against Pinnacle odds.

OPERATOR DECISIONS (locked 2026-05-24):
  - Edge metric: Z-score = (P_emp - P_implied) / std_emp >= 1.5
  - std_emp derived from CI half-width: std = ci_half_width / 1.96
    (assuming CI95 is symmetric and normal — approximation).
  - Confidence gate: alert only for HIGH + MEDIUM (skip LOW).
  - No stake recommendation — operator decides amount.
"""
from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    MarketPredictions,
    MarketProb,
)
from bip.evaluation.tournaments.team_style_profiler.odds_provider import (
    implied_probability,
)


MIN_Z_SCORE = 1.5
"""Operator decision: edge triggers at Z >= 1.5."""

ACCEPTED_CONFIDENCE = {"HIGH", "MEDIUM"}
"""Confidence categories that pass the gate."""


@dataclass(frozen=True)
class ValueAlert:
    """A single value-betting alert for one market."""

    market: str
    decision: str
    """ALERT (passes gate) | SKIP_CONFIDENCE | SKIP_NO_ODDS | SKIP_EDGE."""
    p_empirical: float
    p_implied: float
    std_empirical: float
    decimal_odd: float | None
    z_score: float | None
    edge_pct: float | None
    """ROI per unit: P_emp * odd - 1. Reported alongside z-score for clarity."""
    confidence: str
    rationale: str


def _std_from_market(prob: MarketProb, market_prob_value: float | None = None) -> float:
    """Approximate std from confidence label.

    Confidence -> CI half-width mapping (per cross_team_predictor):
      HIGH:   ci_width <= 0.10  -> std ~= 0.05/1.96
      MEDIUM: ci_width <= 0.20  -> std ~= 0.10/1.96
      LOW:    ci_width >  0.20  -> std ~= 0.15/1.96

    This is an approximation — the underlying CI was computed on a
    bootstrap of the per-team rates, and the propagation through the
    cross-team predictor isn't perfectly preserved as a CI. We use
    confidence label as proxy.
    """
    half_widths = {"HIGH": 0.05, "MEDIUM": 0.10, "LOW": 0.15}
    return half_widths.get(prob.confidence, 0.15) / 1.96


def _z_score(p_emp: float, p_implied: float, std_emp: float) -> float:
    if std_emp <= 0:
        return 0.0
    return (p_emp - p_implied) / std_emp


def _evaluate_market(
    market_prob: MarketProb,
    odds: dict[str, float],
) -> ValueAlert:
    """Apply the value-detection rules to a single market prediction."""
    market = market_prob.market
    decimal_odd = odds.get(market)
    p_emp = market_prob.probability
    std_emp = _std_from_market(market_prob)

    if decimal_odd is None:
        return ValueAlert(
            market=market,
            decision="SKIP_NO_ODDS",
            p_empirical=p_emp,
            p_implied=0.0,
            std_empirical=std_emp,
            decimal_odd=None,
            z_score=None,
            edge_pct=None,
            confidence=market_prob.confidence,
            rationale=f"No Pinnacle odd parsed for market '{market}'",
        )

    p_implied = implied_probability(decimal_odd)
    z = _z_score(p_emp, p_implied, std_emp)
    edge = p_emp * decimal_odd - 1.0

    if market_prob.confidence not in ACCEPTED_CONFIDENCE:
        decision = "SKIP_CONFIDENCE"
        rationale = (
            f"Confidence {market_prob.confidence} below MEDIUM gate. "
            f"Z={z:.2f}, edge={edge:.2%} (suppressed)."
        )
    elif z < MIN_Z_SCORE:
        decision = "SKIP_EDGE"
        rationale = (
            f"Z={z:.2f} below {MIN_Z_SCORE} threshold. "
            f"P_emp={p_emp:.3f}, P_implied={p_implied:.3f}, "
            f"edge={edge:.2%}. Confidence={market_prob.confidence}."
        )
    else:
        decision = "ALERT"
        rationale = (
            f"Z={z:.2f} >= {MIN_Z_SCORE}. "
            f"P_emp={p_emp:.3f} vs P_implied={p_implied:.3f} "
            f"(cuota {decimal_odd:.2f}). "
            f"Edge: {edge:+.2%}. Confidence={market_prob.confidence}. "
            f"{market_prob.rationale}"
        )

    return ValueAlert(
        market=market,
        decision=decision,
        p_empirical=p_emp,
        p_implied=p_implied,
        std_empirical=std_emp,
        decimal_odd=decimal_odd,
        z_score=z,
        edge_pct=edge,
        confidence=market_prob.confidence,
        rationale=rationale,
    )


def detect_value(
    predictions: MarketPredictions,
    pinnacle_odds: dict[str, float],
) -> list[ValueAlert]:
    """Run the value detector across all predicted markets for a fixture.

    Args:
        predictions: MarketPredictions from cross_team_predictor.
        pinnacle_odds: dict of canonical_market_name -> decimal odd,
                       typically from extract_pinnacle_canonical_odds().

    Returns:
        List of ValueAlert — one per market (including SKIPs for audit).
        Filter by ``decision == 'ALERT'`` to get the picks.
    """
    return [_evaluate_market(mp, pinnacle_odds) for mp in predictions.all_markets]


def filter_alerts(alerts: list[ValueAlert]) -> list[ValueAlert]:
    """Convenience: return only ALERT decisions."""
    return [a for a in alerts if a.decision == "ALERT"]
