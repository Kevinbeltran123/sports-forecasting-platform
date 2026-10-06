"""Walk-forward CLV backtest math -- ML-02.

CLV staked odds MUST use opening odds x (1 - slippage_pct), NOT closing odds.
Retrofitting closing-odds CLV later invalidates every earlier backtest result.
"""

from __future__ import annotations

import numpy as np

SLIPPAGE_PCT: float = 0.015          # 1.5% -- mid-range of 1-2% spec (ML-02)
EDGE_THRESHOLD_PCT: float = 0.05     # D-10 — Phase 3 pick engine imports this


def apply_slippage(opening_odds: float) -> float:
    """Bet-taker loses SLIPPAGE_PCT of odds edge to market movement."""
    return opening_odds * (1.0 - SLIPPAGE_PCT)


def compute_clv(staked_odds: float, pinnacle_closing: float) -> float:
    """CLV percentage vs Pinnacle closing (ML-02 / CLV-02).

    CLV% = (staked_odds / pinnacle_closing - 1) * 100.
    Positive -> the bet was taken at a better price than the closing line.
    """
    return (staked_odds / pinnacle_closing - 1.0) * 100.0


def simulate_pick(
    model_probs: np.ndarray,
    opening_odds: np.ndarray,
    threshold: float = EDGE_THRESHOLD_PCT,
) -> int | None:
    """Select the outcome index (0=home, 1=draw, 2=away) with maximum positive edge.

    D-09 + D-10. Phase 3 pick engine (PICK-01) imports this function so that
    backtest CLV and live-pick selection use IDENTICAL edge-filter logic. Do
    NOT duplicate the threshold or the formula anywhere else.

    Returns None when:
      - any opening_odds value is NaN (Betano coverage gap; D-03), or
      - the max edge is below the threshold.

    Args:
        model_probs: shape (3,) — calibrated probabilities for [home, draw, away].
        opening_odds: shape (3,) — Betano decimal opening odds for [home, draw, away].
            NaN entries are treated as missing; the row is skipped (returns None).
        threshold: minimum edge (default EDGE_THRESHOLD_PCT = 0.05 per D-10).

    Returns:
        Outcome index 0/1/2, or None if no outcome clears the threshold.
    """
    if np.any(np.isnan(opening_odds)):
        return None
    edge = model_probs * opening_odds - 1.0
    if float(np.max(edge)) < threshold:
        return None
    return int(np.argmax(edge))
