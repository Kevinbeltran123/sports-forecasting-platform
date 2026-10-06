"""Stake sizing helpers: fractional Kelly, unit rounding and a market-concentration cap.

EDGE_THRESHOLD_PCT and simulate_pick live in bip.train.backtest; this module owns
sizing math only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bip.core.storage.repositories import PickRepository


# ─────────────────────────────────────────────────────────────────────────────
# Kelly sizing + 0.5-unit rounding
# ─────────────────────────────────────────────────────────────────────────────

def quarter_kelly_units(edge: float, odds: float, max_fraction: float = 0.25) -> float:
    """Quarter Kelly stake fraction.

    PICK-02: kelly_fraction = (edge / (odds - 1)) × 0.25.

    Args:
        edge: fractional edge (0.05 = 5%); negative edges clamped to 0.
        odds: decimal odds (≥ 1.0); odds ≤ 1.0 returns 0.0 to avoid div-by-zero.
        max_fraction: Kelly cap (default 0.25 = quarter Kelly).

    Returns:
        Stake as a fraction of bankroll, clamped to [0.0, max_fraction].
    """
    if odds <= 1.0:
        return 0.0
    full_kelly = edge / (odds - 1.0)
    return max(0.0, min(full_kelly, 1.0)) * max_fraction


def round_to_nearest_half_unit(stake: float) -> float:
    """Round stake to nearest 0.5 unit.

    Keeps reported stakes on a coarse, human-readable grid (0.5u instead of 0.01u).
    """
    return round(stake * 2) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Market-concentration cap: rolling 168h check (drop-on-bind)
# ─────────────────────────────────────────────────────────────────────────────

def exceeds_60pct_cap(
    pick_repo: "PickRepository",
    market: str,
    sport: str,
    hours: int = 168,
    min_sample: int = 5,
) -> bool:
    """Return True if adding this pick would push `market` past 60% of sent picks in `hours`.

    D-09: Rolling 168h window. Drop-on-bind, not defer. Phase 3 note — with 1X2-only
    markets the cap is structurally dormant (single market always 100%); the mechanism
    is built now so Phase 6 corners + future BTTS/Over-Under plug in cleanly.

    PATTERNS.md drift risk #10: query reads from Supabase (single source of truth),
    NOT Polars over Parquet — uses idx_picks_sport_market_created (migration 004).

    Args:
        pick_repo: PickRepository — provides get_window_picks(sport, hours).
        market: market this pick would be in.
        sport: sport filter for the window query.
        hours: rolling window size in hours (default 168 = 7 days).
        min_sample: below this many sent picks in the window, do NOT gate (returns False).

    Returns:
        True if (count of `market` + 1) / (total + 1) > 0.60. Else False.
    """
    rows = pick_repo.get_window_picks(sport=sport, hours=hours)
    if len(rows) < min_sample:
        return False
    same_market = sum(1 for r in rows if r.get("market") == market)
    return (same_market + 1) / (len(rows) + 1) > 0.60
