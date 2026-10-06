"""Mispricing window enforcement — closes principle #3 of the design doc.

Section 1 of the v3 engine design notes (Re-diagnóstico ejecutivo) and
section 2.4 (Mispricing detection sin Pinnacle real-time):

> Cruza contra ``book_slowness(market, recent_event)``: si el último
> evento crítico fue hace <90s y el book aún no movió, el edge es
> *fresh mispricing*. Si pasaron >300s sin movimiento y el edge
> persiste, probablemente tu modelo está mal, no el book.

The MES factor ``book_slowness`` already encodes a market-level lag
prior, but it doesn't enforce *temporal eligibility* — i.e. whether
THIS pick is being emitted in a window where the book lag is real.

This module supplies that enforcement. Given the GSV's
``last_critical_event_age_sec``, it classifies the current frame into
one of five windows and returns:

- a label (HOT, OPTIMAL, WARM, COLD, INDEFINITE)
- a multiplier in [0, 1.5] for the Kelly fraction
- whether the window is COLD (used by rule #10)

No-Bet Gate rule #10 (added here) denies a candidate when:
- the window is COLD (>600s since last critical event)
- AND the edge is below ``cold_edge_threshold`` (default 8%)

i.e. "if 10 minutes have passed since the last critical event and we
still only see modest edge, the book already adjusted — what we're
seeing is model noise, not mispricing."

The classifier is pure: no state, no fitting. All thresholds and
multipliers live on ``MispricingWindowConfig`` so the operator can
tune them in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bip.evaluation.live.engine_v3.gsv import GameStateVector


class WindowLabel(str, Enum):
    """The five mispricing-window classes.

    Order is chronological from event:
        HOT       0-60s   — book still digesting, edge directionally raw
        OPTIMAL   60-180s — sweet spot, book lag is structural
        WARM      180-600s — edge degraded, book partly caught up
        COLD      >600s    — book caught up; persistent edge is model noise
        INDEFINITE          — no critical event observed yet
    """

    HOT = "hot"
    OPTIMAL = "optimal"
    WARM = "warm"
    COLD = "cold"
    INDEFINITE = "indefinite"


@dataclass(frozen=True)
class MispricingWindowConfig:
    """Tunable thresholds in one place. Defaults documented in this
    module's docstring.

    All time bounds are inclusive on the LOWER end (HOT starts at 0,
    OPTIMAL starts at ``hot_end`` etc.). The COLD region is open
    (``> cold_start``).
    """

    hot_end_sec: float = 60.0
    optimal_end_sec: float = 180.0
    warm_end_sec: float = 600.0
    multiplier_hot: float = 1.10
    multiplier_optimal: float = 1.20
    multiplier_warm: float = 0.70
    multiplier_cold: float = 0.30
    multiplier_indefinite: float = 0.85
    cold_edge_threshold: float = 0.08  # rule #10 cutoff


@dataclass(frozen=True)
class WindowResult:
    label: WindowLabel
    age_sec: float | None
    multiplier: float

    @property
    def is_cold(self) -> bool:
        return self.label is WindowLabel.COLD


def classify(
    age_sec: float | None,
    cfg: MispricingWindowConfig | None = None,
) -> WindowResult:
    """Classify a critical-event age into a window + multiplier."""
    cfg = cfg or MispricingWindowConfig()
    if age_sec is None:
        return WindowResult(
            label=WindowLabel.INDEFINITE,
            age_sec=None,
            multiplier=cfg.multiplier_indefinite,
        )
    if age_sec < cfg.hot_end_sec:
        return WindowResult(WindowLabel.HOT, age_sec, cfg.multiplier_hot)
    if age_sec <= cfg.optimal_end_sec:
        return WindowResult(WindowLabel.OPTIMAL, age_sec, cfg.multiplier_optimal)
    if age_sec <= cfg.warm_end_sec:
        return WindowResult(WindowLabel.WARM, age_sec, cfg.multiplier_warm)
    return WindowResult(WindowLabel.COLD, age_sec, cfg.multiplier_cold)


def classify_gsv(
    gsv: GameStateVector,
    cfg: MispricingWindowConfig | None = None,
) -> WindowResult:
    """Convenience wrapper that pulls the age from the GSV."""
    return classify(gsv.last_critical_event_age_sec, cfg)


__all__ = [
    "MispricingWindowConfig",
    "WindowLabel",
    "WindowResult",
    "classify",
    "classify_gsv",
]
