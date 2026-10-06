"""Live match evaluation: state aggregation + predictor + value detection.

Spike: live-edge-with-sportmonks (2026-05-09 → 2026-05-23).

Pipeline:

    Sportmonks Fixture (snapshot)
            │
            ▼
    LiveMatchState  ←  aggregates score / minute / stats / pressure / events
            │
            ▼
    LiveMatchPredictor
            │   - Sportmonks ML predictions (baseline)
            │   - Dixon-Robinson scaling for remaining-time markets
            │   - Optional ensemble with our xG-blended Bayesian
            ▼
    Per-market probability set
            │
            ▼
    ValueDetector  ←  compares predictions vs live odds, computes EV / Kelly
            │
            ▼
    LivePick (filtered, ranked) → Telegram / Markdown report
"""

from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.live.predictor import LiveMatchPredictor, MarketProbabilities
from bip.evaluation.live.value_detector import ValueDetector, LivePick

__all__ = [
    "LiveMatchState",
    "LiveMatchPredictor",
    "MarketProbabilities",
    "ValueDetector",
    "LivePick",
]
