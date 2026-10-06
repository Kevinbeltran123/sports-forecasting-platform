"""Phase 4 metrics public API (D-15).

Sport-agnostic per CORE-02. Functions take `sport` as a parameter; never branch on it.
"""
from bip.core.metrics.aggregator import (
    DriftResult,
    MetricsAggregator,
    compute_daily_metrics,
    compute_weekly_drift,
)

__all__ = [
    "DriftResult",
    "MetricsAggregator",
    "compute_daily_metrics",
    "compute_weekly_drift",
]
