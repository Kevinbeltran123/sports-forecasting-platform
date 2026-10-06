"""Alert state tracking and rate limiting.

OPERATOR DECISION (locked 2026-05-24):
  - Max 1 alert per (fixture, market) — first time gate passes
  - Re-alert if edge grows by >=5pp from the previously-emitted edge

The history is in-memory per process. For production durability we'd
persist to disk/DB but MVP keeps it in-memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field


EDGE_INCREASE_THRESHOLD_PP = 0.05
"""5 percentage point increase triggers a re-alert."""


@dataclass
class AlertEntry:
    """One emitted alert in history."""

    fixture_id: int
    market: str
    edge_pct_at_emit: float
    """Edge at the time the alert was sent."""

    z_score_at_emit: float


@dataclass
class AlertHistory:
    """Per-process registry of emitted alerts. Mutable on purpose."""

    by_key: dict[tuple[int, str], AlertEntry] = field(default_factory=dict)

    def record(
        self, fixture_id: int, market: str, edge_pct: float, z_score: float
    ) -> None:
        self.by_key[(fixture_id, market)] = AlertEntry(
            fixture_id=fixture_id,
            market=market,
            edge_pct_at_emit=edge_pct,
            z_score_at_emit=z_score,
        )

    def previous(self, fixture_id: int, market: str) -> AlertEntry | None:
        return self.by_key.get((fixture_id, market))


def should_alert(
    history: AlertHistory,
    fixture_id: int,
    market: str,
    current_edge_pct: float,
) -> tuple[bool, str]:
    """Decide whether to emit an alert based on history + rate-limit rule.

    Returns (should_alert, reason). Caller emits only when should_alert
    is True, then records via ``history.record(...)``.
    """
    prev = history.previous(fixture_id, market)
    if prev is None:
        return (True, "first alert for this (fixture, market)")
    delta = current_edge_pct - prev.edge_pct_at_emit
    if delta >= EDGE_INCREASE_THRESHOLD_PP:
        return (
            True,
            f"edge grew {delta:+.2%} (>= {EDGE_INCREASE_THRESHOLD_PP:.0%} threshold) "
            f"since last alert (was {prev.edge_pct_at_emit:.2%})",
        )
    return (
        False,
        f"already alerted with edge {prev.edge_pct_at_emit:.2%}; "
        f"current {current_edge_pct:.2%}, delta {delta:+.2%} < threshold",
    )
