"""Token-bucket bandwidth governor for Telegram Bot v2 (§E).

Replaces the v1 ``min_interval_seconds`` floor with a token-bucket model
that:

- Allows immediate sends when capacity is available (no quiet-period
  penalty)
- Batches into a digest when capacity is exhausted (no flooding under
  burst)
- Stays well under Telegram's 20 msg/min per-channel cap by design

State is in-memory only — the bucket regenerates from scratch on
watcher restart, which is fine since the burst queue itself
(``tg_burst_queue``) is durable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    """Classical token bucket.

    Capacity is the maximum tokens that accumulate during quiet periods.
    Refill is continuous (fractional tokens accrue between sends).

    Defaults (capacity=5, refill_seconds=6.0) yield ~10 sends/min long-term
    with bursts up to 5 in quick succession — well under Telegram's
    20 msg/min channel cap.
    """

    capacity: int = 5
    refill_seconds: float = 6.0
    _tokens: float = 0.0       # initialized in __post_init__
    _last_refill: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.refill_seconds <= 0:
            raise ValueError("refill_seconds must be positive")
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    @property
    def tokens(self) -> float:
        """Current token count (refilled on read for fairness)."""
        self._refill()
        return self._tokens

    def _refill(self, *, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        gained = elapsed / self.refill_seconds
        self._tokens = min(float(self.capacity), self._tokens + gained)
        self._last_refill = now

    def try_consume(self, cost: float = 1.0) -> bool:
        """Consume tokens if available. Returns True on success."""
        if cost <= 0:
            return True
        self._refill()
        if self._tokens + 1e-9 >= cost:
            self._tokens -= cost
            return True
        return False

    def seconds_until_token(self, target: float = 1.0) -> float:
        """Time until ``target`` tokens are available (0 if already)."""
        self._refill()
        if self._tokens >= target:
            return 0.0
        deficit = target - self._tokens
        return deficit * self.refill_seconds

    def reset_for_test(self, *, tokens: float | None = None) -> None:
        """Test-only — seed bucket state deterministically."""
        if tokens is not None:
            self._tokens = float(tokens)
        self._last_refill = time.monotonic()
