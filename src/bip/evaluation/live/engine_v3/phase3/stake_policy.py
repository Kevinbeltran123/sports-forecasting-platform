"""Stake rotation policy — exposure control per market family.

> Stake rotation policy explícita: max N% del volumen semanal por
> market_type; ramp gradual cuando se introduce mercado nuevo.

Two distinct concerns:

1. **Concentration cap**: no single ``market_family`` may exceed
   ``max_share_pct`` (default 30%) of the rolling weekly volume.
   Excess triggers a throttle for the rest of the week.

2. **New-market ramp**: a market freshly graduated from shadow mode
   ramps over 2-3 weeks before reaching full stake — week 1 at 25%,
   week 2 at 50%, week 3 at 100%. The ramp limits exposure to a market
   whose live calibration is still unproven.

The policy is **advisory** — it returns ``StakeDecision`` with a
multiplier and a reason. The caller (TierDPromoter) is responsible
for actually applying the multiplier and skipping the pick when the
multiplier is 0.

State is intentionally in-memory only here. A persistent variant
(SQLite + cron-roll on Monday 00:00 UTC) is a small extension.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from bip.evaluation.live.engine_v3.thesis import MarketFamily


@dataclass(frozen=True)
class StakeDecision:
    allowed: bool
    multiplier: float  # in [0.0, 1.0] — caller multiplies its stake by this
    reason: str


@dataclass
class StakeRotationPolicy:
    """Tracks weekly per-family volume + ramps new markets."""

    max_share_pct: float = 0.30
    new_market_ramp_weeks: int = 3
    rolling_window_days: int = 7
    # Concentration cap only fires once cumulative weekly volume crosses
    # this floor. Without it, the very first stake on any family is
    # always 100% share and the cap would deny everything.
    min_total_for_cap: float = 5.0

    # state
    _volume_log: list[tuple[datetime, MarketFamily, float]] = field(default_factory=list)
    _market_introduced_at: dict[MarketFamily, datetime] = field(default_factory=dict)

    # ── tracking ────────────────────────────────────────────────────────

    def record_stake(
        self,
        family: MarketFamily,
        stake: float,
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        self._volume_log.append((ts, family, float(stake)))
        if family not in self._market_introduced_at:
            self._market_introduced_at[family] = ts
        self._prune(ts)

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self.rolling_window_days)
        self._volume_log = [(t, f, s) for (t, f, s) in self._volume_log if t >= cutoff]

    # ── queries ─────────────────────────────────────────────────────────

    def rolling_volume(self, now: datetime | None = None) -> dict[MarketFamily, float]:
        now = now or datetime.now(timezone.utc)
        self._prune(now)
        agg: dict[MarketFamily, float] = defaultdict(float)
        for (_, f, s) in self._volume_log:
            agg[f] += s
        return dict(agg)

    def rolling_total(self, now: datetime | None = None) -> float:
        return sum(self.rolling_volume(now).values())

    def share_of(self, family: MarketFamily, now: datetime | None = None) -> float:
        total = self.rolling_total(now)
        if total <= 0:
            return 0.0
        return self.rolling_volume(now).get(family, 0.0) / total

    def is_new_market(
        self, family: MarketFamily, now: datetime | None = None,
    ) -> int:
        """Weeks since first stake on this family. 0 = never seen."""
        introduced = self._market_introduced_at.get(family)
        if introduced is None:
            return 0
        now = now or datetime.now(timezone.utc)
        days = (now - introduced).days
        return max(1, (days // 7) + 1)

    # ── enforcement ─────────────────────────────────────────────────────

    def evaluate(
        self,
        family: MarketFamily,
        stake: float,
        *,
        now: datetime | None = None,
    ) -> StakeDecision:
        """Compute the multiplier + allowed flag for a candidate stake."""
        now = now or datetime.now(timezone.utc)
        self._prune(now)

        # Ramp for new markets.
        ramp_multiplier = 1.0
        if family not in self._market_introduced_at:
            # Brand new — first stake ever. Treat as week-1 ramp.
            week_index = 1
        else:
            week_index = self.is_new_market(family, now)
        if week_index < self.new_market_ramp_weeks:
            ramp_multiplier = (week_index) / float(self.new_market_ramp_weeks)

        # Concentration cap — only enforced once cumulative volume has
        # reached ``min_total_for_cap``. Pre-floor, the share calculation
        # is meaningless (every stake looks like 100%).
        total_after = self.rolling_total(now) + stake
        if total_after >= self.min_total_for_cap:
            share_after = (
                self.rolling_volume(now).get(family, 0.0) + stake
            ) / max(1e-6, total_after)
            if share_after > self.max_share_pct:
                return StakeDecision(
                    allowed=False,
                    multiplier=0.0,
                    reason=(
                        f"{family.value} share would be "
                        f"{share_after:.1%} > {self.max_share_pct:.0%}"
                    ),
                )

        if ramp_multiplier < 1.0:
            return StakeDecision(
                allowed=True,
                multiplier=ramp_multiplier,
                reason=(
                    f"new-market ramp week {week_index}/{self.new_market_ramp_weeks}"
                    f" — staking at {ramp_multiplier:.0%}"
                ),
            )
        return StakeDecision(
            allowed=True, multiplier=1.0,
            reason=f"within concentration cap, no ramp",
        )


__all__ = ["StakeDecision", "StakeRotationPolicy"]
