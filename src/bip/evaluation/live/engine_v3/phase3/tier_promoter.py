"""Tier D promoter — the v3 emit boundary (sec 10, Phase 3).

> Promoción a Tier D (mínimo stake) del v3; sistema actual mantiene
> Tier A/B/C.

A ``MarketCandidate`` that passed the gate is not yet a publishable
pick. Phase 3 routes it through:

1. ``StakeRotationPolicy.evaluate`` — concentration cap + new-market ramp.
2. ``confidence_modulator.modulate`` — multiplies Kelly fraction by
   corroborating evidence.
3. ``CalibrationDriftDetector.is_suspended`` — refuses cells under
   suspension until weekly recalibration.
4. ``family_in_shadow_mode`` — props always shadow until 100+ resolved.

The output is a ``PromotedPick`` carrying:

- ``stage``: ``"shadow"`` (audit only), ``"tier_d"`` (the v3 emit),
  or ``"rejected"`` (with reason).
- ``stake_units``: the operator-actionable stake (between 0 and
  ``base_unit`` × ``confidence_multiplier`` × ``ramp_multiplier``).
- ``audit_trail``: full breakdown of every gate + modulator decision.

The Telegram alert sender is NOT called here — the promoter only
prepares the pick. Wiring to Telegram remains separate so the v3
shadow flow stays *structurally incapable* of leaking to the operator
before the 4-week Tier-D validation window completes (sec 10 Phase 3
criterion).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.market_selector import MarketCandidate
from bip.evaluation.live.engine_v3.phase3.confidence_modulator import (
    ConfidenceBreakdown,
    modulate,
)
from bip.evaluation.live.engine_v3.phase3.drift_detector import (
    CalibrationDriftDetector,
)
from bip.evaluation.live.engine_v3.phase3.regimes import bucket_gsv
from bip.evaluation.live.engine_v3.phase3.stake_policy import (
    StakeDecision,
    StakeRotationPolicy,
)
from bip.evaluation.live.engine_v3.thesis import MarketFamily, ThesisArchetype


Stage = Literal["shadow", "tier_d", "rejected"]


# Families that remain in shadow mode regardless of MES quality, per
# sec 5.5 ("Shadow mode obligatorio para mercados nuevos"). Props
# require ≥100 resolved picks before promotion — until that happens
# the promoter routes them to ``stage="shadow"``.
_SHADOW_ONLY_FAMILIES: frozenset[MarketFamily] = frozenset(
    {MarketFamily.PROPS}
)


@dataclass(frozen=True)
class PromotedPick:
    fixture_id: int
    timestamp_utc: datetime
    candidate: MarketCandidate
    stage: Stage
    stake_units: float
    confidence: ConfidenceBreakdown
    stake_decision: StakeDecision
    rejection_reason: str = ""
    audit: dict = field(default_factory=dict)


@dataclass
class TierDPromoter:
    """Phase-3 promoter — synthesises operator-actionable picks.

    The base unit is the bankroll fraction at which an unscaled Tier-D
    pick would be staked. Confidence + ramp multipliers reduce it; the
    concentration cap can hard-reject.
    """

    stake_policy: StakeRotationPolicy
    drift_detector: CalibrationDriftDetector
    base_unit: float = 1.0  # operator's Tier-D bankroll fraction

    def promote(
        self, candidate: MarketCandidate, gsv: GameStateVector,
    ) -> PromotedPick:
        ts = gsv.timestamp_utc
        family = candidate.family

        # Confidence modulator
        conf = modulate(gsv, candidate.thesis)

        # Drift suspension
        regime = bucket_gsv(gsv).as_str()
        suspended = self.drift_detector.is_suspended(regime, family)

        # Stake policy: derive the would-be stake assuming all multipliers, then
        # ask the policy whether that level breaks the cap.
        prospective_stake = (
            self.base_unit * conf.multiplier
        )
        decision: StakeDecision = self.stake_policy.evaluate(
            family, prospective_stake, now=ts,
        )

        if family in _SHADOW_ONLY_FAMILIES:
            return PromotedPick(
                fixture_id=gsv.fixture_id, timestamp_utc=ts,
                candidate=candidate, stage="shadow",
                stake_units=0.0,
                confidence=conf, stake_decision=decision,
                rejection_reason="family in shadow-mode (props)",
                audit={"regime": regime, "drift_suspended": suspended},
            )

        if suspended:
            return PromotedPick(
                fixture_id=gsv.fixture_id, timestamp_utc=ts,
                candidate=candidate, stage="rejected",
                stake_units=0.0,
                confidence=conf, stake_decision=decision,
                rejection_reason=f"drift-suspended cell {regime}/{family.value}",
                audit={"regime": regime},
            )

        if conf.multiplier == 0.0:
            return PromotedPick(
                fixture_id=gsv.fixture_id, timestamp_utc=ts,
                candidate=candidate, stage="rejected",
                stake_units=0.0,
                confidence=conf, stake_decision=decision,
                rejection_reason="confidence multiplier soft-vetoed (<0.30)",
                audit={"regime": regime},
            )

        if not decision.allowed:
            return PromotedPick(
                fixture_id=gsv.fixture_id, timestamp_utc=ts,
                candidate=candidate, stage="rejected",
                stake_units=0.0,
                confidence=conf, stake_decision=decision,
                rejection_reason=f"stake policy denied: {decision.reason}",
                audit={"regime": regime},
            )

        stake_units = self.base_unit * conf.multiplier * decision.multiplier
        # Record into the policy state so subsequent picks see this volume.
        self.stake_policy.record_stake(family, stake_units, timestamp=ts)

        return PromotedPick(
            fixture_id=gsv.fixture_id, timestamp_utc=ts,
            candidate=candidate, stage="tier_d",
            stake_units=stake_units,
            confidence=conf, stake_decision=decision,
            audit={"regime": regime},
        )


# ──────────────────────────────────────────────────────────────────────
# Promotion outcome recorder — feeds the drift detector
# ──────────────────────────────────────────────────────────────────────


def record_outcome(
    detector: CalibrationDriftDetector,
    pick: PromotedPick,
    outcome: float,
) -> None:
    """Push a resolved pick into the drift detector.

    ``outcome`` is 1.0 when the bet won, 0.0 when it lost. Half-wins /
    push outcomes (asian half lines) should be passed as 0.5 — KS
    handles non-binary outcomes fine.

    The caller has the resolved bet result; we extract the regime + family
    from the audit trail and the predicted probability from the candidate.
    """
    regime = pick.audit.get("regime")
    if regime is None:
        return
    detector.record(
        regime=regime,
        family=pick.candidate.family,
        predicted_p=pick.candidate.fair_prob,
        outcome=outcome,
    )


__all__ = ["PromotedPick", "Stage", "TierDPromoter", "record_outcome"]
