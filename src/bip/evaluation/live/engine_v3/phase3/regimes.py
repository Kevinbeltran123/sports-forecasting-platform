"""Regime bucketing — section 5.1 of the v3 engine design notes.

> No calibres por GSV exacto. Cuantiza a 25-30 regímenes arquetípicos
> (los 12 de sec 4-bis + variaciones por minuto-bucket + variaciones
> por liga). Cada régimen tiene n suficiente.

The function ``bucket_gsv(gsv) → RegimeKey`` is the canonical map.
The output is a string of the form::

    "arch-<archetype>:min-<bucket>:phase-<game_phase>"

Examples:
- "arch-dominant_losing_napoli:min-30-45:phase-open_attacking"
- "arch-none:min-60-75:phase-cagey_closed"
- "arch-cruise_mode:min-80+:phase-cruise"

The number of unique keys is bounded:
- 12 archetypes + "none" = 13
- 5 minute buckets (0-15, 15-30, 30-45, 45-60, 60-75, 75-90 = 6)
- 5 game phases
- Practical count after pruning impossible combos ≈ 25-30.

Bucketing is **stateful-free**: pure function of the GSV. The archetype
component reuses the rule-layer detectors — if a thesis fires, the
regime carries its archetype; otherwise "none".
"""
from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.live.engine_v3.archetypes import generate_theses
from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import ThesisArchetype


_MINUTE_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 15, "00-15"),
    (15, 30, "15-30"),
    (30, 45, "30-45"),
    (45, 60, "45-60"),
    (60, 75, "60-75"),
    (75, 1000, "75+"),
)


def minute_bucket(minute: int) -> str:
    for lo, hi, label in _MINUTE_BUCKETS:
        if lo <= minute < hi:
            return label
    return "75+"


@dataclass(frozen=True)
class RegimeKey:
    """Composite key for a regime bucket.

    Equality + hash are derived → safe as dict key for the calibration
    lookup tables.
    """

    archetype: str  # archetype value or "none"
    minute_bucket: str  # one of the labels above
    game_phase: str

    def as_str(self) -> str:
        return (
            f"arch-{self.archetype}:min-{self.minute_bucket}:phase-{self.game_phase}"
        )


def bucket_gsv(gsv: GameStateVector) -> RegimeKey:
    """Map a GSV to its regime bucket.

    Archetype component:
    - If exactly one archetype fires → its enum value.
    - If multiple fire → the highest-priority one (priority = the
      archetype with the most specific premise; we use enum declaration
      order as proxy).
    - If none fire → "none".

    Phase + minute are read directly from the GSV. The minute bucket is
    inclusive on the low end, exclusive on the high end.
    """
    theses = generate_theses(gsv)
    if not theses:
        archetype = "none"
    else:
        # Priority = declaration order in ThesisArchetype enum.
        order = list(ThesisArchetype)
        active = sorted(
            theses, key=lambda t: order.index(t.archetype),
        )
        archetype = active[0].archetype.value
    return RegimeKey(
        archetype=archetype,
        minute_bucket=minute_bucket(gsv.time.minute),
        game_phase=gsv.tactical.game_phase,
    )


def all_minute_buckets() -> tuple[str, ...]:
    return tuple(label for _, _, label in _MINUTE_BUCKETS)


__all__ = [
    "RegimeKey",
    "all_minute_buckets",
    "bucket_gsv",
    "minute_bucket",
]
