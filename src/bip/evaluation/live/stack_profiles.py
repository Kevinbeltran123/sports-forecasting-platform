"""Pre-configured ValueDetector stack profiles.

Day-1 CV (commit d08ebcd) characterized three viable operating points
along the bias-variance frontier. This module exposes them by name so
the operator can pick one explicitly at deployment time without
remembering which kwargs to toggle.

CV-validated Day-1 numbers (5-fold KFold, n_test ≈ 162/fold):

  Profile        | Mean CV ROI | Std    | When to use
  ---------------+-------------+--------+----------------------------------
  tier_1_only    | +32.10%     | ±8.76  | Calibrator unavailable / corrupt
  balanced       | +43.13%     | ±8.21  | Lower variance preferred
  aggressive     | +50.60%     | ±15.07 | Default — best mean ROI

Recommendation: aggressive unless the operator has explicit
variance-aversion reasons. Switch to balanced if realized variance
across 3+ jornadas exceeds expected std materially.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bip.evaluation.live.value_detector import (
    ValueDetector,
    make_best_stack_detector,
)


@dataclass(frozen=True)
class StackProfile:
    """A named ValueDetector configuration with expected performance."""

    name: str
    description: str
    calibrator_path: str | None
    use_per_market: bool
    expected_roi_mean: float  # percent
    expected_roi_std: float   # percent
    expected_roi_band: str
    extra_kwargs: dict[str, Any]

    def build(
        self,
        *,
        calibrator_path_override: str | None = None,
        **kwargs: Any,
    ) -> ValueDetector:
        """Materialize a ValueDetector for this profile.

        ``calibrator_path_override`` lets callers point at a different
        calibrator JSON without changing the profile (useful for testing).
        Other kwargs are forwarded as overrides to ``ValueDetector``.
        """
        cal_path = calibrator_path_override or self.calibrator_path
        merged = {**self.extra_kwargs, **kwargs}
        return make_best_stack_detector(
            calibrator_path=cal_path,
            use_per_market=self.use_per_market,
            **merged,
        )


_DEFAULT_PER_MARKET_PATH = "data/calibration/per_market_v1.json"
_DEFAULT_GLOBAL_PATH = "data/calibration/isotonic_v1.json"


PROFILES: dict[str, StackProfile] = {
    "tier_1_only": StackProfile(
        name="tier_1_only",
        description=(
            "Tier 1 gates only — no calibrator. Highest defensive margin, "
            "lowest expected return. Use when calibrator JSON is missing, "
            "corrupted, or the operator wants minimum variance."
        ),
        calibrator_path=None,
        use_per_market=False,
        expected_roi_mean=32.10,
        expected_roi_std=8.76,
        expected_roi_band="+23% to +41%",
        extra_kwargs={},
    ),
    "balanced": StackProfile(
        name="balanced",
        description=(
            "Tier 1 + global isotonic calibrator. Single global fit "
            "(~22 breakpoints). Balances bias and variance — lower "
            "expected return than aggressive but tighter spread."
        ),
        calibrator_path=_DEFAULT_GLOBAL_PATH,
        use_per_market=False,
        expected_roi_mean=43.13,
        expected_roi_std=8.21,
        expected_roi_band="+35% to +51%",
        extra_kwargs={},
    ),
    "aggressive": StackProfile(
        name="aggressive",
        description=(
            "Tier 1 + per-market calibrator (threshold=25, ~11 markets "
            "with own fit on Day-1). Best mean CV ROI but higher "
            "variance. Recommended default."
        ),
        calibrator_path=_DEFAULT_PER_MARKET_PATH,
        use_per_market=True,
        expected_roi_mean=50.60,
        expected_roi_std=15.07,
        expected_roi_band="+35% to +66%",
        extra_kwargs={},
    ),
    "low_variance": StackProfile(
        name="low_variance",
        description=(
            "Tier 1 + per-market calibrator with conservative threshold=100 "
            "(only large markets get own fit). Sacrifices some mean ROI for "
            "tighter variance. Useful if Day-N realized variance exceeds "
            "expectations under 'aggressive'."
        ),
        calibrator_path=_DEFAULT_PER_MARKET_PATH,
        use_per_market=True,
        expected_roi_mean=39.43,
        expected_roi_std=9.13,
        expected_roi_band="+30% to +49%",
        extra_kwargs={},
    ),
}


def get_profile(name: str) -> StackProfile:
    """Lookup helper with clear error message for typos."""
    if name not in PROFILES:
        names = ", ".join(sorted(PROFILES.keys()))
        raise KeyError(
            f"Unknown stack profile {name!r}. Available: {names}"
        )
    return PROFILES[name]


def list_profiles() -> str:
    """Human-readable profile listing for CLIs / docs."""
    lines = ["Available stack profiles:", ""]
    for p in PROFILES.values():
        lines.extend([
            f"  [{p.name}]",
            f"    {p.description}",
            f"    Expected ROI: {p.expected_roi_band} "
            f"(mean {p.expected_roi_mean:+.1f}%, std {p.expected_roi_std:.1f})",
            "",
        ])
    return "\n".join(lines)
