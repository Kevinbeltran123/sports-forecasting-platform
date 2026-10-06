"""League strength multipliers for cross-league rate normalization.

Player per-90 rates are NOT directly comparable across leagues. A 0.5 SoT/90
in EPL implies different attacking quality than 0.5 SoT/90 in MLS — the
defenses being faced are different. The multiplier scales raw rates to an
EPL-equivalent baseline so the lineup blender (Phase 2.2) can sum rates
across teammates from heterogeneous leagues.

Reference league: Premier League (multiplier = 1.0, glicko_rating = 2118.8).

Phase 4 (2026-05-15) anchors the multipliers to **Shelopugin (2023) Glicko-2
ratings** (Table V — European leagues, Table VI — South American leagues)
per SYNTHESIS Conclusion 3. The transfer formula is multiplicative on the
log-rate scale:

    multiplier(L) = exp(α · (rating(L) - rating(reference)))

α (``DEFAULT_ALPHA_PER_GLICKO`` = 0.001) was reverse-engineered from the
operator's pre-Phase-4 multiplier table: median of {α implied per league}
across 7 leagues with Shelopugin ratings = 0.001/Glicko-rating-point. This
preserves the operator's accumulated intuition and roots it in the
empirical literature; the value is unblockable Layer-2 calibration via
real transfer outcomes (post-WC2026).

Multipliers live in ``configs/league_strength.yaml``. Each entry now carries
both ``glicko_rating`` (Shelopugin Table V/VI canonical, optional for leagues
outside his sample) and a ``multiplier`` field. When ``glicko_rating`` is
populated, the multiplier should match ``derive_multiplier_from_glicko``
within ``MULTIPLIER_DERIVATION_TOLERANCE``; deliberate overrides are flagged
as warnings rather than errors (operator may have empirical reason to
diverge from the formula).

Lookup is cache-first; missing leagues default to FALLBACK_MULTIPLIER with
a warning logged.

References:
- internal research notes
- internal research notes Conclusion 3 (three-step transfer chain)
- internal planning notes Phase 4
"""

from __future__ import annotations

import math
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field

from bip.core.errors import ConfigurationError

logger = structlog.get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "league_strength.yaml"

# When a player's club_league_id is not in the multiplier table we fall back
# to this value (a moderately-strong-league assumption) and log a warning.
# Set deliberately conservative so unknown leagues don't inflate predictions.
FALLBACK_MULTIPLIER = 0.70

# Shelopugin (2023) Table V — Premier League is the strongest in his sample.
# Used as the multiplier=1.0 anchor.
REFERENCE_GLICKO_RATING = 2118.8

# Multiplicative attenuation per Glicko rating point. Empirical median of
# ``α_implied`` across the 7 operator-tuned leagues that overlap Shelopugin
# Table V/VI (computed in Phase 4 design notes — see module docstring).
# A 250-point gap (Premier vs Brazil) yields multiplier ≈ exp(-0.25) ≈ 0.78,
# matching the operator's pre-Phase-4 Brasileirão multiplier.
DEFAULT_ALPHA_PER_GLICKO = 0.001

# When ``glicko_rating`` is populated and the YAML's ``multiplier`` deviates
# from ``derive_multiplier_from_glicko`` by more than this fraction (relative
# error), emit a warning rather than failing — operator may have empirical
# reason to override (e.g., league experienced a calibre shift).
MULTIPLIER_DERIVATION_TOLERANCE = 0.05  # 5% relative


def derive_multiplier_from_glicko(
    rating: float,
    *,
    reference_rating: float = REFERENCE_GLICKO_RATING,
    alpha: float = DEFAULT_ALPHA_PER_GLICKO,
) -> float:
    """Convert a Shelopugin Glicko-2 rating into a multiplicative scaling factor.

    ``multiplier = exp(α · (rating - reference_rating))``

    Properties:
    - ``rating == reference_rating`` → multiplier exactly 1.0
    - ``rating > reference_rating`` → multiplier > 1.0 (player faced tougher
      opposition than the reference league; rates inflated when transferred
      DOWN to the reference)
    - ``rating < reference_rating`` → multiplier < 1.0 (player's rates need
      attenuation when transferred UP to the reference)
    - ``alpha == 0`` → multiplier always 1.0 regardless of rating gap (sanity
      escape hatch for tests / disabling the layer entirely)
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return math.exp(alpha * (rating - reference_rating))


class LeagueStrength(BaseModel):
    """Multiplier entry for a single league.

    Phase 4 adds the optional ``glicko_rating`` field for Shelopugin Table V/VI
    grounding. Leagues outside his sample (MLS, Saudi, J1, etc.) leave it
    null and rely on the operator-tuned ``multiplier`` alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_football_league_id: int
    name: str
    multiplier: float = Field(ge=0.1, le=1.5)
    confidence: str = Field(pattern=r"^(high|medium|low)$")
    # Shelopugin (2023) Table V (Europe) / Table VI (South America) Glicko-2
    # rating. Optional — null for leagues outside Shelopugin's sample.
    glicko_rating: float | None = Field(default=None, ge=1000.0, le=2500.0)


class LeagueStrengthTable:
    """In-memory lookup of league_id -> multiplier."""

    def __init__(self, entries: list[LeagueStrength]) -> None:
        self._by_id: dict[int, LeagueStrength] = {e.api_football_league_id: e for e in entries}

    def get(self, league_id: int | None) -> float:
        """Return the multiplier for a league, or FALLBACK_MULTIPLIER if missing."""
        if league_id is None:
            logger.warning("league_strength_missing", league_id=None, fallback=FALLBACK_MULTIPLIER)
            return FALLBACK_MULTIPLIER
        entry = self._by_id.get(league_id)
        if entry is None:
            logger.warning(
                "league_strength_missing",
                league_id=league_id,
                fallback=FALLBACK_MULTIPLIER,
            )
            return FALLBACK_MULTIPLIER
        return entry.multiplier

    def apply(self, rate: float, league_id: int | None) -> float:
        """Scale a per-90 rate by the league's multiplier.

        ``apply(rate, league_id) == rate * get(league_id)``. Convenience
        wrapper for callers that want a one-shot adjustment without the
        intermediate multiplier lookup. Used by ``ClubFormLoader`` outputs
        and the ``CompositionBlender`` lineup pre-aggregation step.
        """
        return rate * self.get(league_id)

    def get_entry(self, league_id: int | None) -> LeagueStrength | None:
        """Return the full LeagueStrength entry (multiplier + rating + name).

        Returns None when the league is missing from the table (no warning
        logged — caller decides whether to fall back).
        """
        if league_id is None:
            return None
        return self._by_id.get(league_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, league_id: int) -> bool:
        return league_id in self._by_id


def load_league_strength(path: Path | None = None) -> LeagueStrengthTable:
    """Load and validate the league_strength.yaml config.

    Phase 4: when an entry has ``glicko_rating`` populated, the YAML's
    ``multiplier`` is cross-checked against ``derive_multiplier_from_glicko``.
    Deviations beyond ``MULTIPLIER_DERIVATION_TOLERANCE`` emit a warning
    (deliberate overrides are allowed; the warning surfaces them for review).
    """
    target = path if path is not None else _CONFIG_PATH
    if not target.exists():
        raise ConfigurationError(
            f"League strength config not found: {target}. "
            "Run scripts/seed_league_strength.py or populate manually."
        )
    with target.open() as f:
        data = yaml.safe_load(f)

    entries = [LeagueStrength.model_validate(item) for item in data["leagues"]]
    if not any(e.multiplier == 1.0 for e in entries):
        logger.warning(
            "league_strength_no_reference",
            note=(
                "No league has multiplier=1.0; "
                "ratios are still valid but absolute scale is arbitrary."
            ),
        )

    # Phase 4: cross-check Shelopugin-grounded entries against the formula.
    for entry in entries:
        if entry.glicko_rating is None:
            continue
        derived = derive_multiplier_from_glicko(entry.glicko_rating)
        if derived <= 0:
            continue
        rel_err = abs(entry.multiplier - derived) / derived
        if rel_err > MULTIPLIER_DERIVATION_TOLERANCE:
            logger.warning(
                "league_strength_derivation_mismatch",
                league_id=entry.api_football_league_id,
                name=entry.name,
                glicko_rating=entry.glicko_rating,
                yaml_multiplier=entry.multiplier,
                derived_multiplier=round(derived, 4),
                relative_error=round(rel_err, 4),
                note=(
                    "YAML multiplier diverges from exp(α·ΔR); "
                    "confirm operator override is intentional."
                ),
            )

    return LeagueStrengthTable(entries)
