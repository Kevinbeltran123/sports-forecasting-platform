"""Competition weights for between-tournament Bayesian updates.

Held & Vollnhals (2005) showed that European-competition matches deserve
~5× weight vs domestic league results — Borussia Dortmund jumped 18 ranking
places when properly weighted because their UCL 1997 carried more signal
than league play. The same logic applies to national-team tournaments:
World Cup matches reveal more about a team's true strength than a friendly
against a regional rival.

The mapping is intentionally exposed as a module-level constant so the
operator can override it in one place. Default values follow the spike
doc's tiering (Phase 3 open question 1): WC = 4×, Confederation = 3×,
qualifier = 2×, FIFA-date friendly = 1×, non-FIFA = 0.5×, with finals
bumped one tier up because the stakes magnify the signal.

Reference:
- internal research notes §"Ponderación de partidos"
- internal research notes §"Las dos escalas"
- internal planning notes Phase 3 open Q1
"""

from __future__ import annotations

from enum import StrEnum


class CompetitionType(StrEnum):
    """Categorical competition type. Stable string values for JSON round-trip."""

    WORLD_CUP_FINAL = "wc_final"
    WORLD_CUP = "wc"  # group stage + KO except final
    CONFEDERATION_FINAL = "conf_final"  # Euros / Copa América / AFCON / Asian Cup final
    CONFEDERATION = "conf"  # group stage + KO except final
    QUALIFIER = "qualifier"  # WC qualifiers, Euros qualifiers
    FRIENDLY_FIFA = "friendly_fifa"  # FIFA-date friendly
    FRIENDLY_NON_FIFA = "friendly_non_fifa"  # off-window friendly


# Default weight ratios — operator can override via `set_default_weights()` or
# pass `competition_weight=X` directly on TournamentMatchResult.
DEFAULT_WEIGHTS: dict[CompetitionType, float] = {
    CompetitionType.WORLD_CUP_FINAL: 5.0,
    CompetitionType.WORLD_CUP: 4.0,
    CompetitionType.CONFEDERATION_FINAL: 4.0,
    CompetitionType.CONFEDERATION: 3.0,
    CompetitionType.QUALIFIER: 2.0,
    CompetitionType.FRIENDLY_FIFA: 1.0,
    CompetitionType.FRIENDLY_NON_FIFA: 0.5,
}


def weight_for(competition: CompetitionType | str | None) -> float:
    """Look up the weight for a competition type.

    Accepts the enum, its string value, or None (treated as friendly_fifa
    so unannotated matches contribute one unit of evidence).
    """
    if competition is None:
        return DEFAULT_WEIGHTS[CompetitionType.FRIENDLY_FIFA]
    if isinstance(competition, str):
        try:
            competition = CompetitionType(competition)
        except ValueError as exc:
            raise ValueError(
                f"Unknown competition type '{competition}'. "
                f"Valid: {[c.value for c in CompetitionType]}"
            ) from exc
    return DEFAULT_WEIGHTS[competition]
