"""Match-importance + time-decay weighting for the weighted-MLE strength prior.

Implements the per-match weight from Ley, Van de Wiele & Van Eetvelde (2019)
"Ranking soccer teams on the basis of their current strength: A comparison of
maximum likelihood approaches" (DOI 10.1177/1471082X18817650, §2.2):

    weight(m) = importance(tournament_m) × exp(-ln(2) · age_days(m) / half_life)

where ``importance`` is the eloratings.net K-weight table (the canonical
practitioner consensus surveyed in
``internal research notes §2.4``) and ``half_life`` defaults
to 36 months — the Ley 2019-recommended value for national-team data, also
consistent with the slow-decay consensus on r/algobetting cited in the
practitioner survey.

Tournament-name normalisation is martj42-specific: the ``tournament`` column
in ``data/cache/martj42_international_results.csv`` uses long English names
(e.g. ``"FIFA World Cup qualification"``).
"""

from __future__ import annotations

from datetime import date
from math import exp, log

# ─────────────────────────────────────────────────────────────────
# Match-importance K-weights — eloratings.net canonical values
# (https://www.eloratings.net/about, §"K-factor").
# ─────────────────────────────────────────────────────────────────

K_WC_FINALS: float = 60.0
K_CONTINENTAL_FINALS: float = 50.0
K_QUALIFIERS_AND_NATIONS_LEAGUE: float = 40.0
K_OTHER_TOURNAMENTS: float = 30.0
K_FRIENDLY: float = 20.0

# Canonical martj42 tournament-name → K-weight table. Names not in the table
# fall through to ``K_OTHER_TOURNAMENTS`` (regional / minor tournaments) by
# default; ``"Friendly"`` is the only string mapped to K=20.
_TOURNAMENT_K: dict[str, float] = {
    # WC finals
    "FIFA World Cup": K_WC_FINALS,
    # Continental finals
    "UEFA Euro": K_CONTINENTAL_FINALS,
    "Copa América": K_CONTINENTAL_FINALS,
    "African Cup of Nations": K_CONTINENTAL_FINALS,
    "AFC Asian Cup": K_CONTINENTAL_FINALS,
    "Gold Cup": K_CONTINENTAL_FINALS,
    "CONCACAF Championship": K_CONTINENTAL_FINALS,
    "Oceania Nations Cup": K_CONTINENTAL_FINALS,
    "Confederations Cup": K_CONTINENTAL_FINALS,
    # Qualifiers + Nations League
    "FIFA World Cup qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "UEFA Euro qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "African Cup of Nations qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "AFC Asian Cup qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "Gold Cup qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "Copa América qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "Oceania Nations Cup qualification": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "UEFA Nations League": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    "CONCACAF Nations League": K_QUALIFIERS_AND_NATIONS_LEAGUE,
    # Friendlies
    "Friendly": K_FRIENDLY,
}

DEFAULT_HALF_LIFE_DAYS: float = 1096.0  # 36 months ≈ 3 years (Ley 2019)


def tournament_importance(tournament_name: str) -> float:
    """K-weight for a martj42 tournament name.

    Unknown names fall through to ``K_OTHER_TOURNAMENTS`` (e.g. minor
    regional cups, defunct competitions). This is consistent with
    eloratings.net's "other tournaments" tier.
    """

    return _TOURNAMENT_K.get(tournament_name, K_OTHER_TOURNAMENTS)


def time_decay_weight(
    match_date: date,
    reference_date: date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Exponential time-decay: w = 0.5 ** (age_days / half_life).

    Returns 1.0 at ``match_date == reference_date`` and 0.5 at one
    half-life in the past. Future matches (age < 0) are clamped to age=0
    so that scheduled-but-unplayed fixtures (defensive) cannot up-weight
    the fit.
    """

    age_days = max(0, (reference_date - match_date).days)
    return exp(-log(2.0) * age_days / half_life_days)


def compute_match_weight(
    match_date: date,
    tournament_name: str,
    reference_date: date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Combined importance × time-decay weight for one martj42 match row.

    Drop-in helper that the WeightedMLEFitter applies row-wise to build
    its ``sample_weight`` vector.
    """

    return tournament_importance(tournament_name) * time_decay_weight(
        match_date, reference_date, half_life_days
    )
