"""Regime-shift warning — L3.1 + L4.* finding (internal research notes).

EVIDENCE. Multiple pre-registered Tier-S tests show sign-flips (not nulls)
between discovery (WC18 + Euro20) and hold-out (WC22 + AFCON + Euro24 +
Copa24). Headline: P(fav wins FT | HT 0-0) drops 62% -> 41% from
pre-2022 to post-2022. Plausible drivers: tactical evolution towards low
blocks, AFCON parity contamination, FIFA-window squeeze post-Qatar.

WHY. Lock_v1 trains on martj42 + StatsBomb with implicit stationarity
assumption. The sign-flips indicate that assumption is violated. The
operational risk: lock_v1 will systematically over-favor favorites in
HT 0-0 modern-era matches.

APPLICATION. Emit a warning on the pick metadata when a fixture is in
the modern era AND the live state is HT 0-0. Operator can use the
warning to widen edge thresholds or stand down.

NOT a hard exclude — the regime shift is a calibration concern, not a
deterministic failure. The warning surfaces the risk; operator decides.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


MODERN_ERA_START: date = date(2022, 1, 1)
"""Fixtures on or after this date are considered "modern era" per L3.1
discovery-validation split boundary."""


@dataclass(frozen=True)
class RegimeWarning:
    fixture_date: date
    ht_score: tuple[int, int] | None
    """None if pre-kick; (ht_home, ht_away) once HT is known."""
    is_modern_era: bool
    is_ht_zero_zero: bool
    """True iff HT scoreline is 0-0."""
    warning_active: bool
    rationale: str


def regime_warning_for_fixture(
    fixture_date: date, ht_score: tuple[int, int] | None = None
) -> RegimeWarning:
    """Returns RegimeWarning for a fixture.

    The warning is active when both (a) the fixture is in the modern era
    and (b) the HT scoreline is known to be 0-0. Pre-kick fixtures (ht=None)
    in the modern era get a "potential" warning surfaced via the rationale
    but ``warning_active=False``.

    >>> from datetime import date
    >>> w = regime_warning_for_fixture(date(2026, 6, 12), (0, 0))
    >>> w.warning_active
    True
    >>> w = regime_warning_for_fixture(date(2026, 6, 12), (1, 0))
    >>> w.warning_active
    False
    >>> w = regime_warning_for_fixture(date(2018, 6, 14), (0, 0))
    >>> w.warning_active
    False
    """
    is_modern = fixture_date >= MODERN_ERA_START
    is_zero_zero = ht_score is not None and ht_score == (0, 0)
    active = is_modern and is_zero_zero
    if active:
        rationale = (
            "L3.1 regime shift: P(fav wins FT | HT 0-0) dropped 62% "
            "(WC18+Euro20) -> 41% (WC22+AFCON+Euro24+Copa24). Lock_v1 "
            "trained with stationarity assumption; widen edge threshold "
            "or stand down on live fav-to-win picks."
        )
    elif is_modern and ht_score is None:
        rationale = (
            "Modern-era fixture pre-kick. Warning will activate if HT "
            "scoreline lands at 0-0."
        )
    elif is_zero_zero:
        rationale = (
            "Pre-2022 fixture at HT 0-0; regime warning does not apply "
            "(empirical fav-win rate in this era was 62%)."
        )
    else:
        rationale = "No regime concern for this fixture state."
    return RegimeWarning(
        fixture_date=fixture_date,
        ht_score=ht_score,
        is_modern_era=is_modern,
        is_ht_zero_zero=is_zero_zero,
        warning_active=active,
        rationale=rationale,
    )
