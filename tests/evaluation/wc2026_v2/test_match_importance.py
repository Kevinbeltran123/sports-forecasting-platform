"""Ola 1 — match-importance + time-decay weighting tests.

Boundary cases :
- importance: each tournament tier hits its documented K-weight
- decay: weight at age=0, half_life, and 2×half_life maps to {1.0, 0.5, 0.25}
- combined: importance × decay composes multiplicatively
- unknown tournament defaults to K_OTHER_TOURNAMENTS
"""

from __future__ import annotations

from datetime import date

import pytest

from bip.evaluation.tournaments.wc2026_v2.match_importance import (
    DEFAULT_HALF_LIFE_DAYS,
    K_CONTINENTAL_FINALS,
    K_FRIENDLY,
    K_OTHER_TOURNAMENTS,
    K_QUALIFIERS_AND_NATIONS_LEAGUE,
    K_WC_FINALS,
    compute_match_weight,
    time_decay_weight,
    tournament_importance,
)


class TestTournamentImportance:
    @pytest.mark.parametrize(
        ("tournament", "expected_k"),
        [
            ("FIFA World Cup", K_WC_FINALS),
            ("UEFA Euro", K_CONTINENTAL_FINALS),
            ("Copa América", K_CONTINENTAL_FINALS),
            ("African Cup of Nations", K_CONTINENTAL_FINALS),
            ("AFC Asian Cup", K_CONTINENTAL_FINALS),
            ("Confederations Cup", K_CONTINENTAL_FINALS),
            ("FIFA World Cup qualification", K_QUALIFIERS_AND_NATIONS_LEAGUE),
            ("UEFA Euro qualification", K_QUALIFIERS_AND_NATIONS_LEAGUE),
            ("UEFA Nations League", K_QUALIFIERS_AND_NATIONS_LEAGUE),
            ("Friendly", K_FRIENDLY),
        ],
    )
    def test_known_tournaments(self, tournament: str, expected_k: float) -> None:
        assert tournament_importance(tournament) == expected_k

    def test_unknown_tournament_defaults_to_other(self) -> None:
        assert tournament_importance("Made-up regional cup") == K_OTHER_TOURNAMENTS

    def test_empty_string_defaults_to_other(self) -> None:
        assert tournament_importance("") == K_OTHER_TOURNAMENTS

    def test_relative_ordering(self) -> None:
        """WC > continental > qualifiers > friendly — sanity check."""
        assert K_WC_FINALS > K_CONTINENTAL_FINALS
        assert K_CONTINENTAL_FINALS > K_QUALIFIERS_AND_NATIONS_LEAGUE
        assert K_QUALIFIERS_AND_NATIONS_LEAGUE > K_OTHER_TOURNAMENTS
        assert K_OTHER_TOURNAMENTS > K_FRIENDLY


class TestTimeDecay:
    REFERENCE = date(2026, 5, 23)

    def test_zero_age_returns_one(self) -> None:
        w = time_decay_weight(self.REFERENCE, self.REFERENCE)
        assert w == pytest.approx(1.0, abs=1e-9)

    def test_half_life_returns_half(self) -> None:
        match_date = self.REFERENCE.replace(year=self.REFERENCE.year - 3)
        # 3 years ≈ 1095/1096 days depending on leaps. The default half-life
        # is 1096 (= 3 years × 365.33). The weight should be very close to 0.5
        # but not exactly because of leap rounding.
        w = time_decay_weight(match_date, self.REFERENCE)
        assert w == pytest.approx(0.5, abs=0.005)

    def test_two_half_lives_returns_quarter(self) -> None:
        # 2 × DEFAULT_HALF_LIFE_DAYS days ago.
        import datetime

        match_date = self.REFERENCE - datetime.timedelta(days=int(2 * DEFAULT_HALF_LIFE_DAYS))
        w = time_decay_weight(match_date, self.REFERENCE)
        assert w == pytest.approx(0.25, abs=1e-3)

    def test_future_match_clamped_to_one(self) -> None:
        future = date(2026, 12, 1)
        w = time_decay_weight(future, self.REFERENCE)
        assert w == pytest.approx(1.0, abs=1e-9)

    def test_monotonically_decreasing(self) -> None:
        recent = date(2025, 5, 23)
        older = date(2020, 5, 23)
        assert time_decay_weight(recent, self.REFERENCE) > time_decay_weight(older, self.REFERENCE)

    @pytest.mark.parametrize("custom_half_life", [365.0, 730.0, 1500.0])
    def test_custom_half_life_obeys_half_at_t_half(self, custom_half_life: float) -> None:
        import datetime

        match_date = self.REFERENCE - datetime.timedelta(days=int(custom_half_life))
        w = time_decay_weight(match_date, self.REFERENCE, half_life_days=custom_half_life)
        assert w == pytest.approx(0.5, abs=1e-3)


class TestComputeMatchWeight:
    REFERENCE = date(2026, 5, 23)

    def test_composition_is_multiplicative(self) -> None:
        # Same-day WC final: weight = K_WC × 1.0 = 60.
        w = compute_match_weight(self.REFERENCE, "FIFA World Cup", self.REFERENCE)
        assert w == pytest.approx(K_WC_FINALS, abs=1e-9)

    def test_old_friendly_smaller_than_recent_wc_qualifier(self) -> None:
        old_friendly = compute_match_weight(date(2018, 1, 1), "Friendly", self.REFERENCE)
        recent_wc_qual = compute_match_weight(
            date(2025, 11, 1), "FIFA World Cup qualification", self.REFERENCE
        )
        assert recent_wc_qual > old_friendly

    def test_unknown_tournament_uses_default_k(self) -> None:
        w = compute_match_weight(self.REFERENCE, "Some Regional Cup", self.REFERENCE)
        assert w == pytest.approx(K_OTHER_TOURNAMENTS, abs=1e-9)

    def test_zero_weight_for_extremely_old_match(self) -> None:
        # 1872 match (oldest in martj42). At ~150 years, weight is essentially 0.
        w = compute_match_weight(date(1872, 11, 30), "Friendly", self.REFERENCE)
        assert w < 1e-10
