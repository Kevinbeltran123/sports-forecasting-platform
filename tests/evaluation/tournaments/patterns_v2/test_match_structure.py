"""Tests for match structure → market family (G2, spike 50)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    FAVORITO_MAX_GAP,
    PICK_EM_MAX_GAP,
    STRUCTURE_TABLE,
    classify_structure,
    match_structure_verdict,
)


class TestClassifyStructure:
    # Boundary parametrize en AMBOS umbrales :
    # justo-debajo / exacto / justo-encima, y simetría de signo (|ΔElo|).
    @pytest.mark.parametrize(
        "gap,expected",
        [
            (0.0, "pick_em"),
            (99.9, "pick_em"),
            (100.0, "favorito"),
            (100.1, "favorito"),
            (199.9, "favorito"),
            (200.0, "mismatch"),
            (200.1, "mismatch"),
            (303.0, "mismatch"),   # México-Sudáfrica real (pick córners ganador)
            (-99.9, "pick_em"),    # el signo no importa: clasifica por |ΔElo|
            (-200.0, "mismatch"),
        ],
    )
    def test_thresholds(self, gap: float, expected: str) -> None:
        assert classify_structure(gap) == expected

    def test_thresholds_match_exported_constants(self) -> None:
        assert classify_structure(PICK_EM_MAX_GAP - 0.1) == "pick_em"
        assert classify_structure(PICK_EM_MAX_GAP) == "favorito"
        assert classify_structure(FAVORITO_MAX_GAP - 0.1) == "favorito"
        assert classify_structure(FAVORITO_MAX_GAP) == "mismatch"


class TestStructureTable:
    def test_three_buckets(self) -> None:
        assert set(STRUCTURE_TABLE) == {"pick_em", "favorito", "mismatch"}

    def test_probabilities_in_range_and_cis_ordered(self) -> None:
        for cell in STRUCTURE_TABLE.values():
            for p in (cell.p_fav_more_corners, cell.p_corner_tie, cell.p_fav_more_shots,
                      cell.p_fav_win, cell.p_over25_group, cell.p_btts_group,
                      cell.p_dog_tt_under15_group, cell.p_dog_tt_zero_group):
                assert 0.0 <= p <= 1.0
            assert cell.ci_low <= cell.p_fav_more_corners <= cell.ci_high

    def test_corpus_size(self) -> None:
        # los 314 partidos del corpus reparten en los 3 buckets (terciles)
        assert sum(c.n for c in STRUCTURE_TABLE.values()) == 314

    def test_monotonicity_invariant(self) -> None:
        """Propiedad validada en el EDA: el gradiente de fuerza es monótono en
        córners y tiros. Si una edición manual de la tabla la rompe, este test
        la atrapa (la tabla solo se regenera con el spike 50)."""
        pe, fa, mm = (STRUCTURE_TABLE[b] for b in ("pick_em", "favorito", "mismatch"))
        assert pe.p_fav_more_corners <= fa.p_fav_more_corners <= mm.p_fav_more_corners
        assert pe.p_fav_more_shots <= fa.p_fav_more_shots <= mm.p_fav_more_shots
        assert pe.corner_diff_mean <= fa.corner_diff_mean <= mm.corner_diff_mean

    def test_pick_em_is_coin_flip_and_mismatch_is_signal(self) -> None:
        """La lección Corea-Chequia/México-SA cuantificada: pick'em sin señal
        (CI cruza 0.50), mismatch con señal (CI bajo > 0.50)."""
        assert STRUCTURE_TABLE["pick_em"].ci_low < 0.50 < STRUCTURE_TABLE["pick_em"].ci_high
        assert STRUCTURE_TABLE["mismatch"].ci_low > 0.50


class TestMatchStructureVerdict:
    def test_mismatch_real_case(self) -> None:
        # México 1961 vs Sudáfrica 1658 (J1 real): mismatch, México favorito
        v = match_structure_verdict("Mexico", "South Africa", 1961.0, 1658.0)
        assert v is not None
        assert v.structure == "mismatch"
        assert v.favorite == "Mexico"
        assert v.underdog == "South Africa"
        assert v.corners_signal == "strong"

    def test_pick_em_real_case(self) -> None:
        # Corea-Chequia (gap chico): pick'em → córners sin señal
        v = match_structure_verdict("South Korea", "Czech Republic", 1810.0, 1777.0)
        assert v is not None
        assert v.structure == "pick_em"
        assert v.corners_signal == "none"

    def test_away_favorite(self) -> None:
        v = match_structure_verdict("Qatar", "Spain", 1500.0, 1900.0)
        assert v is not None
        assert v.favorite == "Spain"
        assert v.underdog == "Qatar"
        assert v.elo_gap == 400.0

    def test_missing_elo_returns_none(self) -> None:
        assert match_structure_verdict("A", "B", None, 1800.0) is None
        assert match_structure_verdict("A", "B", 1800.0, None) is None
        assert match_structure_verdict("A", "B", None, None) is None

    def test_exact_tie_prefers_home_as_favorite(self) -> None:
        v = match_structure_verdict("A", "B", 1700.0, 1700.0)
        assert v is not None
        assert v.favorite == "A"
        assert v.structure == "pick_em"
