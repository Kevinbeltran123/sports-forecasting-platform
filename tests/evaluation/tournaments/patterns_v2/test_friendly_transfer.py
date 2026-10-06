"""Tests for friendly→tournament transfer findings (G4, spike 51)."""
from __future__ import annotations

from bip.evaluation.tournaments.patterns_v2 import (
    CARDS_WC_SHIFT_CI,
    CARDS_WC_SHIFT_PER_TEAM,
    FRIENDLY_TRANSFER,
)


class TestFriendlyTransfer:
    def test_all_betting_metrics_present(self) -> None:
        assert set(FRIENDLY_TRANSFER) == {
            "shots", "sot", "possession", "goals_for", "goals_against",
            "corners", "cards", "fouls",
        }

    def test_ranking_labels_valid(self) -> None:
        for f in FRIENDLY_TRANSFER.values():
            assert f.ranking in ("transfers", "weak", "none")
            assert f.note

    def test_key_findings_frozen(self) -> None:
        """Las conclusiones operativas que el brief cita — si una edición las cambia
        sin re-correr el spike 51, este test la atrapa."""
        # córners amistosos NO predicen torneo (la coherencia con G2/estructura)
        assert FRIENDLY_TRANSFER["corners"].ranking == "none"
        assert FRIENDLY_TRANSFER["corners"].r_wc22 == -0.05
        # defensa NO transfiere (réplica del hallazgo original r=+0.04)
        assert FRIENDLY_TRANSFER["goals_against"].ranking == "none"
        # tiros/posesión SÍ transfieren (ambos corpus ≥ +0.35)
        for m in ("shots", "possession"):
            f = FRIENDLY_TRANSFER[m]
            assert f.ranking == "transfers"
            assert f.r_modern >= 0.35

    def test_transfers_label_backed_by_evidence(self) -> None:
        """'transfers' exige r ≥ 0.35 en al menos un corpus (regla del spike)."""
        for f in FRIENDLY_TRANSFER.values():
            if f.ranking == "transfers":
                rs = [r for r in (f.r_wc22, f.r_modern) if r is not None]
                assert max(rs) >= 0.35

    def test_cards_regime_shift(self) -> None:
        lo, hi = CARDS_WC_SHIFT_CI
        assert lo <= CARDS_WC_SHIFT_PER_TEAM <= hi
        assert lo > 0  # el shift es real (CI no cruza 0): amistoso→WC SUBEN las TA
