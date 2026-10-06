"""Tests for team-level transfer filter (Iter 2 finding)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.patterns_v2 import (
    TEAM_TRANSFER_TILT,
    team_transfer_verdict,
)


class TestTeamTransferTilt:
    def test_senegal_and_morocco_are_over_performers(self) -> None:
        assert TEAM_TRANSFER_TILT["Senegal"]["tilt"] > 1.0
        assert TEAM_TRANSFER_TILT["Morocco"]["tilt"] > 1.0

    def test_tunisia_and_ghana_are_under_performers(self) -> None:
        assert TEAM_TRANSFER_TILT["Tunisia"]["tilt"] < 1.0
        assert TEAM_TRANSFER_TILT["Ghana"]["tilt"] < 1.0

    def test_evidence_present_for_every_entry(self) -> None:
        for team, entry in TEAM_TRANSFER_TILT.items():
            assert isinstance(entry["evidence"], str)
            assert len(entry["evidence"]) > 20, team
            # Evidence should cite tournament(s).
            assert any(
                token in entry["evidence"]
                for token in ("AFCON23", "WC22", "WC18")
            ), team


class TestTeamTransferVerdict:
    @pytest.mark.parametrize("team", ["Senegal", "Morocco", "Nigeria"])
    def test_known_over_performers(self, team: str) -> None:
        v = team_transfer_verdict(team)
        assert v.has_transfer_signal is True
        assert v.tilt > 1.0

    @pytest.mark.parametrize("team", ["Tunisia", "Ghana", "Egypt"])
    def test_known_under_performers(self, team: str) -> None:
        v = team_transfer_verdict(team)
        assert v.has_transfer_signal is True
        assert v.tilt < 1.0

    def test_unknown_team_neutral(self) -> None:
        v = team_transfer_verdict("Argentina")
        assert v.has_transfer_signal is False
        assert v.tilt == 1.0

    def test_evidence_returned(self) -> None:
        v = team_transfer_verdict("Senegal")
        assert "AFCON23" in v.evidence
        assert "z=+1.98" in v.evidence
