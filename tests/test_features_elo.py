"""ELO rating features — D-02 penaltyblog.ratings.Elo usage with temporal snapshot."""

from __future__ import annotations

from pathlib import Path


class TestEloFeatures:
    """ELO ratings snapshot — D-02 / RESEARCH Pitfall 5."""

    def test_elo_temporal_snapshot(self):
        """_elo_snapshot returns ratings from the Elo state as passed in.

        Caller is responsible for the temporal-scoped Elo state (Pitfall 5).
        After ``Arsenal`` beats ``Chelsea`` (result=0 → home win), Arsenal's
        rating must exceed Chelsea's.
        """
        from penaltyblog.ratings import Elo

        from bip.sports.football.features import FeatureEngineer

        elo = Elo(k=20.0, home_field_advantage=100.0)
        elo.update_ratings("Arsenal", "Chelsea", 0)  # Arsenal win
        eng = FeatureEngineer()
        feats = eng._elo_snapshot(
            elo=elo, home_team="Arsenal", away_team="Chelsea"
        )
        assert feats["elo_home"] > feats["elo_away"]
        assert feats["elo_diff"] == feats["elo_home"] - feats["elo_away"]

    def test_uses_penaltyblog_elo(self):
        """features.py must not hand-roll a class named Elo locally."""
        features_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "bip"
            / "sports"
            / "football"
            / "features.py"
        )
        src = features_path.read_text(encoding="utf-8")
        # No locally-defined Elo class; use penaltyblog.ratings.Elo instead.
        assert "\nclass Elo" not in src

    def test_home_field_advantage_applied(self):
        """penaltyblog.ratings.Elo applies home_field_advantage in probabilities."""
        from penaltyblog.ratings import Elo

        elo = Elo(k=20.0, home_field_advantage=100.0)
        probs = elo.calculate_match_probabilities("A", "B")
        # With identical (default 1500) ratings and HFA=100, home_win > away_win
        assert probs["home_win"] > probs["away_win"]
