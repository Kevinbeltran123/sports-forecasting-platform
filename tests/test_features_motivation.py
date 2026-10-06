"""Motivation context features — D-02 table-state temporal correctness."""

from __future__ import annotations

from datetime import UTC, datetime


class TestMotivationFeatures:
    """Motivation uses league-table state snapshot provided by caller."""

    def test_motivation_temporal(self):
        """top-4 flag set for top-6 teams, clear for mid-table."""
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        feats = eng._motivation_features(
            home="TopTeam",
            away="MidTeam",
            raw_standings={
                "standings": [
                    {
                        "team": "TopTeam",
                        "rank": 2,
                        "points": 60,
                        "played": 30,
                        "total_games": 38,
                    },
                    {
                        "team": "MidTeam",
                        "rank": 10,
                        "points": 40,
                        "played": 30,
                        "total_games": 38,
                    },
                ]
            },
            cutoff=datetime(2024, 4, 1, tzinfo=UTC),
        )
        assert feats["motivation_top4_home"] == 1.0
        assert feats["motivation_top4_away"] == 0.0

    def test_dead_rubber_classification(self):
        """motivation_dead_rubber=1 when neither team in race AND <=5 games left."""
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        feats = eng._motivation_features(
            home="A",
            away="B",
            raw_standings={
                "standings": [
                    {
                        "team": "A",
                        "rank": 10,
                        "points": 40,
                        "played": 35,
                        "total_games": 38,
                    },
                    {
                        "team": "B",
                        "rank": 11,
                        "points": 39,
                        "played": 35,
                        "total_games": 38,
                    },
                ]
            },
            cutoff=datetime(2024, 5, 1, tzinfo=UTC),
        )
        assert feats["motivation_dead_rubber"] == 1.0
