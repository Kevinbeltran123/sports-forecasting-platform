"""Head-to-head features — D-02 temporal correctness."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def _hist(rows: list[tuple[datetime, str, str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "kickoff_utc": [r[0] for r in rows],
            "home_team": [r[1] for r in rows],
            "away_team": [r[2] for r in rows],
            "home_goals": [r[3] for r in rows],
            "away_goals": [r[4] for r in rows],
        }
    )


class TestH2HFeatures:
    """H2H aggregates use only meetings before fixture.kickoff_utc."""

    def test_h2h_temporal(self):
        """H2H aggregation uses only meetings before cutoff."""
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        cutoff = datetime(2024, 3, 1, tzinfo=UTC)
        hist = _hist(
            [
                # before cutoff — counted
                (datetime(2024, 1, 1, tzinfo=UTC), "A", "B", 2, 1),
                # after cutoff — excluded
                (datetime(2024, 4, 1, tzinfo=UTC), "A", "B", 3, 0),
            ]
        )
        feats = eng._h2h_features(
            home="A", away="B", historical=hist, cutoff=cutoff
        )
        assert feats["h2h_total_meetings"] == 1.0

    def test_last_meeting_recency(self):
        """h2h_days_since_last computed from most recent prior meeting.

        Non-leap reference window: Feb 20, 2023 → Mar 1, 2023 = 9 days exactly
        (2023 is not a leap year, so Feb has 28 days).
        """
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        cutoff = datetime(2023, 3, 1, tzinfo=UTC)
        hist = _hist(
            [(datetime(2023, 2, 20, tzinfo=UTC), "A", "B", 1, 1)]
        )
        feats = eng._h2h_features(
            home="A", away="B", historical=hist, cutoff=cutoff
        )
        assert feats["h2h_days_since_last"] == 9.0
