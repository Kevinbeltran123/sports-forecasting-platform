"""Rolling form features — D-02 point-in-time correctness (DATA-02/05)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl


def _make_hist(
    rows: list[tuple[datetime, str, str, int, int]],
) -> pl.DataFrame:
    """Build a Polars DataFrame of historical matches.

    Columns: kickoff_utc, home_team, away_team, home_goals, away_goals.
    """
    return pl.DataFrame(
        {
            "kickoff_utc": [r[0] for r in rows],
            "home_team": [r[1] for r in rows],
            "away_team": [r[2] for r in rows],
            "home_goals": [r[3] for r in rows],
            "away_goals": [r[4] for r in rows],
        }
    )


class TestRollingForm:
    """Rolling form features use only matches before fixture.kickoff_utc."""

    def test_no_future_match_leakage(self):
        """Rolling form for fixture F uses only matches before F.kickoff_utc."""
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        cutoff = datetime(2024, 3, 1, tzinfo=UTC)
        hist = _make_hist(
            [
                # before cutoff — counted
                (datetime(2024, 2, 1, tzinfo=UTC), "A", "B", 2, 1),
                (datetime(2024, 2, 15, tzinfo=UTC), "A", "C", 3, 0),
                # AFTER cutoff — must be excluded
                (datetime(2024, 3, 5, tzinfo=UTC), "A", "D", 5, 0),
            ]
        )
        feats = eng._rolling_form(
            team="A", historical=hist, cutoff=cutoff, side="home", windows=[3]
        )
        # A scored 2 + 3 = 5 (not 2 + 3 + 5 = 10)
        assert feats["form_home_goals_for_3"] == 5.0

    def test_rolling_windows_3_5_10(self):
        """Rolling windows [3, 5, 10] produce distinct aggregates per metric."""
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        cutoff = datetime(2024, 4, 1, tzinfo=UTC)
        # 10 wins for team A (A home, 2-1) — all before cutoff
        hist = _make_hist(
            [
                (
                    datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
                    "A",
                    "X",
                    2,
                    1,
                )
                for i in range(10)
            ]
        )
        feats = eng._rolling_form(
            team="A",
            historical=hist,
            cutoff=cutoff,
            side="home",
            windows=[3, 5, 10],
        )
        assert feats["form_home_wins_3"] == 3.0
        assert feats["form_home_wins_5"] == 5.0
        assert feats["form_home_wins_10"] == 10.0

    def test_home_away_split(self):
        """side='home' and side='away' produce distinct key prefixes."""
        from bip.sports.football.features import FeatureEngineer

        eng = FeatureEngineer()
        cutoff = datetime(2024, 3, 1, tzinfo=UTC)
        hist = _make_hist(
            [(datetime(2024, 2, 1, tzinfo=UTC), "A", "B", 2, 1)]
        )
        h = eng._rolling_form(
            team="A", historical=hist, cutoff=cutoff, side="home", windows=[3]
        )
        a = eng._rolling_form(
            team="A", historical=hist, cutoff=cutoff, side="away", windows=[3]
        )
        assert "form_home_goals_for_3" in h
        assert "form_away_goals_for_3" in a
        assert "form_home_goals_for_3" not in a
        assert "form_away_goals_for_3" not in h
