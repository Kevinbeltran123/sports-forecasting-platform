"""D-04 + Pitfall 4 — results/ and odds/ Parquet stores with 3-level partitioning.

Pitfall 1 regression guards: fixture_id dtype must be pl.Int64 everywhere.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def store(tmp_path: Path):
    from bip.core.storage.parquet_store import ParquetStore
    return ParquetStore(base_path=tmp_path)


@pytest.fixture
def sample_results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fixture_id": [1001, 1002, 1003],
            "sport": ["football"] * 3,
            "league": ["premier_league", "premier_league", "la_liga"],
            "season": ["2024-2025"] * 3,
            "home_goals": [2, 1, 0],
            "away_goals": [1, 1, 3],
            "status": ["finished", "finished", "finished"],
        },
        schema_overrides={"fixture_id": pl.Int64},
    )


@pytest.fixture
def sample_odds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fixture_id": [1001, 1002, 1003],
            "sport": ["football"] * 3,
            "league": ["premier_league", "premier_league", "la_liga"],
            "season": ["2024-2025"] * 3,
            "bookmaker": ["Betano"] * 3,
            "opening_home": [2.10, 2.50, 1.80],
            "opening_draw": [3.40, 3.30, 3.60],
            "opening_away": [3.20, 2.70, 4.20],
            "closing_home": [2.05, 2.45, 1.82],
            "closing_draw": [3.45, 3.35, 3.55],
            "closing_away": [3.25, 2.75, 4.15],
            "pinnacle_close_home": [None, None, None],
            "pinnacle_close_draw": [None, None, None],
            "pinnacle_close_away": [None, None, None],
        },
        schema_overrides={
            "fixture_id": pl.Int64,
            "pinnacle_close_home": pl.Float64,
            "pinnacle_close_draw": pl.Float64,
            "pinnacle_close_away": pl.Float64,
        },
    )


class TestResultsStore:
    def test_round_trip_results(self, store, sample_results):
        store.write_results(sample_results)
        result = store.read_results(sport="football", league="premier_league")
        assert len(result) == 2
        assert "home_goals" in result.columns
        assert "away_goals" in result.columns
        assert "status" in result.columns

    def test_results_is_3_level_not_4_level(self, store, sample_results):
        """Pitfall 4: no matchday=... directory under results/."""
        store.write_results(sample_results)
        results_dir = store.base_path / "results"
        assert results_dir.exists()
        assert not any("matchday=" in str(p) for p in results_dir.rglob("*"))

    def test_fixture_id_is_int64_in_results(self, store, sample_results):
        """Pitfall 1 regression guard."""
        store.write_results(sample_results)
        result = store.read_results(sport="football", league="premier_league")
        assert result.schema["fixture_id"] == pl.Int64


class TestOddsStore:
    def test_round_trip_odds(self, store, sample_odds):
        store.write_odds(sample_odds)
        result = store.read_odds(sport="football", league="premier_league")
        assert len(result) == 2
        assert "opening_home" in result.columns
        assert "closing_home" in result.columns
        # nullable pinnacle_* columns survive serialization
        assert "pinnacle_close_home" in result.columns
        assert result["pinnacle_close_home"].null_count() == 2

    def test_odds_is_3_level_not_4_level(self, store, sample_odds):
        store.write_odds(sample_odds)
        odds_dir = store.base_path / "odds"
        assert odds_dir.exists()
        assert not any("matchday=" in str(p) for p in odds_dir.rglob("*"))

    def test_fixture_id_is_int64_in_odds(self, store, sample_odds):
        store.write_odds(sample_odds)
        result = store.read_odds(sport="football", league="premier_league")
        assert result.schema["fixture_id"] == pl.Int64


class TestFeaturesUnchanged:
    """Regression guard: write_features still 4-level (Phase 1 compat)."""

    def test_features_still_4_level(self, store):
        df = pl.DataFrame(
            {
                "fixture_id": [1001, 1002],
                "sport": ["football"] * 2,
                "league": ["premier_league"] * 2,
                "season": ["2024-2025"] * 2,
                "matchday": [1, 2],
                "computed_at": ["2024-08-01T00:00:00+00:00"] * 2,
                "feat_a": [0.5, 0.6],
            },
            schema_overrides={"fixture_id": pl.Int64, "matchday": pl.Int64},
        )
        store.write_features(df)
        features_dir = store.base_path / "features"
        assert features_dir.exists()
        # At least one directory must be matchday=... (4-level preserved)
        matchday_dirs = list(features_dir.rglob("matchday=*"))
        assert len(matchday_dirs) >= 1


class TestPerFixtureAppend:
    """CR-01 regression: per-fixture writes with overwrite=False must append.

    The default ``overwrite=True`` semantic is correct for batch writes (one
    DataFrame containing the whole partition's rows in a single call) but
    catastrophic for per-fixture writes — every call would shutil.rmtree the
    partition before writing the new single-row DataFrame, silently wiping
    every prior fixture in the partition.
    """

    @staticmethod
    def _result_row(fixture_id: int, home: int, away: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "fixture_id": [fixture_id],
                "sport": ["football"],
                "league": ["premier_league"],
                "season": ["2024-2025"],
                "home_goals": [home],
                "away_goals": [away],
                "status": ["finished"],
            },
            schema_overrides={"fixture_id": pl.Int64},
        )

    @staticmethod
    def _odds_row(
        fixture_id: int, home: float, draw: float, away: float,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "fixture_id": [fixture_id],
                "sport": ["football"],
                "league": ["premier_league"],
                "season": ["2024-2025"],
                "bookmaker": ["Betano"],
                "opening_home": [home],
                "opening_draw": [draw],
                "opening_away": [away],
                "closing_home": [home],
                "closing_draw": [draw],
                "closing_away": [away],
                "pinnacle_close_home": [None],
                "pinnacle_close_draw": [None],
                "pinnacle_close_away": [None],
            },
            schema_overrides={
                "fixture_id": pl.Int64,
                "pinnacle_close_home": pl.Float64,
                "pinnacle_close_draw": pl.Float64,
                "pinnacle_close_away": pl.Float64,
            },
        )

    def test_per_fixture_results_appends_not_overwrites(self, store):
        """Two single-row writes preserve both rows when overwrite=False."""
        store.write_results(self._result_row(2001, 2, 1), overwrite=False)
        store.write_results(self._result_row(2002, 0, 0), overwrite=False)
        out = store.read_results(sport="football", league="premier_league")
        ids = sorted(out["fixture_id"].to_list())
        assert ids == [2001, 2002], (
            f"Expected both fixtures retained; got {ids}. "
            "Per-fixture write with overwrite=False must not wipe the partition."
        )

    def test_per_fixture_odds_appends_not_overwrites(self, store):
        store.write_odds(self._odds_row(3001, 2.0, 3.4, 3.2), overwrite=False)
        store.write_odds(self._odds_row(3002, 1.8, 3.6, 4.2), overwrite=False)
        out = store.read_odds(sport="football", league="premier_league")
        ids = sorted(out["fixture_id"].to_list())
        assert ids == [3001, 3002]

    def test_default_overwrite_still_replaces_partition(self, store):
        """Regression guard: default overwrite=True keeps batch-write semantics.

        Existing tests rely on a single ``write_results(big_df)`` call replacing
        the partition wholesale.
        """
        first_batch = pl.concat(
            [self._result_row(4001, 1, 0), self._result_row(4002, 2, 2)],
            how="diagonal",
        )
        second_batch = self._result_row(4003, 0, 1)  # different fixture
        store.write_results(first_batch)              # default overwrite=True
        store.write_results(second_batch)             # default overwrite=True
        out = store.read_results(sport="football", league="premier_league")
        # Default overwrite=True wipes the partition before each write, so only
        # the most recent batch's row(s) survive.
        assert sorted(out["fixture_id"].to_list()) == [4003]

    def test_per_fixture_results_dedups_repeat_fixture_id(self, store):
        """Re-writing the same fixture_id keeps only the latest row."""
        store.write_results(self._result_row(5001, 0, 0), overwrite=False)
        # Status updated from initial 'finished' (placeholder) to corrected goals
        store.write_results(self._result_row(5001, 3, 1), overwrite=False)
        out = store.read_results(sport="football", league="premier_league")
        assert len(out) == 1
        assert out["fixture_id"].to_list() == [5001]
        # Latest write wins (keep="last")
        assert out["home_goals"].to_list() == [3]
        assert out["away_goals"].to_list() == [1]
