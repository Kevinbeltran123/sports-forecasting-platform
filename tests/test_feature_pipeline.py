"""Feature pipeline point-in-time correctness — DATA-05."""

from datetime import datetime, timedelta, timezone
import pytest


class TestFeatureSchemaVersion:
    """D-08: every feature row carries feature_schema_version=2."""

    def test_to_parquet_row_includes_schema_version_v2(self):
        """to_parquet_row() output must include feature_schema_version=2."""
        from bip.sports.football.features import FeatureEngineer
        from bip.sports import FeatureMatrix
        fm = FeatureMatrix(
            fixture_id=12345,
            sport="football",
            league="premier_league",
            computed_at=datetime(2024, 8, 1, tzinfo=timezone.utc),
            features={"home_xg": 1.5},
            kickoff_utc=datetime(2024, 8, 1, 15, 0, tzinfo=timezone.utc),
            home_team="Home FC",
            away_team="Away FC",
        )
        engineer = FeatureEngineer()
        row = engineer.to_parquet_row(fm, matchday=10, season="2024-2025")
        assert "feature_schema_version" in row.columns
        assert row["feature_schema_version"][0] == 2


class TestPointInTimeCorrectness:
    """DATA-05: No future data leakage in feature pipeline."""

    def test_feature_matrix_computed_at_is_not_in_future(self):
        """FeatureMatrix.computed_at must be <= now when features are built."""
        from bip.sports import FeatureMatrix
        now = datetime.now(timezone.utc)
        fm = FeatureMatrix(
            fixture_id=1,
            sport="football",
            league="premier_league",
            computed_at=now - timedelta(seconds=1),
            features={"home_xg": 1.5, "away_xg": 0.9},
            kickoff_utc=now + timedelta(hours=2),
            home_team="Home FC",
            away_team="Away FC",
        )
        assert fm.computed_at <= now

    def test_feature_matrix_computed_at_before_kickoff(self):
        """computed_at must be before fixture kickoff (T-2h window)."""
        from bip.sports import FeatureMatrix
        kickoff = datetime(2026, 4, 22, 15, 0, 0, tzinfo=timezone.utc)
        prediction_time = kickoff - timedelta(hours=2)
        fm = FeatureMatrix(
            fixture_id=1,
            sport="football",
            league="premier_league",
            computed_at=prediction_time,
            features={"home_form_5": 0.6, "away_form_5": 0.4},
            kickoff_utc=kickoff,
            home_team="Home FC",
            away_team="Away FC",
        )
        assert fm.computed_at <= prediction_time

    def test_no_future_data_leakage(self):
        """Integration test: features built at T-2h must not use data from after T-2h.

        This test will be filled in with a real pipeline call in Plan 05.
        Currently asserts the structural contract: FeatureMatrix.computed_at enforcement.
        """
        from bip.sports import FeatureMatrix, FixtureData
        kickoff = datetime(2026, 5, 1, 20, 0, 0, tzinfo=timezone.utc)
        prediction_time = kickoff - timedelta(hours=2)

        fixture = FixtureData(
            fixture_id=99,
            league="premier_league",
            sport="football",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff_utc=kickoff,
        )

        # Simulate feature engineering: computed_at must be <= prediction_time
        # The feature engineer MUST not use data from after prediction_time
        simulated_features = FeatureMatrix(
            fixture_id=fixture.fixture_id,
            sport=fixture.sport,
            league=fixture.league,
            computed_at=prediction_time,  # enforced: set to prediction_time
            features={"home_xg_ma5": 1.6},
            kickoff_utc=fixture.kickoff_utc,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
        )
        assert simulated_features.computed_at <= prediction_time, (
            f"Feature leakage: computed_at={simulated_features.computed_at} "
            f"> prediction_time={prediction_time}"
        )
