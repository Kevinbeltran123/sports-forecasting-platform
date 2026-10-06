"""ParquetStore tests — DATA-02 (4-level Hive partitioning)."""

from pathlib import Path
import pytest
import polars as pl


@pytest.fixture
def store(tmp_path: Path):
    """ParquetStore backed by a temporary directory."""
    from bip.core.storage.parquet_store import ParquetStore
    return ParquetStore(base_path=tmp_path)


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Sample feature DataFrame with all 4 partition columns."""
    return pl.DataFrame({
        "fixture_id": [1001, 1002, 1003],
        "sport": ["football", "football", "football"],
        "league": ["premier_league", "premier_league", "la_liga"],
        "season": ["2024-2025", "2024-2025", "2024-2025"],
        "matchday": [20, 20, 19],
        "home_xg": [1.8, 0.9, 2.1],
        "away_xg": [1.2, 1.5, 0.8],
    })


class TestWriteFeatures:
    """DATA-02: write_features creates correct 4-level Hive directory structure."""

    def test_creates_hive_partitioned_directory(self, store, sample_features):
        """Write creates sport=/league=/season=/matchday= Hive structure."""
        store.write_features(sample_features)
        features_dir = store.base_path / "features"
        parquet_files = list(features_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0

        partition_dirs = {str(p.parent.relative_to(features_dir)) for p in parquet_files}
        assert any("sport=football" in d for d in partition_dirs)
        assert any("league=premier_league" in d for d in partition_dirs)
        assert any("matchday=20" in d for d in partition_dirs)

    def test_round_trip_read_write(self, store, sample_features):
        """Features written can be read back with all columns intact."""
        store.write_features(sample_features)
        result = store.read_features(sport="football", league="premier_league")
        assert len(result) == 2  # 2 premier_league rows
        assert "fixture_id" in result.columns

    def test_overwrite_semantics(self, store, sample_features):
        """Writing same partition twice does not duplicate rows."""
        store.write_features(sample_features)
        store.write_features(sample_features)  # write again
        result = store.read_features(sport="football")
        assert len(result) == 3  # still 3 rows, not 6

    def test_empty_dataframe_is_noop(self, store):
        """Writing empty DataFrame must not raise and must not create files."""
        store.write_features(pl.DataFrame())
        features_dir = store.base_path / "features"
        assert not features_dir.exists()

    def test_missing_partition_columns_raises_storage_error(self, store):
        """Missing partition column raises StorageError (not generic exception)."""
        from bip.core.errors import StorageError
        bad_df = pl.DataFrame({"fixture_id": [1], "home_xg": [1.5]})
        with pytest.raises(StorageError):
            store.write_features(bad_df)


class TestReadFeatures:
    """DATA-02: read_features with partition filters returns correct rows."""

    def test_read_with_sport_filter(self, store, sample_features):
        """Sport filter returns only matching rows."""
        store.write_features(sample_features)
        result = store.read_features(sport="football")
        assert len(result) == 3

    def test_read_empty_store_returns_empty_dataframe(self, store):
        """Reading from empty store returns empty DataFrame (not error)."""
        result = store.read_features()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
