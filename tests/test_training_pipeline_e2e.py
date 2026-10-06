"""D-17: Synthetic integration test for TrainingPipeline.run() end-to-end.

Uses real Parquet I/O in tmp_path. Asserts CLV + logloss WIRING (fields
populated as float-or-None), NOT specific CLV values — synthetic data
cannot guarantee any particular CLV magnitude.

Constructor reality (post-02.1-10): TrainingPipeline(settings: Settings).
n_splits is hardcoded to 5 inside pipeline.run() via WalkForwardSplitter.
The pipeline reads parquet_base_path + model_dir from Settings, so the
test points both at tmp_path via env vars.

Plan-deviation note (Rule 1 — bug fix): the plan-as-written reused a
single computed_at literal for every row, which trips
WalkForwardSplitter's strict-< temporal-leakage assertion (max == min).
This test instead spaces computed_at by hours so each row has a unique
timestamp while preserving total ordering.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def store(tmp_path: Path):
    from bip.core.storage.parquet_store import ParquetStore
    return ParquetStore(base_path=tmp_path)


@pytest.fixture
def synthetic_e2e_data():
    """200-row synthetic dataset: features + results + odds.

    ~10% of opening_home is None to exercise the D-03 null_clv_rows path.
    computed_at varies by row (hour spacing) so WalkForwardSplitter's
    strict-< temporal-leakage assertion holds.
    """
    rng = np.random.default_rng(42)
    n = 200
    fixture_ids = list(range(1001, 1001 + n))
    base_dt = datetime(2024, 8, 1, tzinfo=timezone.utc)
    computed_at = [
        (base_dt + timedelta(hours=i)).isoformat() for i in range(n)
    ]

    df_features = pl.DataFrame(
        {
            "fixture_id": fixture_ids,
            "sport": ["football"] * n,
            "league": ["test_league"] * n,
            "season": ["2024-2025"] * n,
            "matchday": [i % 38 + 1 for i in range(n)],
            "computed_at": computed_at,
            "feature_schema_version": [2] * n,
            "feat_home_xg": rng.uniform(0.5, 2.5, n).tolist(),
            "feat_away_xg": rng.uniform(0.3, 2.0, n).tolist(),
            "feat_h2h_wins": rng.integers(0, 10, n).tolist(),
            "feat_form_home": rng.uniform(-1.0, 1.0, n).tolist(),
            "feat_form_away": rng.uniform(-1.0, 1.0, n).tolist(),
        },
        schema_overrides={"fixture_id": pl.Int64, "matchday": pl.Int64},
    )

    df_results = pl.DataFrame(
        {
            "fixture_id": fixture_ids,
            "sport": ["football"] * n,
            "league": ["test_league"] * n,
            "season": ["2024-2025"] * n,
            "home_goals": rng.integers(0, 4, n).tolist(),
            "away_goals": rng.integers(0, 4, n).tolist(),
            "status": ["finished"] * n,
        },
        schema_overrides={"fixture_id": pl.Int64},
    )

    opening_home = rng.uniform(1.5, 4.0, n).tolist()
    opening_draw = rng.uniform(2.5, 4.5, n).tolist()
    opening_away = rng.uniform(1.8, 5.0, n).tolist()
    # Inject ~10% None to trigger null_clv_rows increments (D-03)
    for i in rng.choice(n, size=n // 10, replace=False).tolist():
        opening_home[i] = None

    df_odds = pl.DataFrame(
        {
            "fixture_id": fixture_ids,
            "sport": ["football"] * n,
            "league": ["test_league"] * n,
            "season": ["2024-2025"] * n,
            "bookmaker": ["Betano"] * n,
            "opening_home": opening_home,
            "opening_draw": opening_draw,
            "opening_away": opening_away,
            "closing_home": opening_home,
            "closing_draw": opening_draw,
            "closing_away": opening_away,
            "pinnacle_close_home": [None] * n,
            "pinnacle_close_draw": [None] * n,
            "pinnacle_close_away": [None] * n,
        },
        schema_overrides={
            "fixture_id": pl.Int64,
            "opening_home": pl.Float64,
            "opening_draw": pl.Float64,
            "opening_away": pl.Float64,
            "closing_home": pl.Float64,
            "closing_draw": pl.Float64,
            "closing_away": pl.Float64,
            "pinnacle_close_home": pl.Float64,
            "pinnacle_close_draw": pl.Float64,
            "pinnacle_close_away": pl.Float64,
        },
    )

    return df_features, df_results, df_odds


class TestTrainingPipelineE2E:
    """D-17: synthetic gate — TrainingPipeline.run() wiring assertions."""

    def test_pipeline_produces_metadata_with_all_fields_populated(
        self, store, synthetic_e2e_data, monkeypatch, tmp_path
    ):
        """All CLV + logloss metadata fields must be populated (float or None).

        Wiring test: verifies the pipeline CALLS simulate_pick, compute_clv,
        and log_loss. Does NOT assert specific CLV values (synthetic data).
        """
        df_features, df_results, df_odds = synthetic_e2e_data

        # Write all three stores into tmp_path
        store.write_features(df_features)
        store.write_results(df_results)
        store.write_odds(df_odds)

        # Settings env vars: pipeline reads parquet_base_path + model_dir from
        # Settings(); point both at tmp_path so the pipeline picks up the
        # synthetic stores and writes artifacts inside the sandbox.
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-supabase-key")
        monkeypatch.setenv("API_FOOTBALL_KEY", "test-api-football-key")
        monkeypatch.setenv("ODDS_API_KEY", "test-odds-api-key")
        monkeypatch.setenv("PARQUET_BASE_PATH", str(tmp_path))
        monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))

        from bip.core.settings import Settings
        from bip.train.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(settings=Settings())
        meta = pipeline.run(league="test_league", version="v1")

        # --- CLV wiring assertions (D-09, D-11, D-16) ---
        # synthetic data may not hit ≥20 picks per fold → clv_pct may be None
        # but the FIELD must exist (no AttributeError / KeyError)
        assert meta.walk_forward_mean_clv_pct is None or isinstance(
            meta.walk_forward_mean_clv_pct, float
        )

        # --- Logloss wiring assertions (D-13, D-14) ---
        assert meta.logloss_uncalibrated is not None, (
            "logloss_uncalibrated must be set after pipeline.run()"
        )
        assert meta.logloss_calibrated is not None, (
            "logloss_calibrated must be set after pipeline.run()"
        )
        assert meta.logloss_improvement_pct is not None, (
            "logloss_improvement_pct must be set after pipeline.run()"
        )
        assert isinstance(meta.logloss_uncalibrated, float)
        assert isinstance(meta.logloss_calibrated, float)
        assert isinstance(meta.logloss_improvement_pct, float)

        # --- null_clv_rows wiring (D-03) ---
        assert meta.null_clv_rows > 0, (
            "~10% of opening_home is None — null_clv_rows must be > 0"
        )

        # --- Fold structure assertions (D-14) ---
        # Pipeline hardcodes WalkForwardSplitter(n_splits=5)
        assert len(meta.walk_forward_fold_details) == 5
        for i, fold in enumerate(meta.walk_forward_fold_details):
            assert "clv_pct" in fold, f"fold missing 'clv_pct' key: {fold}"
            assert "logloss_uncalibrated" in fold, (
                f"fold missing 'logloss_uncalibrated': {fold}"
            )
            assert "logloss_calibrated" in fold, (
                f"fold missing 'logloss_calibrated': {fold}"
            )
            # clv_pct may be None (not enough picks) or a float
            assert fold["clv_pct"] is None or isinstance(fold["clv_pct"], float)
            # logloss_uncalibrated must always be a float (computed every fold)
            assert isinstance(fold["logloss_uncalibrated"], float)
            # logloss_calibrated: only the last fold has it; earlier folds None
            if i == len(meta.walk_forward_fold_details) - 1:
                assert isinstance(fold["logloss_calibrated"], float)
            else:
                assert fold["logloss_calibrated"] is None
