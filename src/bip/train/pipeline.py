"""Training pipeline orchestrator — per-league ensemble training + calibration + save."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
import structlog
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss

from bip.core.settings import Settings
from bip.core.storage.parquet_store import ParquetStore
from bip.train.backtest import (
    EDGE_THRESHOLD_PCT,
    SLIPPAGE_PCT,
    apply_slippage,
    compute_clv,
    simulate_pick,
)
from bip.train.base_models import CB_PARAMS, LGBM_PARAMS, XGB_PARAMS
from bip.train.calibration import calibrate, select_calibrator
from bip.train.metadata import ModelMetadata, feature_set_hash
from bip.train.stacking import StackedEnsemble
from bip.train.walkforward import WalkForwardSplitter

logger = structlog.get_logger(__name__)


# 1X2 label mapping (matches penaltyblog): 0=home win, 1=draw, 2=away win
def _label_from_goals(hg: int, ag: int) -> int:
    if hg > ag:
        return 0
    if hg == ag:
        return 1
    return 2


class _EnsembleProbaWrapper(ClassifierMixin, BaseEstimator):
    """sklearn 1.8-compatible wrapper around a fitted StackedEnsemble.

    Required for CalibratedClassifierCV(estimator=FrozenEstimator(...)) under
    sklearn 1.8: the underlying estimator must implement fit/predict and
    expose __sklearn_tags__ (provided by BaseEstimator). Defined at module
    scope so joblib.dump can serialize the calibrated classifier (local
    nested classes are not addressable for joblib's serializer).
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        ens: StackedEnsemble,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        dates_fit: np.ndarray,
    ) -> None:
        ens.fit_fold(X_fit, y_fit, X_fit[:1], dates_fit)
        self._ens = ens
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X_in: np.ndarray) -> np.ndarray:
        from bip.train.stacking import _align_proba
        probs = np.mean(
            [_align_proba(m, X_in) for m in self._ens._final_base],
            axis=0,
        )
        return self._ens._meta.predict_proba(probs)

    def predict(self, X_in: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X_in)
        return self.classes_[np.argmax(proba, axis=1)]

    def fit(self, X_in, y_in):
        return self


@dataclass
class TrainingPipeline:
    settings: Settings

    def run(self, league: str, version: str) -> ModelMetadata:
        """Train ensemble for one league — ML-01 + ML-02 + ML-03 + ML-04."""
        logger.info("pipeline_start", league=league, version=version)
        store = ParquetStore(base_path=Path(self.settings.parquet_base_path))

        # Load all historical features for the league
        df = store.read_features(sport="football", league=league)
        if len(df) == 0:
            raise RuntimeError(
                f"No feature rows for league={league}. "
                f"Run scripts/seed_historical.py first."
            )

        # Phase 02.1 (D-05): inner-join finished results, left-join Betano odds
        # on fixture_id. Replaces the Phase 2 home_goals schema guard with a
        # join-produces-zero-rows runtime check (data pipeline failure, not
        # feature-store schema failure).
        df_results = store.read_results(sport="football", league=league)
        df_odds = store.read_odds(sport="football", league=league)
        # WR-01: empty stores produce zero-column DataFrames; pl.col("status")
        # / pl.col("bookmaker") on those raises ColumnNotFoundError BEFORE the
        # post-join len(df) == 0 guard can fire. Front-load a clear "run the
        # seed script" message instead of leaking a Polars stack trace to the
        # fresh-installer.
        if df_results.is_empty() or "status" not in df_results.columns:
            raise RuntimeError(
                f"No results rows for league={league}. "
                f"Run scripts/seed_historical.py first."
            )
        if df_odds.is_empty() or "bookmaker" not in df_odds.columns:
            raise RuntimeError(
                f"No odds rows for league={league}. "
                f"Run scripts/seed_historical.py first."
            )
        df = df.join(
            df_results.filter(pl.col("status") == "finished").select(
                ["fixture_id", "home_goals", "away_goals", "status"]
            ),
            on="fixture_id",
            how="inner",
        )
        df = df.join(
            df_odds.filter(pl.col("bookmaker") == "Betano").select(
                ["fixture_id", "opening_home", "opening_draw", "opening_away"]
            ),
            on="fixture_id",
            how="left",
        )
        if len(df) == 0:
            raise RuntimeError(
                f"features ⋈ results ⋈ odds produced zero rows for league={league}. "
                f"Run scripts/seed_historical.py first."
            )

        # Expect columns: fixture_id, computed_at, and feature_* columns
        # Drop partition columns + metadata for the feature matrix
        drop_cols = {
            "fixture_id", "sport", "league", "season", "matchday", "computed_at",
            "status",  # joined in from results, not a feature
            "opening_home", "opening_draw", "opening_away",  # joined in from odds
        }
        y = np.array([
            _label_from_goals(int(hg), int(ag))
            for hg, ag in zip(df["home_goals"], df["away_goals"], strict=True)
        ])
        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and c not in {"home_goals", "away_goals"}
        ]
        feature_cols.sort()
        X = df.select(feature_cols).to_numpy()
        dates_col = df["computed_at"].to_list()
        dates = np.array([
            datetime.fromisoformat(d) if isinstance(d, str) else d
            for d in dates_col
        ])

        # Extract opening odds as (n_rows, 3) numpy matrix; NaN where Betano
        # coverage is missing (D-03). closing proxy = opening in 02.1 (D-02).
        opening_home_col = df["opening_home"].to_numpy(allow_copy=True).astype(float)
        opening_draw_col = df["opening_draw"].to_numpy(allow_copy=True).astype(float)
        opening_away_col = df["opening_away"].to_numpy(allow_copy=True).astype(float)
        opening_matrix = np.column_stack(
            [opening_home_col, opening_draw_col, opening_away_col]
        )
        closing_matrix = opening_matrix.copy()  # D-02: closing proxy = opening
        null_clv_rows_total: int = 0

        # Sort by date to guarantee monotonic increasing
        order = np.argsort(dates)
        X, y, dates = X[order], y[order], dates[order]
        opening_matrix = opening_matrix[order]
        closing_matrix = closing_matrix[order]

        # Walk-forward CV + nested OOF (ML-02, D-03b)
        splitter = WalkForwardSplitter(n_splits=5)
        fold_details: list[dict] = []
        all_test_probs: list[np.ndarray] = []
        all_test_idx: list[np.ndarray] = []
        last_ensemble: StackedEnsemble | None = None
        for fold_idx, train_idx, test_idx in splitter.split(X, dates):
            ens = StackedEnsemble()
            probs = ens.fit_fold(
                X[train_idx], y[train_idx], X[test_idx], dates[train_idx]
            )
            all_test_probs.append(probs)
            all_test_idx.append(test_idx)
            last_ensemble = ens

            # Phase 02.1 — per-fold CLV via simulate_pick + apply_slippage +
            # compute_clv (D-09, D-10, D-11). Skip rows where Betano coverage is
            # missing (D-03) or no outcome clears EDGE_THRESHOLD_PCT.
            fold_clvs: list[float] = []
            null_this_fold = 0
            opening_i = opening_matrix[test_idx]   # shape (n_test, 3)
            closing_i = closing_matrix[test_idx]
            for i in range(len(test_idx)):
                o_sel = simulate_pick(probs[i], opening_i[i])  # D-09 + D-10
                if o_sel is None:
                    null_this_fold += 1
                    continue
                staked = apply_slippage(float(opening_i[i][o_sel]))
                if np.isnan(closing_i[i][o_sel]):
                    null_this_fold += 1
                    continue
                fold_clvs.append(compute_clv(staked, float(closing_i[i][o_sel])))

            n_picks = len(fold_clvs)
            # D-11: require >=20 picks for a fold's mean CLV to be meaningful
            fold_clv_pct = float(np.mean(fold_clvs)) if n_picks >= 20 else None

            # D-14: per-fold uncalibrated logloss with explicit labels=[0,1,2]
            # (research finding 2: missing-class folds crash without explicit labels).
            fold_logloss_uncal = float(
                log_loss(y[test_idx], probs, labels=[0, 1, 2])
            )

            fold_details.append({
                "fold_idx": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_picks": n_picks,
                "clv_pct": fold_clv_pct,
                "logloss_uncalibrated": fold_logloss_uncal,
                "logloss_calibrated": None,  # filled on last fold after calibrate()
            })
            null_clv_rows_total += null_this_fold

        assert last_ensemble is not None, "Walk-forward produced zero folds"

        # Calibration on the last fold's test window (ML-03)
        last_test_idx = all_test_idx[-1]
        calib_method = select_calibrator(len(last_test_idx))

        wrapper = _EnsembleProbaWrapper(
            last_ensemble,
            X_fit=X[:max(all_test_idx[-1][0], 1)],
            y_fit=y[:max(all_test_idx[-1][0], 1)],
            dates_fit=dates[:max(all_test_idx[-1][0], 1)],
        )
        calibrator = calibrate(wrapper, X[last_test_idx], y[last_test_idx])

        # D-13: overall logloss delta on the last fold's test window.
        # labels=[0, 1, 2] is mandatory in EVERY log_loss call (research finding 2).
        raw_probs_last = wrapper.predict_proba(X[last_test_idx])
        cal_probs_last = calibrator.predict_proba(X[last_test_idx])
        logloss_uncal = float(
            log_loss(y[last_test_idx], raw_probs_last, labels=[0, 1, 2])
        )
        logloss_cal = float(
            log_loss(y[last_test_idx], cal_probs_last, labels=[0, 1, 2])
        )
        logloss_improvement_pct = (
            (logloss_uncal - logloss_cal) / logloss_uncal * 100.0
            if logloss_uncal > 0
            else 0.0
        )
        # Back-fill last fold's calibrated logloss into fold_details (D-14)
        fold_details[-1]["logloss_calibrated"] = logloss_cal

        # D-11 / D-13: walk-forward mean CLV across qualifying folds (>=20 picks).
        qualifying_clvs = [
            f["clv_pct"] for f in fold_details if f["clv_pct"] is not None
        ]
        walk_forward_mean_clv_pct = (
            float(np.mean(qualifying_clvs)) if qualifying_clvs else None
        )

        # Persist artifacts to models/football/{league}/{version}/
        artifact_dir = (
            Path(self.settings.model_dir) / "football" / league / version
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(last_ensemble, artifact_dir / "ensemble.joblib")
        joblib.dump(calibrator, artifact_dir / "calibrator.joblib")

        meta = ModelMetadata(
            league=league,
            version=version,
            training_date=datetime.now(UTC),
            training_data_seasons=["2023-2024", "2024-2025", "2025-2026"],
            training_rows=int(len(X)),
            feature_names=feature_cols,
            feature_set_hash=feature_set_hash(feature_cols),
            calibration_method=calib_method.value,
            calibration_samples=int(len(last_test_idx)),
            walk_forward_folds=len(fold_details),
            walk_forward_mean_clv_pct=walk_forward_mean_clv_pct,
            walk_forward_fold_details=fold_details,
            base_model_params={
                "xgboost": XGB_PARAMS,
                "catboost": CB_PARAMS,
                "lightgbm": LGBM_PARAMS,
            },
            base_model_packages=last_ensemble.base_model_packages(),
            sklearn_version=sklearn.__version__,
            null_clv_rows=null_clv_rows_total,
            slippage_pct=SLIPPAGE_PCT,
            # Phase 02.1 (D-13): logloss before/after calibration on last fold
            logloss_uncalibrated=logloss_uncal,
            logloss_calibrated=logloss_cal,
            logloss_improvement_pct=logloss_improvement_pct,
        )
        (artifact_dir / "metadata.json").write_text(
            json.dumps(meta.to_dict(), indent=2, sort_keys=True)
        )
        logger.info(
            "pipeline_done",
            league=league,
            version=version,
            mean_clv_pct=walk_forward_mean_clv_pct,
            logloss_improvement_pct=logloss_improvement_pct,
            null_clv_rows=null_clv_rows_total,
            rows=len(X),
        )
        return meta
