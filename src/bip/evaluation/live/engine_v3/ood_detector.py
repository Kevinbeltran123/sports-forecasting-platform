"""Mahalanobis-based OOD detector for the GSV (Risk #2 mitigation).

Design doc, section 8, Risk #2:

> Conditional predictor "default to global mean" en estados raros —
> Predicciones convergen a la media incondicional en OOD; % picks con
> interval ancho creciendo.
>
> Mitigación: OOD detector explícito (Mahalanobis sobre GSV vs training
> distribution); no-bet por regla #8 cuando OOD.

This module realizes that mitigation. The detector projects a GSV onto
a fixed numeric vector, learns the mean and (regularized) covariance
of the training distribution, and scores each new GSV by Mahalanobis
distance from the training centroid. Anything past the 99th percentile
of training scores is flagged OOD.

Feature schema history:
  v1 (17-feature): included total_goals, xg_total, goal_diff — these
    are correlated with the target and contributed to distribution shift
    when high-scoring MLS/Swiss games (total_goals 8-12) appeared.
    The v2 detector (ood_detector_v2_real.pkl) was fit on this schema.
  v2 (14-feature, current FEATURE_NAMES): drop total_goals, xg_total,
    goal_diff. Forward-looking refit schema. New fits use vectorize_gsv
    (14-dim). Legacy-loaded detectors use vectorize_gsv_legacy (17-dim)
    to score without shape mismatch.

OOD scoring path:
  OODDetector.score() checks the detector's own ``feature_names`` field
  (persisted with joblib). If it matches the legacy 17-feature set,
  vectorize_gsv_legacy is used. Otherwise vectorize_gsv (14-feature).

The regularization is Ledoit-Wolf-style shrinkage scaled by
``tr(Σ)/d``: this is well-defined even when n < d, so the detector
degrades gracefully on small training sets (cold start). When not
fitted, ``is_ood`` returns ``False`` with a structlog warning — the
fail-safe is to NOT block.

No new dependencies: numpy + joblib are already on the stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import structlog

from bip.evaluation.live.engine_v3.gsv import GameStateVector

_log = structlog.get_logger(__name__)


# ── Feature projection ───────────────────────────────────────────────────


# FORWARD-LOOKING schema (14 features) — used for new refits and by
# vectorize_gsv(). Drops the three forward-looking/correlated features
# that caused distribution shift in MLS/Swiss high-scoring games:
#   - total_goals  (correlated with score → OOD for high-scoring states)
#   - xg_total     (same issue)
#   - goal_diff    (directly reflects score; better captured by minutes_since_last_goal)
# To refit with this schema: run an offline research script (not published)
# (do NOT run it now — refit is a deliberate offline step after shadow accrual).
FEATURE_NAMES: tuple[str, ...] = (
    "minute",
    "numerical_advantage",
    "xg_diff",
    "xg_vs_score_divergence",
    "xg_per_min_home_last_15",
    "xg_per_min_away_last_15",
    "possession_home_5min",
    "attacks_last_10min",
    "dangerous_attacks_last_10min",
    "total_corners",
    "corner_rate_last_15min",
    "total_yellows",
    "card_rate_last_15min",
    "minutes_since_last_goal",
)
N_FEATURES = len(FEATURE_NAMES)

# LEGACY schema (17 features) — the full feature set that ood_detector_v2_real.pkl
# was fitted on. Required to score the loaded legacy pkl without a shape mismatch.
_FEATURE_NAMES_LEGACY: tuple[str, ...] = (
    "minute",
    "goal_diff",
    "total_goals",
    "numerical_advantage",
    "xg_diff",
    "xg_total",
    "xg_vs_score_divergence",
    "xg_per_min_home_last_15",
    "xg_per_min_away_last_15",
    "possession_home_5min",
    "attacks_last_10min",
    "dangerous_attacks_last_10min",
    "total_corners",
    "corner_rate_last_15min",
    "total_yellows",
    "card_rate_last_15min",
    "minutes_since_last_goal",
)
_N_FEATURES_LEGACY = len(_FEATURE_NAMES_LEGACY)  # 17


def vectorize_gsv(gsv: GameStateVector) -> np.ndarray:
    """Project a GSV into the 14-feature forward vector.

    Missing values map to 0.0 by construction — the GSV schema already
    fills defaults so this is rarely needed, but the tuple-sum guards
    against tuples being None at edge cases.

    Use this function for NEW refits (an offline research script (not published)).
    For scoring the legacy 17-feature pkl use vectorize_gsv_legacy.
    """
    a_home, a_away = gsv.flow.attacks_last_10min
    da_home, da_away = gsv.flow.dangerous_attacks_last_10min
    y_home, y_away = gsv.cards.yellows
    return np.asarray(
        [
            float(gsv.time.minute),
            float(gsv.numerical.numerical_advantage),
            float(gsv.xg.xg_diff),
            float(gsv.xg.xg_vs_score_divergence),
            float(gsv.xg.xg_per_min_home_last_15),
            float(gsv.xg.xg_per_min_away_last_15),
            float(gsv.flow.possession_home_5min),
            float((a_home or 0) + (a_away or 0)),
            float((da_home or 0) + (da_away or 0)),
            float(gsv.corners.corners_home + gsv.corners.corners_away),
            float(gsv.corners.corner_rate_last_15min),
            float((y_home or 0) + (y_away or 0)),
            float(gsv.cards.card_rate_last_15min),
            float(gsv.score.minutes_since_last_goal),
        ],
        dtype=np.float64,
    )


def vectorize_gsv_legacy(gsv: GameStateVector) -> np.ndarray:
    """Project a GSV into the legacy 17-feature vector.

    Used ONLY to score OODDetector instances that were fit on the
    17-feature schema (ood_detector_v2_real.pkl). Do NOT use for new
    refits — new fits should use vectorize_gsv (14-feature).
    """
    a_home, a_away = gsv.flow.attacks_last_10min
    da_home, da_away = gsv.flow.dangerous_attacks_last_10min
    y_home, y_away = gsv.cards.yellows
    return np.asarray(
        [
            float(gsv.time.minute),
            float(gsv.score.goal_diff),
            float(gsv.score.home_goals + gsv.score.away_goals),
            float(gsv.numerical.numerical_advantage),
            float(gsv.xg.xg_diff),
            float(gsv.xg.home_xg_total + gsv.xg.away_xg_total),
            float(gsv.xg.xg_vs_score_divergence),
            float(gsv.xg.xg_per_min_home_last_15),
            float(gsv.xg.xg_per_min_away_last_15),
            float(gsv.flow.possession_home_5min),
            float((a_home or 0) + (a_away or 0)),
            float((da_home or 0) + (da_away or 0)),
            float(gsv.corners.corners_home + gsv.corners.corners_away),
            float(gsv.corners.corner_rate_last_15min),
            float((y_home or 0) + (y_away or 0)),
            float(gsv.cards.card_rate_last_15min),
            float(gsv.score.minutes_since_last_goal),
        ],
        dtype=np.float64,
    )


# ── Detector ─────────────────────────────────────────────────────────────


@dataclass
class OODDetector:
    """Mahalanobis OOD detector for the GSV.

    Fit once on a reference distribution, then call ``is_ood(gsv)`` per
    game state. ``threshold`` defaults to the 99th percentile of
    training scores — i.e., 1% of training samples themselves are
    flagged, calibrating the false-positive rate.
    """

    mean: np.ndarray | None = None
    cov_inv: np.ndarray | None = None
    threshold: float = math.inf  # before fitting, never flag
    n_train: int = 0
    is_fitted: bool = False
    feature_names: tuple[str, ...] = field(default=FEATURE_NAMES)
    regularization: float = 1e-2

    def fit(
        self,
        gsvs: Iterable[GameStateVector],
        *,
        regularization: float | None = None,
        threshold_percentile: float = 99.0,
    ) -> "OODDetector":
        """Estimate mean + regularized covariance from reference GSVs.

        ``regularization`` (default ``self.regularization``) is the
        Ledoit-Wolf-style shrinkage factor: we add ``r × tr(Σ)/d × I``
        to the empirical covariance before inverting. This keeps the
        inverse well-defined even when n < d.

        ``threshold_percentile`` sets the OOD cutoff against the
        training distribution itself.
        """
        if regularization is not None:
            self.regularization = regularization

        X = np.asarray([vectorize_gsv(g) for g in gsvs], dtype=np.float64)
        if X.size == 0:
            raise ValueError("OODDetector.fit: no GSVs provided")
        if X.shape[1] != N_FEATURES:
            raise ValueError(f"feature width mismatch: got {X.shape[1]}, want {N_FEATURES}")

        n = X.shape[0]
        mu = X.mean(axis=0)
        Xc = X - mu
        # Sample covariance (use n for stability when n is small;
        # bias is absorbed by shrinkage).
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        d = cov.shape[0]
        trace_per_dim = float(np.trace(cov) / d) if d > 0 else 1.0
        # Avoid degenerate shrinkage if cov is exactly zero (all-identical training).
        shrink_scale = trace_per_dim if trace_per_dim > 1e-12 else 1.0
        cov_reg = cov + self.regularization * shrink_scale * np.eye(d)

        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)

        # Score the training set itself → choose threshold by percentile.
        d2 = np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc)
        scores = np.sqrt(np.clip(d2, 0.0, None))
        threshold = float(np.percentile(scores, threshold_percentile))

        self.mean = mu
        self.cov_inv = cov_inv
        self.threshold = threshold
        self.n_train = n
        self.is_fitted = True
        return self

    def _select_vectorizer(self):
        """Return the correct vectorize function for this detector's schema.

        If the detector was persisted with the legacy 17-feature schema
        (``_FEATURE_NAMES_LEGACY``), use ``vectorize_gsv_legacy`` so the
        vector dimension matches the stored mean/cov_inv. Otherwise use
        the current 14-feature ``vectorize_gsv``.

        This prevents the shape mismatch that would occur if we scored
        the legacy ood_detector_v2_real.pkl (17-dim) with the new
        14-feature vectorizer. See known_correctness_constraint #1.
        """
        if self.feature_names == _FEATURE_NAMES_LEGACY:
            return vectorize_gsv_legacy
        return vectorize_gsv

    def score(self, gsv: GameStateVector) -> float:
        """Mahalanobis distance from the training centroid.

        Returns ``0.0`` if not fitted. Callers that care about the
        unfitted state should check ``is_fitted``.

        Automatically selects the correct vectorizer based on the
        detector's persisted ``feature_names`` (legacy 17-dim vs
        forward 14-dim) so a loaded legacy pkl scores without a shape
        mismatch.
        """
        if not self.is_fitted:
            return 0.0
        assert self.mean is not None and self.cov_inv is not None
        vec_fn = self._select_vectorizer()
        v = vec_fn(gsv) - self.mean
        d2 = float(v @ self.cov_inv @ v)
        return math.sqrt(max(d2, 0.0))

    def is_ood(
        self,
        gsv: GameStateVector,
        *,
        threshold: float | None = None,
    ) -> bool:
        """Flag the GSV as out-of-distribution.

        When not fitted, returns ``False`` (fail-safe) and emits a
        structlog warning. Callers wanting to enforce "always-fitted"
        should check ``is_fitted`` themselves and refuse to run."""
        if not self.is_fitted:
            _log.warning("ood_detector_unfitted_returning_false")
            return False
        s = self.score(gsv)
        t = self.threshold if threshold is None else threshold
        return s > t

    # ── persistence ────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "OODDetector":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"loaded object is {type(obj)!r}, not OODDetector")
        return obj


__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "OODDetector",
    "vectorize_gsv",
    "vectorize_gsv_legacy",
]
