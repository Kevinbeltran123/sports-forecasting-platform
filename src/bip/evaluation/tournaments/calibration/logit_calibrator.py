"""Logit-space calibration for tournament-evaluator predictors.

`LogisticLogitCalibrator` implements Ojeda 2023 Step 3:

    p_cal = σ(α + β · logit(p_raw))

where logit(p) = log(p / (1-p)) and σ is the logistic (sigmoid) function.

Fit one (α, β) pair per class via maximum-likelihood logistic regression on
logit-transformed raw probabilities. For multiclass (K>2), apply K independent
binary calibrations on the K probability columns, then renormalise rows to sum
to 1.

Why logit-space over isotonic:
    Isotonic regression fits a step-function. When the calibration set has a
    different probability distribution than the test set (covariate shift —
    which club→national-team transfer *is by definition*), the steps land in
    the wrong probability bins. The logit model has only 2 free parameters so
    it generalises far better under shift.

    Ojeda 2023 tested 23 shift scenarios: Beta and Logistic calibrators passed
    all 23; isotonic failed 18 of 23. See internal research notes
    NOTAS_ojeda2023_calibracion_ml.md.

References:
    Ojeda 2023 — *Calibration of probabilistic classifiers under covariate shift*
    Walsh & Joshi 2024 — *Calibration of probabilistic predictions for sports betting*
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.special import logit as scipy_logit

# Probability clipping to keep logit finite.
_EPS = 1e-7

# Minimum positive-class samples before falling back to identity transform
# (α=0, β=1 → p_cal = p_raw). Below this the MLE is unreliable.
_MIN_CLASS_SAMPLES = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nll_binary(params: np.ndarray, logit_x: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for binary logistic calibration.

    NLL = -Σ [ y_i log σ(z_i) + (1−y_i) log(1−σ(z_i)) ]
        = -Σ [ y_i log_expit(z_i) + (1−y_i) log_expit(−z_i) ]
    """
    alpha, beta = params
    z = alpha + beta * logit_x
    return float(-np.sum(y * log_expit(z) + (1.0 - y) * log_expit(-z)))


def _fit_class_params(
    logit_x: np.ndarray, y: np.ndarray, *, min_samples: int
) -> tuple[float, float]:
    """Fit (α, β) for one class column.

    Falls back to identity transform (α=0, β=1) when there are fewer than
    `min_samples` positive or negative examples — MLE is unreliable on
    small class-conditional samples and isotonic-style step fitting would
    overfit even more.
    """
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    if n_pos < min_samples or n_neg < min_samples:
        return 0.0, 1.0

    result = minimize(
        _nll_binary,
        x0=np.array([0.0, 1.0]),
        args=(logit_x, y),
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1])


# ---------------------------------------------------------------------------
# LogisticLogitCalibrator
# ---------------------------------------------------------------------------


class LogisticLogitCalibrator:
    """Logit-space linear calibrator — p_cal = σ(α + β · logit(p_raw)).

    Binary markets (pass probs as 1-D array of P(class=1)):
        cal = LogisticLogitCalibrator()
        cal.fit(probs_1d, outcomes_0_1)
        calibrated_1d = cal.transform(probs_1d)

    Multiclass markets (pass probs as 2-D array of shape (N, K)):
        cal = LogisticLogitCalibrator()
        cal.fit(probs_matrix, outcomes_0_to_K)
        calibrated_matrix = cal.transform(probs_matrix)  # rows sum to 1
    """

    def __init__(self, *, min_samples_per_class: int = _MIN_CLASS_SAMPLES) -> None:
        self._min_samples = min_samples_per_class
        self._alphas: tuple[float, ...] | None = None
        self._betas: tuple[float, ...] | None = None
        self._n_classes: int | None = None

    # ---- fitting ------------------------------------------------------------

    def fit(
        self,
        probs: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
        outcomes: Sequence[int] | np.ndarray,
    ) -> Self:
        """Fit (α, β) per class from raw probabilities and observed outcomes.

        Args:
            probs:    1-D for binary (P(class=1) per sample) or 2-D (N, K)
                      for multiclass.
            outcomes: Class indices — integers in {0} for binary or
                      {0, …, K−1} for multiclass.
        """
        probs_arr = np.asarray(probs, dtype=float)
        outcomes_arr = np.asarray(outcomes, dtype=int)

        if probs_arr.ndim == 1:
            # Binary: treat as 2-column matrix so the loop below is uniform.
            probs_arr = np.column_stack([1.0 - probs_arr, probs_arr])

        n, k = probs_arr.shape
        if len(outcomes_arr) != n:
            raise ValueError(
                f"probs and outcomes length mismatch: {n} vs {len(outcomes_arr)}"
            )

        alphas: list[float] = []
        betas: list[float] = []
        for cls in range(k):
            col = np.clip(probs_arr[:, cls], _EPS, 1.0 - _EPS)
            logit_x = scipy_logit(col)
            y_cls = (outcomes_arr == cls).astype(float)
            a, b = _fit_class_params(logit_x, y_cls, min_samples=self._min_samples)
            alphas.append(a)
            betas.append(b)

        self._alphas = tuple(alphas)
        self._betas = tuple(betas)
        self._n_classes = k
        return self

    def transform(
        self,
        probs: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Apply fitted calibration to raw probabilities.

        Returns an array of the same shape as the input.  Multiclass rows
        are renormalised to sum to 1.0 after per-class calibration.

        Raises:
            RuntimeError: if called before `fit`.
        """
        if self._alphas is None:
            raise RuntimeError("LogisticLogitCalibrator has not been fitted yet")

        probs_arr = np.asarray(probs, dtype=float)
        binary_input = probs_arr.ndim == 1
        if binary_input:
            probs_arr = np.column_stack([1.0 - probs_arr, probs_arr])

        _n, k = probs_arr.shape
        if k != self._n_classes:
            raise ValueError(
                f"probs has {k} columns but calibrator was fitted on {self._n_classes}"
            )

        result = np.empty_like(probs_arr)
        for cls in range(k):
            col = np.clip(probs_arr[:, cls], _EPS, 1.0 - _EPS)
            logit_x = scipy_logit(col)
            result[:, cls] = expit(self._alphas[cls] + self._betas[cls] * logit_x)

        row_sums = result.sum(axis=1, keepdims=True)
        result /= np.maximum(row_sums, _EPS)

        if binary_input:
            return result[:, 1]
        return result

    def fit_transform(
        self,
        probs: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
        outcomes: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit then transform the same data."""
        return self.fit(probs, outcomes).transform(probs)

    # ---- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise fitted parameters to a plain dict (JSON-safe)."""
        if self._alphas is None:
            raise RuntimeError("Calibrator has not been fitted yet")
        return {
            "alphas": list(self._alphas),
            "betas": list(self._betas),
            "n_classes": self._n_classes,
            "min_samples_per_class": self._min_samples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Restore a fitted calibrator from a serialised dict."""
        cal = cls(min_samples_per_class=d.get("min_samples_per_class", _MIN_CLASS_SAMPLES))
        cal._alphas = tuple(d["alphas"])
        cal._betas = tuple(d["betas"])
        cal._n_classes = int(d["n_classes"])
        return cal

    # ---- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        if self._alphas is None:
            return "LogisticLogitCalibrator(not fitted)"
        params = ", ".join(
            f"cls{k}=(α={a:.4f}, β={b:.4f})"
            for k, (a, b) in enumerate(zip(self._alphas, self._betas))
        )
        return f"LogisticLogitCalibrator({params})"


# ---------------------------------------------------------------------------
# Deprecated shim
# ---------------------------------------------------------------------------


class _IsotonicCalibrator:
    """DEPRECATED — use `LogisticLogitCalibrator` instead.

    Isotonic regression fails under covariate shift: Ojeda 2023 found it
    failed 18 of 23 simulation scenarios while the logistic calibrator passed
    all 23.  Club→national-team transfer is by definition a covariate shift;
    the calibration set (club matches) differs from the test distribution
    (international fixtures).

    This shim exists to raise a `DeprecationWarning` if any code path
    attempts to instantiate the naive isotonic approach, and still function
    correctly for binary calibration so callers can migrate without an
    immediate breakage.
    """

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn(
            "_IsotonicCalibrator is deprecated. "
            "Use LogisticLogitCalibrator (Ojeda 2023 Step 3) which is robust "
            "under covariate shift. "
            "Isotonic regression failed 18/23 shift scenarios in Ojeda 2023; "
            "the logistic calibrator passed all 23.",
            DeprecationWarning,
            stacklevel=2,
        )
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> Self:
        """Fit isotonic regression on 1-D binary probabilities (P(class=1))."""
        self._iso.fit(list(probs), list(outcomes))
        self._fitted = True
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        """Apply fitted isotonic mapping."""
        return self._iso.transform(list(probs))

    def fit_transform(
        self, probs: Sequence[float], outcomes: Sequence[int]
    ) -> np.ndarray:
        return self.fit(probs, outcomes).transform(probs)
