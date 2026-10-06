"""Beta calibration — Kull, Silva Filho & Flach (2017).

Implements

    q = σ( a · log(p) − b · log(1 − p) + c )

per-class. This is strictly more general than the existing Ojeda-style
``LogisticLogitCalibrator`` (which is the constrained ``a = b`` case): the
extra degree of freedom lets it correct asymmetric miscalibration that
Platt-style scaling cannot. At a=1, b=1, c=0 the map collapses to the
identity, so an "uncalibrated" diagnostic is reachable as a limit.

Reference: Kull, Silva Filho, Flach (2017), *Beyond sigmoids: How to
obtain well-calibrated probabilities from binary classifiers with beta
calibration*. Electronic Journal of Statistics 11(2): 5052–5080.
DOI: 10.1214/17-EJS1338SI.

Why this module exists alongside ``LogisticLogitCalibrator``:
- WC2026 lock_v2 spike — keep the existing Ojeda calibrator as-is so the
  current lock keeps working; ship Beta as an additive component so the
  ablation in Ola 6 can isolate its marginal effect.

Interface mirrors ``LogisticLogitCalibrator`` for substitutability:
``fit / transform / fit_transform / to_dict / from_dict``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

# Probability clipping to keep log(p) and log(1-p) finite.
_EPS = 1e-7

# Minimum per-class positive/negative samples before falling back to
# identity transform. Matches the existing LogisticLogitCalibrator default
# and is consistent with Kull et al. 2017's note that ML estimation of
# (a, b, c) is unreliable below ~50 per-class samples.
_MIN_CLASS_SAMPLES = 50


def _nll_binary(params: np.ndarray, log_p: np.ndarray, log_1mp: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for one class.

    z = a · log(p) − b · log(1−p) + c
    NLL = −Σ [ y·log_expit(z) + (1−y)·log_expit(−z) ]
    """

    a, b, c = params
    z = a * log_p - b * log_1mp + c
    return float(-np.sum(y * log_expit(z) + (1.0 - y) * log_expit(-z)))


def _grad_binary(
    params: np.ndarray,
    log_p: np.ndarray,
    log_1mp: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Analytical gradient of _nll_binary.

    d/dparam Σ -[y log σ(z) + (1-y) log σ(-z)]  =  Σ (σ(z) - y) · dz/dparam
    """

    a, b, c = params
    z = a * log_p - b * log_1mp + c
    diff = expit(z) - y
    return np.array(
        [
            float(np.sum(diff * log_p)),
            float(np.sum(diff * (-log_1mp))),
            float(np.sum(diff)),
        ],
        dtype=np.float64,
    )


def _fit_class_params(
    p: np.ndarray, y: np.ndarray, *, min_samples: int
) -> tuple[float, float, float]:
    """Fit (a, b, c) for one class column. Falls back to identity below
    ``min_samples`` per-class positive or negative observations."""

    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    if n_pos < min_samples or n_neg < min_samples:
        return 1.0, 1.0, 0.0

    p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
    log_p = np.log(p_clipped)
    log_1mp = np.log(1.0 - p_clipped)

    result = minimize(
        _nll_binary,
        x0=np.array([1.0, 1.0, 0.0]),
        jac=_grad_binary,
        args=(log_p, log_1mp, y),
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


class BetaCalibrator:
    """Beta-calibrated probability map — q = σ(a·log p − b·log(1−p) + c).

    Binary usage (1-D probs of class 1):
        cal = BetaCalibrator()
        cal.fit(p_1d, y_0_1)
        q_1d = cal.transform(p_1d)

    Multiclass usage (2-D probs of shape (N, K)):
        cal = BetaCalibrator()
        cal.fit(p_matrix, y_0_to_Kminus1)
        q_matrix = cal.transform(p_matrix)   # rows sum to 1
    """

    def __init__(self, *, min_samples_per_class: int = _MIN_CLASS_SAMPLES) -> None:
        self._min_samples = min_samples_per_class
        self._a: tuple[float, ...] | None = None
        self._b: tuple[float, ...] | None = None
        self._c: tuple[float, ...] | None = None
        self._n_classes: int | None = None

    def fit(
        self,
        probs: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
        outcomes: Sequence[int] | np.ndarray,
    ) -> Self:
        probs_arr = np.asarray(probs, dtype=float)
        outcomes_arr = np.asarray(outcomes, dtype=int)

        if probs_arr.ndim == 1:
            probs_arr = np.column_stack([1.0 - probs_arr, probs_arr])

        n, k = probs_arr.shape
        if len(outcomes_arr) != n:
            raise ValueError(f"probs and outcomes length mismatch: {n} vs {len(outcomes_arr)}")

        a_list: list[float] = []
        b_list: list[float] = []
        c_list: list[float] = []
        for cls in range(k):
            col = probs_arr[:, cls]
            y_cls = (outcomes_arr == cls).astype(float)
            a, b, c = _fit_class_params(col, y_cls, min_samples=self._min_samples)
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)

        self._a = tuple(a_list)
        self._b = tuple(b_list)
        self._c = tuple(c_list)
        self._n_classes = k
        return self

    def transform(
        self,
        probs: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        if self._a is None or self._b is None or self._c is None:
            raise RuntimeError("BetaCalibrator has not been fitted yet")

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
            log_p = np.log(col)
            log_1mp = np.log(1.0 - col)
            z = self._a[cls] * log_p - self._b[cls] * log_1mp + self._c[cls]
            result[:, cls] = expit(z)

        # Renormalise rows so probabilities sum to 1 (multiclass) or 2-col
        # binary contract stays consistent.
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
        return self.fit(probs, outcomes).transform(probs)

    def to_dict(self) -> dict[str, Any]:
        if self._a is None or self._b is None or self._c is None:
            raise RuntimeError("BetaCalibrator has not been fitted yet")
        return {
            "kind": "beta",
            "n_classes": self._n_classes,
            "a": list(self._a),
            "b": list(self._b),
            "c": list(self._c),
            "min_samples_per_class": self._min_samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        if payload.get("kind") != "beta":
            raise ValueError(f"Expected kind='beta', got {payload.get('kind')!r}")
        instance = cls(
            min_samples_per_class=int(payload.get("min_samples_per_class", _MIN_CLASS_SAMPLES))
        )
        instance._a = tuple(float(v) for v in payload["a"])
        instance._b = tuple(float(v) for v in payload["b"])
        instance._c = tuple(float(v) for v in payload["c"])
        instance._n_classes = int(payload["n_classes"])
        return instance
