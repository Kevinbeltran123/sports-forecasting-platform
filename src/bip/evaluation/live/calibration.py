"""Probability calibration for live picks.

Day-1 audit (2026-05-10) found the predictor's `our_probability` is
systematically overconfident at the top of the distribution: D9-D10
(p >= 0.87) show a 17pp gap between predicted and empirical win rate.
The Tier 1.4 Kelly haircut in ValueDetector is a stopgap; this module
is the principled replacement.

Design:

- Isotonic regression on (predicted_probability, won_int) pairs.
- Fit offline from `picks_graded.parquet` via an offline fitting script
  (not published); result persists to JSON.
- ValueDetector loads the calibrator at init and applies `transform()`
  to `our_probability` before Kelly/edge computation.
- When a calibrator is loaded, the Tier 1.4 haircut should be disabled
  (set `high_prob_haircut_threshold=1.01`) to avoid double-correction.

Why isotonic (not Platt):
- Day-1 reliability diagram is non-monotone in places (D6 underconfident,
  D9-D10 overconfident). Platt assumes a smooth sigmoid distortion; the
  empirical pattern fits a piecewise-monotone curve better.
- Isotonic is non-parametric and clips out-of-range probabilities, which
  is what we want for safety at the extremes.

Sample-size caveat:
- n=813 settled picks is borderline for isotonic at fine granularity.
  The implementation uses sklearn's default which fits a step function
  with as many breakpoints as the data supports. For Day-1 fit, this
  produces ~10-15 breakpoints — enough resolution without overfitting.
- For future side-aware variants, fit a separate calibrator per polarity
  (positive vs negative side); these would need n>=200 per side which
  Day-1 negative side (n=410) supports but positive side (n=105) does not.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicProbabilityCalibrator:
    """Isotonic regression mapping raw model probability -> calibrated.

    A piecewise-monotone, non-parametric calibrator. Stores breakpoints
    (x_thresholds, y_thresholds) for cheap interpolation at inference.
    """

    __slots__ = ("_x", "_y", "fitted_at", "n_train", "ece_before", "ece_after")

    def __init__(
        self,
        x_thresholds: list[float],
        y_thresholds: list[float],
        *,
        fitted_at: str | None = None,
        n_train: int = 0,
        ece_before: float = 0.0,
        ece_after: float = 0.0,
    ) -> None:
        if len(x_thresholds) != len(y_thresholds):
            raise ValueError(
                f"Mismatched threshold lengths: x={len(x_thresholds)}, "
                f"y={len(y_thresholds)}"
            )
        if len(x_thresholds) < 2:
            raise ValueError(
                f"Need at least 2 thresholds; got {len(x_thresholds)}"
            )
        # Validate monotonicity (sklearn guarantees this, but persisted
        # JSON could be edited by hand)
        x_arr = np.asarray(x_thresholds, dtype=float)
        y_arr = np.asarray(y_thresholds, dtype=float)
        if np.any(np.diff(x_arr) < 0):
            raise ValueError("x_thresholds must be non-decreasing")
        if np.any(np.diff(y_arr) < 0):
            raise ValueError("y_thresholds must be non-decreasing")
        self._x = x_arr
        self._y = y_arr
        self.fitted_at = fitted_at or datetime.now(UTC).isoformat()
        self.n_train = n_train
        self.ece_before = ece_before
        self.ece_after = ece_after

    def transform(self, p: float, *, market: str | None = None) -> float:
        """Map raw probability to calibrated probability.

        ``market`` is accepted for interface compatibility with
        ``PerMarketCalibrator`` but ignored — this is a global calibrator.

        Clips to the fitted x-range, then linearly interpolates between
        breakpoints. Output is clipped to [0, 1] for numerical safety.
        """
        del market  # unused — global calibrator
        if p <= self._x[0]:
            return float(np.clip(self._y[0], 0.0, 1.0))
        if p >= self._x[-1]:
            return float(np.clip(self._y[-1], 0.0, 1.0))
        return float(np.clip(np.interp(p, self._x, self._y), 0.0, 1.0))

    def transform_batch(self, probs: np.ndarray) -> np.ndarray:
        """Vectorized variant for offline use."""
        return np.clip(np.interp(probs, self._x, self._y), 0.0, 1.0)

    @classmethod
    def fit(
        cls,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        *,
        n_train: int | None = None,
    ) -> IsotonicProbabilityCalibrator:
        """Fit isotonic calibrator from prediction/outcome pairs."""
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        if len(raw_probs) != len(outcomes):
            raise ValueError(
                f"Mismatched lengths: probs={len(raw_probs)}, "
                f"outcomes={len(outcomes)}"
            )
        if len(raw_probs) < 10:
            raise ValueError(
                f"Need >=10 fit samples for isotonic; got {len(raw_probs)}"
            )
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_probs, outcomes)
        ece_before = _ece(raw_probs, outcomes)
        ece_after = _ece(iso.predict(raw_probs), outcomes)
        return cls(
            iso.X_thresholds_.tolist(),
            iso.y_thresholds_.tolist(),
            n_train=int(n_train if n_train is not None else len(raw_probs)),
            ece_before=ece_before,
            ece_after=ece_after,
        )

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "type": "isotonic",
            "version": 1,
            "x_thresholds": self._x.tolist(),
            "y_thresholds": self._y.tolist(),
            "fitted_at": self.fitted_at,
            "n_train": self.n_train,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
        }, indent=2))

    @classmethod
    def load_json(cls, path: Path | str) -> IsotonicProbabilityCalibrator:
        p = Path(path)
        d = json.loads(p.read_text())
        if d.get("type") != "isotonic":
            raise ValueError(
                f"Expected calibrator type='isotonic', got {d.get('type')!r}"
            )
        return cls(
            d["x_thresholds"],
            d["y_thresholds"],
            fitted_at=d.get("fitted_at"),
            n_train=int(d.get("n_train", 0)),
            ece_before=float(d.get("ece_before", 0.0)),
            ece_after=float(d.get("ece_after", 0.0)),
        )

    def __repr__(self) -> str:
        return (
            f"IsotonicProbabilityCalibrator(n_train={self.n_train}, "
            f"breakpoints={len(self._x)}, ece={self.ece_before:.4f}->"
            f"{self.ece_after:.4f})"
        )


class PerMarketCalibrator:
    """Per-market isotonic calibrator with a global fallback.

    Each market with at least ``min_samples_per_market`` settled picks
    gets its own ``IsotonicProbabilityCalibrator``. Markets below the
    threshold (or unseen markets at inference time) fall back to the
    global isotonic fit across the full dataset.

    Day-1 motivation: Phase 2C found `btts` swings 168pp if flipped
    (yes -> no). A global calibrator can't represent this because the
    correction depends on market. Per-market calibration captures the
    structural bias surgically — and is the natural form of the
    "BTTS coefficient recalibration" item from the standard TODO.

    Implements the same ``transform(p, *, market=None) -> float``
    interface as ``IsotonicProbabilityCalibrator``, so a ValueDetector
    can use either type interchangeably.
    """

    __slots__ = (
        "fallback", "per_market", "min_samples_per_market", "fitted_at",
        "n_train_total", "markets_with_own_fit", "markets_using_fallback",
    )

    def __init__(
        self,
        *,
        fallback: IsotonicProbabilityCalibrator,
        per_market: dict[str, IsotonicProbabilityCalibrator],
        min_samples_per_market: int = 25,
        fitted_at: str | None = None,
        n_train_total: int = 0,
        markets_with_own_fit: list[str] | None = None,
        markets_using_fallback: list[str] | None = None,
    ) -> None:
        self.fallback = fallback
        self.per_market = dict(per_market)
        self.min_samples_per_market = int(min_samples_per_market)
        self.fitted_at = fitted_at or datetime.now(UTC).isoformat()
        self.n_train_total = int(n_train_total)
        self.markets_with_own_fit = list(
            markets_with_own_fit
            if markets_with_own_fit is not None
            else sorted(per_market.keys())
        )
        self.markets_using_fallback = list(markets_using_fallback or [])

    def transform(self, p: float, *, market: str | None = None) -> float:
        """Map raw probability to calibrated probability.

        Uses the per-market calibrator when ``market`` is provided and
        has a fitted calibrator; otherwise falls back to the global fit.
        """
        if market is not None:
            cal = self.per_market.get(market)
            if cal is not None:
                return cal.transform(p)
        return self.fallback.transform(p)

    @classmethod
    def fit(
        cls,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        markets: list[str] | np.ndarray,
        *,
        min_samples_per_market: int = 25,
    ) -> PerMarketCalibrator:
        """Fit one isotonic per market (when n >= threshold) + global fallback."""
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        markets_arr = np.asarray(markets, dtype=object)
        if not (len(raw_probs) == len(outcomes) == len(markets_arr)):
            raise ValueError(
                f"Mismatched lengths: probs={len(raw_probs)}, "
                f"outcomes={len(outcomes)}, markets={len(markets_arr)}"
            )
        fallback = IsotonicProbabilityCalibrator.fit(
            raw_probs, outcomes, n_train=len(raw_probs),
        )
        per_market: dict[str, IsotonicProbabilityCalibrator] = {}
        with_own_fit: list[str] = []
        using_fallback: list[str] = []
        unique_markets = np.unique(markets_arr)
        for m in unique_markets:
            mask = markets_arr == m
            n_m = int(mask.sum())
            if n_m < min_samples_per_market:
                using_fallback.append(str(m))
                continue
            try:
                per_market[str(m)] = IsotonicProbabilityCalibrator.fit(
                    raw_probs[mask], outcomes[mask], n_train=n_m,
                )
                with_own_fit.append(str(m))
            except ValueError:
                # Edge cases (all-zero outcomes, etc.) — fallback
                using_fallback.append(str(m))
        return cls(
            fallback=fallback,
            per_market=per_market,
            min_samples_per_market=min_samples_per_market,
            n_train_total=len(raw_probs),
            markets_with_own_fit=sorted(with_own_fit),
            markets_using_fallback=sorted(using_fallback),
        )

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "type": "per_market",
            "version": 1,
            "min_samples_per_market": self.min_samples_per_market,
            "fitted_at": self.fitted_at,
            "n_train_total": self.n_train_total,
            "markets_with_own_fit": self.markets_with_own_fit,
            "markets_using_fallback": self.markets_using_fallback,
            "fallback": {
                "x_thresholds": self.fallback._x.tolist(),  # type: ignore[attr-defined]
                "y_thresholds": self.fallback._y.tolist(),  # type: ignore[attr-defined]
                "n_train": self.fallback.n_train,
                "ece_before": self.fallback.ece_before,
                "ece_after": self.fallback.ece_after,
            },
            "per_market": {
                mkt: {
                    "x_thresholds": cal._x.tolist(),  # type: ignore[attr-defined]
                    "y_thresholds": cal._y.tolist(),  # type: ignore[attr-defined]
                    "n_train": cal.n_train,
                    "ece_before": cal.ece_before,
                    "ece_after": cal.ece_after,
                }
                for mkt, cal in self.per_market.items()
            },
        }, indent=2))

    @classmethod
    def load_json(cls, path: Path | str) -> PerMarketCalibrator:
        p = Path(path)
        d = json.loads(p.read_text())
        if d.get("type") != "per_market":
            raise ValueError(
                f"Expected calibrator type='per_market', got {d.get('type')!r}"
            )
        fb = d["fallback"]
        fallback = IsotonicProbabilityCalibrator(
            fb["x_thresholds"], fb["y_thresholds"],
            n_train=int(fb.get("n_train", 0)),
            ece_before=float(fb.get("ece_before", 0.0)),
            ece_after=float(fb.get("ece_after", 0.0)),
        )
        per_market = {
            mkt: IsotonicProbabilityCalibrator(
                spec["x_thresholds"], spec["y_thresholds"],
                n_train=int(spec.get("n_train", 0)),
                ece_before=float(spec.get("ece_before", 0.0)),
                ece_after=float(spec.get("ece_after", 0.0)),
            )
            for mkt, spec in d.get("per_market", {}).items()
        }
        return cls(
            fallback=fallback,
            per_market=per_market,
            min_samples_per_market=int(d.get("min_samples_per_market", 25)),
            fitted_at=d.get("fitted_at"),
            n_train_total=int(d.get("n_train_total", 0)),
            markets_with_own_fit=list(d.get("markets_with_own_fit", [])),
            markets_using_fallback=list(d.get("markets_using_fallback", [])),
        )

    def __repr__(self) -> str:
        return (
            f"PerMarketCalibrator(n={self.n_train_total}, "
            f"per_market_fits={len(self.per_market)}, "
            f"fallback={len(self.markets_using_fallback)} markets)"
        )


def _ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — diagnostic only."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    if n_total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_pred = probs[mask].mean()
        bin_emp = outcomes[mask].mean()
        ece += (mask.sum() / n_total) * abs(bin_pred - bin_emp)
    return float(ece)
