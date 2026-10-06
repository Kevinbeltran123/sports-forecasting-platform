"""Isotonic-regression calibrator per market.

OPERATOR DECISION (locked 2026-05-24): use isotonic regression for
per-market calibration when backtest reveals systematic mis-calibration.

Isotonic regression learns a monotonic mapping from predicted_prob ->
calibrated_prob using held-out validation pairs. Non-parametric — doesn't
assume sigmoid form (Platt) or linear scaling (temperature).

Usage:
    cal = IsotonicCalibrator()
    cal.fit(market="BTTS_yes", predictions=[0.6, 0.7, 0.65], outcomes=[1, 0, 1])
    cal.calibrate(market="BTTS_yes", prob=0.68) -> 0.72 (or whatever the
                                                          monotone fit gives)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class IsotonicCalibrator:
    """Per-market isotonic calibrators."""

    by_market: dict[str, IsotonicRegression] = field(default_factory=dict)
    fitted_markets: set[str] = field(default_factory=set)

    def fit(
        self, market: str, predictions: np.ndarray, outcomes: np.ndarray
    ) -> None:
        """Fit the calibrator for a single market.

        Args:
            market: market name (e.g. 'BTTS_yes').
            predictions: 1D array of predicted probabilities in [0, 1].
            outcomes: 1D array of 0/1 ground truth.

        Requires len(predictions) == len(outcomes) >= 10 for a defensible
        fit. Below that threshold, raises ValueError.
        """
        p = np.asarray(predictions, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if p.size != y.size:
            raise ValueError("predictions and outcomes must have same length")
        if p.size < 10:
            raise ValueError(
                f"Need >=10 samples to fit calibrator for {market}; got {p.size}"
            )

        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p, y)
        self.by_market[market] = ir
        self.fitted_markets.add(market)

    def calibrate(self, market: str, prob: float) -> float:
        """Map a raw predicted probability through the fitted calibrator.

        Returns the input unchanged when no calibrator has been fit for
        this market.
        """
        if market not in self.by_market:
            return prob
        ir = self.by_market[market]
        return float(ir.predict([prob])[0])

    def calibrate_batch(self, market: str, probs: np.ndarray) -> np.ndarray:
        """Calibrate an array of probs for a market."""
        if market not in self.by_market:
            return probs
        ir = self.by_market[market]
        return ir.predict(probs)

    def is_fit(self, market: str) -> bool:
        return market in self.fitted_markets
