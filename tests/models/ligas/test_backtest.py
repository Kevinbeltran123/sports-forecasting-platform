"""Walk-forward backtest harness tests — Sprint 1 Ola C.

Synthetic data with a known signal to verify:
  - No temporal leakage (assertion would fire if violated)
  - Bootstrap CI + gate decision wires end-to-end
  - LigasModel.evaluate() delegates to run_walkforward and returns ModelMetrics
"""

from __future__ import annotations

import numpy as np
import pytest

from bip.models.base import ModelMetrics
from bip.models.ligas import LigasModel
from bip.models.ligas.backtest import run_walkforward
from bip.models.ligas.gate import GateVerdict


def _synthetic_payload(
    *,
    n: int = 240,
    n_features: int = 6,
    signal_strength: float = 0.8,
    seed: int = 42,
    add_odds: bool = False,
):
    """Generate a synthetic 1x2 dataset with a controllable signal.

    Dates are sequential daily timestamps so WalkForwardSplitter has
    something monotone to assert on.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # Signal: y depends on the first 3 features
    scores = X[:, :3] * signal_strength + rng.normal(size=(n, 3))
    y = scores.argmax(axis=1).astype(int)
    dates = np.arange(n).astype("datetime64[D]")
    payload = {"X": X, "y": y, "dates": dates}
    if add_odds:
        # Synthetic odds inversely correlated with true class (fair odds + noise)
        # Use 3.0 as base so EV gate is easy to clear when prediction is correct
        odds = np.full((n, 3), 3.0)
        payload["opening_odds"] = odds
    return payload


class TestRunWalkforward:
    def test_returns_metrics_and_decision(self):
        payload = _synthetic_payload(n=150, signal_strength=0.6)
        metrics, decision = run_walkforward(
            payload, n_splits=3, n_bootstrap=200
        )
        assert isinstance(metrics, ModelMetrics)
        assert decision.verdict in {GateVerdict.PASS, GateVerdict.SHADOW, GateVerdict.FAIL}
        assert metrics.brier_score is not None
        assert "verdict=" in metrics.notes

    def test_no_picks_when_no_odds_given(self):
        payload = _synthetic_payload(n=120)
        metrics, decision = run_walkforward(payload, n_splits=3, n_bootstrap=100)
        # No opening_odds → no picks
        assert metrics.n_picks == 0
        assert decision.roi_ci is None

    def test_picks_emitted_with_odds(self):
        payload = _synthetic_payload(n=200, signal_strength=1.0, add_odds=True)
        metrics, _ = run_walkforward(
            payload, n_splits=3, n_bootstrap=100, ev_threshold=0.0
        )
        # With odds=3.0 across the board, any prediction with p>1/3 wins
        # an EV>0 trial; threshold=0 lets them through.
        assert metrics.n_picks > 0
        assert metrics.hit_rate >= 0.0

    def test_payload_missing_keys_raises(self):
        with pytest.raises(ValueError, match="Missing payload keys"):
            run_walkforward({"X": np.zeros((5, 3))})

    def test_payload_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            run_walkforward(
                {
                    "X": np.zeros((5, 3)),
                    "y": np.zeros(4),
                    "dates": np.arange(5).astype("datetime64[D]"),
                }
            )

    def test_temporal_no_leakage_assertion_holds(self):
        # With monotone dates the WalkForwardSplitter assertion must pass;
        # the test just confirms the run completes without AssertionError.
        payload = _synthetic_payload(n=120)
        run_walkforward(payload, n_splits=3, n_bootstrap=50)


class TestLigasModelEvaluate:
    def test_delegates_to_run_walkforward(self):
        m = LigasModel()
        # n=300 gives enough rows per inner TimeSeriesSplit fold to see
        # all 3 classes; with n<200 the inner OOF can produce folds
        # missing class 1, which trips XGB's class-validation.
        payload = _synthetic_payload(n=300)
        metrics = m.evaluate(payload)
        assert isinstance(metrics, ModelMetrics)
        assert metrics.brier_score is not None
