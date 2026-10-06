"""Tests for backtest metrics."""
from __future__ import annotations

import numpy as np
import pytest

from bip.evaluation.tournaments.team_style_profiler.backtest.metrics import (
    MARKET_BRIER_GATES,
    BacktestMetrics,
    MarketMetrics,
    brier_score,
    expected_calibration_error,
    hit_rate,
    market_passes_brier_gate,
)


class TestBrierScore:
    def test_perfect_prediction(self) -> None:
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        out = np.array([1, 0, 1, 0])
        assert brier_score(probs, out) == 0.0

    def test_completely_wrong(self) -> None:
        probs = np.array([1.0, 0.0])
        out = np.array([0, 1])
        assert brier_score(probs, out) == 1.0

    def test_uninformed(self) -> None:
        # Always predicting 0.5 -> Brier = 0.25
        probs = np.array([0.5] * 100)
        out = np.array([0, 1] * 50)
        assert brier_score(probs, out) == pytest.approx(0.25)

    def test_empty(self) -> None:
        import math
        assert math.isnan(brier_score(np.array([]), np.array([])))


class TestECE:
    def test_perfect_calibration(self) -> None:
        # All predictions at 0.5 with 50% positive rate
        probs = np.array([0.5] * 100)
        out = np.array([0, 1] * 50)
        ece = expected_calibration_error(probs, out, n_bins=10)
        assert ece == 0.0

    def test_overconfident(self) -> None:
        # Predict 0.9 but only 0.5 positive -> ECE = 0.4
        probs = np.array([0.9] * 100)
        out = np.array([1] * 50 + [0] * 50)
        ece = expected_calibration_error(probs, out, n_bins=10)
        assert ece == pytest.approx(0.4)


class TestHitRate:
    def test_basic(self) -> None:
        # probs >= 0.5 -> pred=1
        probs = np.array([0.7, 0.3, 0.8, 0.4])
        out = np.array([1, 0, 0, 1])
        # preds = [1, 0, 1, 0], outcomes = [1, 0, 0, 1] -> 2/4 = 0.5
        assert hit_rate(probs, out) == 0.5


class TestMarketGate:
    def test_btts_gate(self) -> None:
        assert market_passes_brier_gate("BTTS_yes", 0.20) is True
        assert market_passes_brier_gate("BTTS_yes", 0.25) is False

    def test_o25_gate(self) -> None:
        assert market_passes_brier_gate("O2.5", 0.22) is True
        assert market_passes_brier_gate("O2.5", 0.231) is False

    def test_unknown_market_fails(self) -> None:
        assert market_passes_brier_gate("UNKNOWN_MARKET", 0.01) is False

    def test_canonical_gates(self) -> None:
        # Operator-locked gates
        assert MARKET_BRIER_GATES["BTTS_yes"] == 0.235
        assert MARKET_BRIER_GATES["O2.5"] == 0.230


class TestBacktestMetrics:
    def test_summary_lines(self) -> None:
        mm = MarketMetrics(
            market="BTTS_yes", n=10, brier=0.20, ece=0.05,
            hit_rate=0.65, mean_predicted=0.55, mean_outcome=0.60,
            passes_brier_gate=True,
        )
        bm = BacktestMetrics(
            n_fixtures=12, n_with_predictions=10, coverage_pct=83.3,
            market_metrics={"BTTS_yes": mm},
        )
        out = bm.summary_lines()
        assert any("BTTS_yes" in line for line in out)
        assert any("83.3" in line for line in out)
        assert bm.all_markets_pass is True

    def test_all_markets_pass_false_when_any_fails(self) -> None:
        ok = MarketMetrics(
            market="A", n=10, brier=0.1, ece=0.05, hit_rate=0.7,
            mean_predicted=0.5, mean_outcome=0.5, passes_brier_gate=True,
        )
        bad = MarketMetrics(
            market="B", n=10, brier=0.4, ece=0.2, hit_rate=0.4,
            mean_predicted=0.5, mean_outcome=0.5, passes_brier_gate=False,
        )
        bm = BacktestMetrics(
            n_fixtures=10, n_with_predictions=10, coverage_pct=100.0,
            market_metrics={"A": ok, "B": bad},
        )
        assert bm.all_markets_pass is False
