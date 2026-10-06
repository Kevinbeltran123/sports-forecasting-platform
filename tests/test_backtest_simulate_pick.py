"""D-09 + D-10 — simulate_pick unit tests + EDGE_THRESHOLD_PCT constant."""
from __future__ import annotations

import numpy as np


class TestSimulatePick:
    """D-09 / D-10 — edge-filter selection shared with Phase 3 pick engine."""

    def test_edge_threshold_default_0_05(self):
        from bip.train.backtest import EDGE_THRESHOLD_PCT
        assert EDGE_THRESHOLD_PCT == 0.05

    def test_simulate_pick_returns_argmax_when_edge_clears(self):
        from bip.train.backtest import simulate_pick
        probs = np.array([0.55, 0.25, 0.20])
        odds = np.array([2.10, 3.40, 4.00])  # edge_home = 0.55*2.10 - 1 = 0.155
        assert simulate_pick(probs, odds) == 0

    def test_simulate_pick_returns_none_below_threshold(self):
        from bip.train.backtest import simulate_pick
        probs = np.array([0.34, 0.33, 0.33])
        odds = np.array([2.00, 3.00, 3.00])  # max edge ≈ -0.01
        assert simulate_pick(probs, odds) is None

    def test_simulate_pick_returns_none_when_any_odds_nan(self):
        """D-03: Betano coverage gap => skip this row from CLV."""
        from bip.train.backtest import simulate_pick
        probs = np.array([0.55, 0.25, 0.20])
        odds = np.array([2.10, np.nan, 4.00])
        assert simulate_pick(probs, odds) is None

    def test_simulate_pick_accepts_custom_threshold(self):
        from bip.train.backtest import simulate_pick
        probs = np.array([0.30, 0.33, 0.37])
        odds = np.array([3.0, 3.0, 3.0])  # edge_away = 0.37*3.0 - 1 = 0.11
        assert simulate_pick(probs, odds, threshold=0.01) == 2

    def test_simulate_pick_is_importable_from_bip_train_backtest(self):
        """Plan 02.1-10 imports these three symbols together."""
        from bip.train.backtest import (
            EDGE_THRESHOLD_PCT,
            SLIPPAGE_PCT,
            simulate_pick,
        )
        assert EDGE_THRESHOLD_PCT == 0.05
        assert SLIPPAGE_PCT == 0.015
        assert callable(simulate_pick)
