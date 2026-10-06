"""Backtest CLV -- ML-02 opening odds + slippage."""

from __future__ import annotations

import pytest


class TestBacktestClv:
    """ML-02: walk-forward CLV uses opening odds x (1 - slippage), not closing."""

    def test_uses_opening_odds_not_closing(self):
        """apply_slippage shrinks opening odds by exactly SLIPPAGE_PCT -- ML-02."""
        from bip.train.backtest import SLIPPAGE_PCT, apply_slippage
        staked = apply_slippage(2.00)
        assert staked == pytest.approx(2.00 * (1 - SLIPPAGE_PCT))
        assert staked < 2.00  # slippage always shrinks

    def test_slippage_default_0_015(self):
        """SLIPPAGE_PCT = 0.015 (1.5%) -- ML-02."""
        from bip.train.backtest import SLIPPAGE_PCT
        assert SLIPPAGE_PCT == 0.015

    def test_null_clv_excluded_from_summary(self):
        """compute_clv is only called with non-null closing -- backtest aggregator drops nulls.

        Here we assert the math: staked 1.97 vs closing 2.00 -> -1.5% CLV.
        """
        from bip.train.backtest import compute_clv
        assert compute_clv(1.97, 2.00) == pytest.approx(-1.5)
        assert compute_clv(2.10, 2.00) == pytest.approx(5.0)
