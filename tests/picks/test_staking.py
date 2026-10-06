"""Tests for stake sizing and the market-concentration cap.

Implementation: src/bip/core/picks/staking.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestQuarterKelly:
    def test_quarter_kelly(self):
        from bip.core.picks.staking import quarter_kelly_units
        # PICK-02 math: edge=0.05, odds=2.10 → (0.05/1.10) * 0.25 ≈ 0.01136
        result = quarter_kelly_units(0.05, 2.10)
        assert abs(result - (0.05 / 1.10 * 0.25)) < 1e-9
        # Edge clamping: negative edge → 0
        assert quarter_kelly_units(-0.01, 2.0) == 0.0
        # Avoid div-by-zero on odds ≤ 1.0
        assert quarter_kelly_units(0.05, 1.0) == 0.0
        assert quarter_kelly_units(0.05, 0.5) == 0.0

    def test_round_half_unit(self):
        from bip.core.picks.staking import round_to_nearest_half_unit
        assert round_to_nearest_half_unit(1.7) == 1.5
        assert round_to_nearest_half_unit(1.8) == 2.0
        assert round_to_nearest_half_unit(0.0) == 0.0
        assert round_to_nearest_half_unit(0.24) == 0.0
        assert round_to_nearest_half_unit(0.26) == 0.5


class TestMarketCap:
    def test_market_cap_drop(self):
        from bip.core.picks.staking import exceeds_60pct_cap
        # 10 prior picks: 6 in 1X2, 4 in BTTS. Adding another 1X2 → 7/11 = 63.6% > 60%.
        repo = MagicMock()
        repo.get_window_picks.return_value = [
            *[{"market": "1X2"}] * 6,
            *[{"market": "BTTS"}] * 4,
        ]
        assert exceeds_60pct_cap(repo, market="1X2", sport="football") is True
        # Adding a BTTS instead → 5/11 = 45.5% (no cap)
        assert exceeds_60pct_cap(repo, market="BTTS", sport="football") is False

    def test_market_cap_dormant_under_5_picks(self):
        from bip.core.picks.staking import exceeds_60pct_cap
        repo = MagicMock()
        repo.get_window_picks.return_value = [{"market": "1X2"}] * 4
        # Sample too small → cap dormant
        assert exceeds_60pct_cap(repo, market="1X2", sport="football") is False
