"""Stake rotation policy unit tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bip.evaluation.live.engine_v3.phase3 import StakeRotationPolicy
from bip.evaluation.live.engine_v3.thesis import MarketFamily


def test_new_market_first_week_is_ramped():
    """A market never seen before is ramped at 1/3 in week 1."""
    pol = StakeRotationPolicy(new_market_ramp_weeks=3)
    decision = pol.evaluate(MarketFamily.CARDS, stake=1.0)
    assert decision.allowed is True
    assert decision.multiplier < 1.0
    assert "ramp" in decision.reason


def test_concentration_cap_denies_excess():
    """Once a single family exceeds 30% it is denied."""
    pol = StakeRotationPolicy(max_share_pct=0.30, new_market_ramp_weeks=1)
    now = datetime.now(timezone.utc)
    # Establish multi-family history so the cap is meaningful.
    pol.record_stake(MarketFamily.GOALS, 5.0, timestamp=now - timedelta(hours=1))
    pol.record_stake(MarketFamily.GOALS, 5.0, timestamp=now)
    pol.record_stake(MarketFamily.CARDS, 1.0, timestamp=now)
    # Goals share already 10/11 = 91% → ALL new goals stakes should be denied.
    decision = pol.evaluate(MarketFamily.GOALS, stake=1.0, now=now)
    assert decision.allowed is False
    assert decision.multiplier == 0.0
    assert "share" in decision.reason


def test_mature_market_full_stake():
    """A market introduced > 3 weeks ago and well below the cap → full stake."""
    pol = StakeRotationPolicy(new_market_ramp_weeks=3, max_share_pct=0.50)
    long_ago = datetime.now(timezone.utc) - timedelta(weeks=10)
    pol.record_stake(MarketFamily.GOALS, 1.0, timestamp=long_ago)
    pol.record_stake(MarketFamily.CORNERS, 1.0, timestamp=long_ago)
    # Use a fresh stamp to evaluate after the ramp window.
    decision = pol.evaluate(MarketFamily.GOALS, stake=1.0)
    assert decision.allowed is True
    # The 10-week-old history has rolled off, so the goals family is new again.
    # However we keep _market_introduced_at indefinitely.
    assert decision.multiplier in (1.0,)


def test_rolling_volume_pruning_drops_old_entries():
    pol = StakeRotationPolicy(rolling_window_days=7)
    old = datetime.now(timezone.utc) - timedelta(days=14)
    fresh = datetime.now(timezone.utc)
    pol.record_stake(MarketFamily.CARDS, 1.0, timestamp=old)
    pol.record_stake(MarketFamily.CARDS, 1.0, timestamp=fresh)
    vol = pol.rolling_volume()
    assert vol[MarketFamily.CARDS] == 1.0


def test_share_of_family():
    pol = StakeRotationPolicy()
    pol.record_stake(MarketFamily.GOALS, 3.0)
    pol.record_stake(MarketFamily.CARDS, 1.0)
    assert abs(pol.share_of(MarketFamily.GOALS) - 0.75) < 1e-9
