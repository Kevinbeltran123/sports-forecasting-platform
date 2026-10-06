"""Layer-1 tests for the token-bucket bandwidth governor (§E)."""

from __future__ import annotations

import pytest

from bip.core.telegram.bandwidth import TokenBucket


class TestInit:
    def test_full_at_construction(self):
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        assert b.tokens == pytest.approx(5.0)

    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError):
            TokenBucket(capacity=0)

    def test_rejects_negative_refill(self):
        with pytest.raises(ValueError):
            TokenBucket(refill_seconds=-1.0)


class TestConsumeBoundary:
    def test_consume_below_capacity(self):
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        for _ in range(5):
            assert b.try_consume(1.0) is True
        # 6th consume — exact boundary of empty bucket
        assert b.try_consume(1.0) is False

    def test_consume_zero_is_free(self):
        # Long refill window so microsecond drift between reset and read
        # doesn't materially affect the assertion.
        b = TokenBucket(capacity=1, refill_seconds=600.0)
        b.reset_for_test(tokens=0.0)
        assert b.try_consume(0.0) is True
        # Tokens should still be ~0 (drift bounded by elapsed/600s).
        assert b.tokens < 0.01

    def test_consume_fractional_cost(self):
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        b.reset_for_test(tokens=1.5)
        assert b.try_consume(1.0) is True
        assert b.tokens == pytest.approx(0.5, abs=1e-3)
        assert b.try_consume(1.0) is False

    def test_consume_at_exact_balance(self):
        """Boundary: bucket with exactly 1.0 token must allow 1-cost consume."""
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        b.reset_for_test(tokens=1.0)
        assert b.try_consume(1.0) is True

    def test_consume_just_below_balance(self):
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        b.reset_for_test(tokens=0.999)
        assert b.try_consume(1.0) is False


class TestRefill:
    def test_refill_caps_at_capacity(self):
        import time
        b = TokenBucket(capacity=3, refill_seconds=0.001)
        b.reset_for_test(tokens=0.0)
        time.sleep(0.05)   # >> enough to overflow capacity
        assert b.tokens == pytest.approx(3.0)

    def test_seconds_until_token(self):
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        b.reset_for_test(tokens=0.0)
        # Need 1.0 token at refill rate 1/6s → 6.0 seconds
        wait = b.seconds_until_token(target=1.0)
        assert 5.5 < wait <= 6.0  # tolerate small monotonic drift

    def test_seconds_until_token_zero_when_ready(self):
        b = TokenBucket(capacity=5, refill_seconds=6.0)
        assert b.seconds_until_token(target=1.0) == 0.0
