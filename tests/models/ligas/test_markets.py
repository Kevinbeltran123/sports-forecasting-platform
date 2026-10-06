"""Market1x2 + MarketOverUnder wrapper tests on synthetic data.

Sprint 1 Ola A.
"""

from __future__ import annotations

import numpy as np
import pytest

from bip.models.ligas.markets import Market1x2, MarketOverUnder


@pytest.fixture(scope="module")
def synthetic_3class():
    rng = np.random.default_rng(seed=7)
    n, k = 300, 6
    X = rng.normal(size=(n, k))
    # Signal: y depends on the first 3 features
    scores = X[:, :3] + rng.normal(size=(n, 3)) * 0.4
    y = scores.argmax(axis=1).astype(int)
    return X, y


@pytest.fixture(scope="module")
def synthetic_binary():
    rng = np.random.default_rng(seed=11)
    n, k = 250, 5
    X = rng.normal(size=(n, k))
    y = (X[:, 0] + rng.normal(size=n) * 0.4 > 0).astype(int)
    return X, y


class TestMarket1x2:
    def test_fit_predict_proba_shapes(self, synthetic_3class):
        X, y = synthetic_3class
        m = Market1x2().fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 3)
        # rows sum ~1.0
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-2)

    def test_predict_for_fixture_returns_named_selections(self, synthetic_3class):
        X, y = synthetic_3class
        m = Market1x2().fit(X, y)
        out = m.predict_for_fixture(X[0])
        assert set(out.keys()) == {"home", "draw", "away"}
        assert all(0.0 <= v <= 1.0 for v in out.values())
        assert abs(sum(out.values()) - 1.0) < 1e-2

    def test_predict_proba_before_fit_raises(self):
        m = Market1x2()
        with pytest.raises(RuntimeError):
            m.predict_proba(np.zeros((1, 3)))

    def test_shape_mismatch_raises(self):
        m = Market1x2()
        with pytest.raises(ValueError):
            m.fit(np.zeros((5, 3)), np.zeros(4))


class TestMarketOverUnder:
    def test_fit_predict_proba_shapes(self, synthetic_binary):
        X, y = synthetic_binary
        m = MarketOverUnder(line=2.5).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 2)
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-2)

    def test_predict_for_fixture_returns_named_selections(self, synthetic_binary):
        X, y = synthetic_binary
        m = MarketOverUnder(line=2.5).fit(X, y)
        out = m.predict_for_fixture(X[0])
        assert set(out.keys()) == {"over", "under"}
        assert abs(sum(out.values()) - 1.0) < 1e-2

    def test_market_id_includes_line(self):
        assert MarketOverUnder(line=2.5).market_id == "ou_2.5"
        assert MarketOverUnder(line=1.5).market_id == "ou_1.5"
        assert MarketOverUnder(line=3.5).market_id == "ou_3.5"

    def test_implausible_line_rejected(self):
        with pytest.raises(ValueError):
            MarketOverUnder(line=0.0)
        with pytest.raises(ValueError):
            MarketOverUnder(line=20.0)

    def test_rejects_non_binary_y(self):
        m = MarketOverUnder(line=2.5)
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 3)), np.array([0, 1, 2] + [0] * 7))
