"""Walk-forward temporal integrity -- ML-02."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest


class TestWalkForwardSplitter:
    """ML-02: all training dates strictly < test dates per fold."""

    def test_no_future_dates_in_train(self, synthetic_training_data):
        """Every training date must be strictly < every test date per fold -- ML-02."""
        from bip.train.walkforward import WalkForwardSplitter
        X, _y, dates = synthetic_training_data
        splitter = WalkForwardSplitter(n_splits=5)
        n_folds = 0
        for _fold, train_idx, test_idx in splitter.split(X, dates):
            assert dates[train_idx].max() < dates[test_idx].min()
            n_folds += 1
        assert n_folds == 5

    def test_n_splits_produces_n_folds(self, synthetic_training_data):
        """WalkForwardSplitter(n_splits=3) yields 3 folds -- ML-02."""
        from bip.train.walkforward import WalkForwardSplitter
        X, _y, dates = synthetic_training_data
        splitter = WalkForwardSplitter(n_splits=3)
        folds = list(splitter.split(X, dates))
        assert len(folds) == 3

    def test_assertion_raises_on_unsorted_dates(self):
        """Splitter raises AssertionError when dates are not monotonically increasing."""
        from bip.train.walkforward import WalkForwardSplitter
        n = 20
        X = np.zeros((n, 3))
        # Intentionally scrambled -- the first half dates are AFTER the second half
        later = np.array([datetime(2024, 6, 1) + timedelta(days=i) for i in range(n // 2)])
        earlier = np.array([datetime(2024, 1, 1) + timedelta(days=i) for i in range(n // 2)])
        dates = np.concatenate([later, earlier])
        splitter = WalkForwardSplitter(n_splits=3)
        with pytest.raises(AssertionError, match="temporal leakage"):
            list(splitter.split(X, dates))
