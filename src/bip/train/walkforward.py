"""Temporal walk-forward CV -- ML-02.

Wraps sklearn.model_selection.TimeSeriesSplit with a MANDATORY assertion
that every training date is strictly < every test date per fold.
Violating the assertion means temporal leakage (RESEARCH.md Pitfall 1).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import structlog
from sklearn.model_selection import TimeSeriesSplit

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardSplitter:
    """Expanding-window temporal splitter with leakage assertion."""

    n_splits: int = 5

    def split(
        self, X: np.ndarray, dates: np.ndarray,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield (fold_idx, train_idx, test_idx).

        Raises AssertionError when a fold's max training date is not
        strictly less than its min test date.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            assert dates[train_idx].max() < dates[test_idx].min(), (
                f"Fold {fold_idx} temporal leakage: "
                f"train max={dates[train_idx].max()} >= test min={dates[test_idx].min()}"
            )
            logger.info(
                "fold_split",
                fold=fold_idx,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
            yield fold_idx, train_idx, test_idx
