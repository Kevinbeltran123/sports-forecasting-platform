"""Ola 6 — ablation harness tests.

Marked ``slow`` because each ablation run fits 5 × 4 = 20 v2 pipelines.
Module-scope fixture so the heavy run executes exactly once and the
assertions multiplex on the shared result.
"""

from __future__ import annotations

import pytest

from bip.evaluation.tournaments.wc2026_v2.ablation import (
    ABLATION_CONFIGS,
    AblationCell,
    format_ablation_table,
    run_ablation_study,
)
from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cells() -> dict[str, AblationCell]:
    split = build_split()
    return run_ablation_study(
        train=split.train,
        calibration=split.calibration,
        heldout=split.heldout,
        n_bootstrap=200,
        seed=42,
    )


@pytest.mark.requires_data
class TestAblationStudy:
    def test_runs_all_configs(self, cells: dict[str, AblationCell]) -> None:
        names = {cfg.name for cfg in ABLATION_CONFIGS}
        assert set(cells.keys()) == names
        for cell in cells.values():
            assert isinstance(cell, AblationCell)
            assert 0.15 < cell.overall_brier_1x2.point < 0.30

    def test_full_row_has_no_delta(self, cells: dict[str, AblationCell]) -> None:
        assert cells["full"].delta_brier_1x2 is None
        for name, cell in cells.items():
            if name == "full":
                continue
            assert cell.delta_brier_1x2 is not None

    def test_format_table_renders(self, cells: dict[str, AblationCell]) -> None:
        out = format_ablation_table(cells)
        assert "WC2026 v2 Ablation" in out
        assert "full" in out
        assert "afcon_2023" in out
        assert "copa_2024" in out
        assert "wc_2022" in out
        assert "euro_2024" in out

    def test_paired_delta_close_to_zero_for_collapsed_config(
        self, cells: dict[str, AblationCell]
    ) -> None:
        """no_dibp + no_beta_calibration should be statistically identical to
        the baseline because the DIBP fit on this corpus returns π=0 anyway
        (the regime where DIBP collapses to plain BP)."""
        d = cells["no_dibp"].delta_brier_1x2
        assert d is not None
        assert abs(d.point) < 0.01
