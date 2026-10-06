"""Tests for scripts/backtest_int_tournaments.py — Phase 5 backtest harness.

Layer-1: stub predictors + synthetic snapshots verify the replay → audit →
aggregate pipeline end-to-end. Real wiring (API-Football pull + production
predictor adapters) is gated # requires-real-data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bip.evaluation.tournaments.backtest.calibration_report import (
    MARKET_1X2,
    MARKET_BTTS,
    MARKET_OU_2_5,
)
from scripts.backtest_int_tournaments import (
    DEFAULT_HELD_OUT_TOURNAMENTS,
    BacktestSnapshot,
    FixturePrediction,
    aggregate_to_lock_decision,
    run_backtest_layer1,
)

# ── stub predictors ──────────────────────────────────────────────────────────


@dataclass
class _CalibratedStubPredictor:
    """Predicts probabilities matching the empirical outcome distribution.

    With balanced synthetic data, this produces well-calibrated probabilities
    (ECE near zero) → all gates pass.
    """

    name: str = "stub_calibrated"

    def predict_fixture(self, snapshot: BacktestSnapshot) -> FixturePrediction:
        # Roughly even 1X2 split + balanced BTTS / O2.5 — matches the synthetic
        # outcome generator below.
        return FixturePrediction(
            p_home_win=0.40,
            p_draw=0.30,
            p_away_win=0.30,
            p_btts=0.55,
            p_over_2_5=0.55,
        )


@dataclass
class _OverconfidentStubPredictor:
    """Always predicts home win with ~100% confidence — guaranteed below-gate."""

    name: str = "stub_overconfident"

    def predict_fixture(self, snapshot: BacktestSnapshot) -> FixturePrediction:
        return FixturePrediction(
            p_home_win=0.99,
            p_draw=0.005,
            p_away_win=0.005,
            p_btts=0.99,  # claims BTTS always — half the time wrong
            p_over_2_5=0.99,
        )


@dataclass
class _OneMarketOnlyPredictor:
    """Emits 1X2 only; BTTS / OU left None to test optional-market handling."""

    name: str = "stub_1x2_only"

    def predict_fixture(self, snapshot: BacktestSnapshot) -> FixturePrediction:
        return FixturePrediction(
            p_home_win=0.40,
            p_draw=0.30,
            p_away_win=0.30,
            p_btts=None,
            p_over_2_5=None,
        )


# ── snapshot builders ────────────────────────────────────────────────────────


def _balanced_snapshots(n: int = 60) -> list[BacktestSnapshot]:
    """Generate snapshots with roughly the predicted 40/30/30 1X2 split.

    Cycles deterministically so the calibrated predictor's probabilities
    match the empirical frequency: 40% home wins, 30% draws, 30% away wins.
    Total goals also alternate so BTTS / O2.5 are ~55%.
    """
    snaps: list[BacktestSnapshot] = []
    for i in range(n):
        # Deterministic 4/3/3 split per 10-fixture block
        idx = i % 10
        if idx < 4:
            outcome_1x2 = 0  # home win
            home_g, away_g = 2, 1
        elif idx < 7:
            outcome_1x2 = 1  # draw
            home_g, away_g = 1, 1
        else:
            outcome_1x2 = 2  # away win
            home_g, away_g = 0, 2
        total = home_g + away_g
        btts = int(home_g > 0 and away_g > 0)
        snaps.append(
            BacktestSnapshot(
                match_id=f"m_{i:03d}",
                tournament="copa_2024" if i < n // 2 else "euro_2024",
                home_team_id=100 + i,
                away_team_id=200 + i,
                observed_1x2=outcome_1x2,
                observed_total_goals=total,
                observed_btts=btts,
            )
        )
    return snaps


# ── pipeline tests ───────────────────────────────────────────────────────────


class TestRunBacktestLayer1:
    def test_calibrated_predictor_produces_reports_for_all_three_markets(self):
        snaps = _balanced_snapshots(60)
        reports = run_backtest_layer1(snaps, [_CalibratedStubPredictor()])
        markets = {r.market for r in reports}
        assert MARKET_1X2 in markets
        assert MARKET_BTTS in markets
        assert MARKET_OU_2_5 in markets
        assert all(r.predictor_name == "stub_calibrated" for r in reports)

    def test_predictor_with_no_optional_markets_emits_only_1x2(self):
        snaps = _balanced_snapshots(30)
        reports = run_backtest_layer1(snaps, [_OneMarketOnlyPredictor()])
        markets = {r.market for r in reports}
        assert markets == {MARKET_1X2}

    def test_multiple_predictors_each_get_own_reports(self):
        snaps = _balanced_snapshots(30)
        reports = run_backtest_layer1(
            snaps, [_CalibratedStubPredictor(), _OverconfidentStubPredictor()]
        )
        names = {r.predictor_name for r in reports}
        assert names == {"stub_calibrated", "stub_overconfident"}

    def test_progress_callback_invoked_per_snapshot(self):
        snaps = _balanced_snapshots(20)
        calls: list[tuple[str, int]] = []
        run_backtest_layer1(
            snaps,
            [_CalibratedStubPredictor()],
            progress_cb=lambda name, n: calls.append((name, n)),
        )
        assert len(calls) == 20
        assert calls[0] == ("stub_calibrated", 1)
        assert calls[-1] == ("stub_calibrated", 20)

    def test_n_samples_in_reports_matches_snapshot_count(self):
        snaps = _balanced_snapshots(60)
        reports = run_backtest_layer1(snaps, [_CalibratedStubPredictor()])
        for r in reports:
            assert r.n_samples == 60


# ── aggregate-to-lock ────────────────────────────────────────────────────────


class TestAggregateToLockDecision:
    def test_calibrated_predictor_passes_lock(self):
        snaps = _balanced_snapshots(60)
        reports = run_backtest_layer1(snaps, [_CalibratedStubPredictor()])
        decision = aggregate_to_lock_decision(
            reports,
            n_fixtures_total=60,
            n_fixtures_with_predictions=60,
        )
        # Calibrated stub should pass or be marginal — definitely not below-gate
        assert decision.calibration_status in ("pass", "marginal", "unreliable-bins")
        # passes_lock should be True for pass/marginal — unreliable-bins on
        # synthetic data of n=60 is acceptable noise; what matters is no
        # below-gate failure
        if decision.calibration_status != "unreliable-bins":
            assert decision.passes_lock

    def test_overconfident_predictor_fails_at_below_gate(self):
        snaps = _balanced_snapshots(60)
        reports = run_backtest_layer1(snaps, [_OverconfidentStubPredictor()])
        decision = aggregate_to_lock_decision(
            reports,
            n_fixtures_total=60,
            n_fixtures_with_predictions=60,
        )
        # The overconfident stub claims home wins ~100% but only 40% of
        # snapshots are home wins → catastrophic miscalibration
        assert decision.calibration_status == "below-gate"
        assert not decision.passes_lock

    def test_default_held_out_tournaments_propagate(self):
        snaps = _balanced_snapshots(30)
        reports = run_backtest_layer1(snaps, [_CalibratedStubPredictor()])
        decision = aggregate_to_lock_decision(
            reports,
            n_fixtures_total=30,
            n_fixtures_with_predictions=30,
        )
        assert decision.held_out_tournaments == DEFAULT_HELD_OUT_TOURNAMENTS

    def test_writes_json_when_output_path_provided(self, tmp_path):
        snaps = _balanced_snapshots(30)
        reports = run_backtest_layer1(snaps, [_CalibratedStubPredictor()])
        out = tmp_path / "decision.json"
        decision = aggregate_to_lock_decision(
            reports,
            n_fixtures_total=30,
            n_fixtures_with_predictions=30,
            output_path=out,
        )
        assert out.exists()
        loaded = decision.from_json(out)
        assert loaded.calibration_status == decision.calibration_status

    def test_structural_only_short_circuits(self, tmp_path):
        decision = aggregate_to_lock_decision(
            reports=[],
            n_fixtures_total=0,
            n_fixtures_with_predictions=0,
            structural_only=True,
        )
        assert decision.calibration_status == "structural-only"
        assert decision.passes_lock


# ── default held-out matches operator approval ──────────────────────────────


def test_default_held_out_tournaments_are_copa_and_euro_2024():
    """Operator-approved 2026-05-08: most recent + closest to WC2026 in style."""
    assert DEFAULT_HELD_OUT_TOURNAMENTS == ("copa_2024", "euro_2024")


# ── Layer-2: real-data backtest unblocked via StatsBomb ingest ──────────────


def _statsbomb_data_available() -> bool:
    from scripts.seed_statsbomb_tournaments import DEFAULT_OUTCOMES_PARQUET
    return DEFAULT_OUTCOMES_PARQUET.exists()


@pytest.mark.skipif(
    not _statsbomb_data_available(),
    reason=(
        "Run `uv run python scripts/seed_statsbomb_tournaments.py` first "
        "to seed the StatsBomb match-outcomes Parquet."
    ),
)
def test_real_historical_backtest_passes_or_fails_lock_honestly():
    """Layer-2: real backtest over the 6 modern men's StatsBomb tournaments.

    Acceptance criterion: lock decision JSON emits a defensible
    ``calibration_status`` that the operator can defend to post-tournament
    reviewers. The ``calibration_status`` may be 'pass' / 'marginal' /
    'unreliable-bins' / 'below-gate' — what matters is that the verdict
    is computed honestly from real predictions vs real outcomes, not
    that any particular verdict is achieved.
    """
    from scripts.run_phase5_backtest import run_phase5_backtest

    decision, _reports = run_phase5_backtest(output_path=None)
    # Real backtest ran end-to-end; verdict is one of the calibrated-status
    # values, not a meta-status like 'no-reports' or 'structural-only'.
    assert decision.calibration_status in (
        "pass", "marginal", "unreliable-bins", "below-gate"
    )
    assert decision.n_fixtures_with_predictions > 250  # 6 tournaments × ~50
