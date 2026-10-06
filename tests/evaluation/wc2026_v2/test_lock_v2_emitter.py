"""Ola 7 — lock_v2 emitter integrity tests.

Verifies the FAIL-verdict lock_v2.json:
- Same 72 fixtures as lock_v1 (no fixture drift)
- calibration_status == 'below-gate'
- SHA-256 tamper-verifies
- Operator overrides + forced_emit_reason record the ablation finding
- Probabilities valid (in [0, 1] and 1X2 sums to 1)

These tests construct an in-memory lock_v2 via a tiny fixture predictor;
they do NOT depend on the on-disk lock_v2.json (so they pass even before
emit_lock_v2 has been run on a fresh clone).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from bip.evaluation.tournaments.backtest.lock_emitter import verify_lock_json
from bip.evaluation.tournaments.wc2026_v2.dibp import DIBPParams
from bip.evaluation.tournaments.wc2026_v2.lock_v2_emitter import (
    BACKTEST_METRICS,
    FORCED_EMIT_REASON,
    OPERATOR_OVERRIDES,
    PREDICTOR_NAME,
    _build_fixture_predictions,
    _build_lock_decision,
    _load_lock_v1_fixtures,
    emit_lock_v2,
)
from bip.evaluation.tournaments.wc2026_v2.v2_predictor import CalibratedDIBPPredictor
from bip.evaluation.tournaments.wc2026_v2.weighted_strength import (
    TeamStrength,
    WeightedMLEResult,
)


def _stub_predictor() -> CalibratedDIBPPredictor:
    """A predictor whose strengths cover every team referenced in lock_v1."""
    fixtures = _load_lock_v1_fixtures()
    all_names = {fx["home_team_name"] for fx in fixtures} | {
        fx["away_team_name"] for fx in fixtures
    }
    strengths = {name: TeamStrength(team=name, attack=0.0, defense=0.0) for name in all_names}
    result = WeightedMLEResult(
        strengths=strengths,
        home_advantage=0.20,
        intercept=0.10,
        reference_date=date(2026, 5, 23),
        n_train_matches=10_000,
        n_teams=len(strengths),
    )
    return CalibratedDIBPPredictor(strengths=result, dibp_params=DIBPParams(0.0, 0.0, 0.5))


class TestLockDecision:
    def test_decision_has_below_gate_status(self) -> None:
        d = _build_lock_decision()
        assert d.calibration_status == "below-gate"

    def test_holds_out_four_tournaments(self) -> None:
        d = _build_lock_decision()
        assert set(d.held_out_tournaments) == {
            "wc_2022",
            "afcon_2023",
            "copa_2024",
            "euro_2024",
        }

    def test_operator_overrides_record_ablation(self) -> None:
        d = _build_lock_decision()
        joined = " ".join(d.operator_overrides)
        assert "ablation" in joined.lower()
        assert "DIBP" in joined
        assert "beta calibration" in joined.lower()
        assert "match-importance" in joined.lower()

    def test_metrics_above_gate_thresholds(self) -> None:
        """Sanity-check that the lock_v2 metrics genuinely don't pass the gate."""
        assert BACKTEST_METRICS["1x2"]["brier_for_gate"] > 0.21
        assert BACKTEST_METRICS["btts"]["brier_for_gate"] > 0.20
        assert BACKTEST_METRICS["1x2"]["ece"] > 0.05


class TestFixturePredictions:
    def test_one_record_per_lock_v1_fixture(self) -> None:
        v1_fixtures = _load_lock_v1_fixtures()
        records = _build_fixture_predictions(_stub_predictor(), v1_fixtures)
        assert len(records) == len(v1_fixtures) == 72

    def test_predictor_name_carried(self) -> None:
        v1_fixtures = _load_lock_v1_fixtures()
        records = _build_fixture_predictions(_stub_predictor(), v1_fixtures)
        for rec in records:
            assert PREDICTOR_NAME in rec.predictions
            markets = rec.predictions[PREDICTOR_NAME]
            for key in ("p_home_win", "p_draw", "p_away_win", "p_btts", "p_over_2_5"):
                assert key in markets
                assert 0.0 <= markets[key] <= 1.0

    def test_1x2_sums_to_one_per_fixture(self) -> None:
        v1_fixtures = _load_lock_v1_fixtures()
        records = _build_fixture_predictions(_stub_predictor(), v1_fixtures)
        for rec in records:
            m = rec.predictions[PREDICTOR_NAME]
            assert m["p_home_win"] + m["p_draw"] + m["p_away_win"] == pytest.approx(1.0, abs=1e-6)


class TestEmitterIntegrity:
    """End-to-end: emit a fresh lock_v2 to a temp path, then verify integrity."""

    def test_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "lock_v2_test.json"
        out = emit_lock_v2(
            _stub_predictor(),
            git_sha="0000000",
            output_path=target,
            locked_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
        )
        assert out == target
        assert target.exists()

        loaded, is_valid = verify_lock_json(target)
        assert is_valid is True
        assert loaded.calibration_status == "below-gate"
        assert loaded.tournament_slug == "world_cup_2026"
        assert len(loaded.fixtures) == 72
        # forced_emit_reason was recorded and includes the FAIL narrative.
        assert loaded.forced_emit_reason is not None
        assert "FAIL" in loaded.forced_emit_reason

    def test_does_not_overwrite_lock_v1(self, tmp_path: Path) -> None:
        """Sanity: writing lock_v2 to its default path doesn't clobber lock_v1.

        We don't actually call ``emit_lock_v2(predictor)`` here (would write
        to the real repo path) — but we verify the default path is distinct
        from lock_v1's by comparing the two module-level constants.
        """
        from bip.evaluation.tournaments.wc2026_v2.lock_v2_emitter import (
            _LOCK_V1_PATH,
            LOCK_V2_PATH,
        )

        assert LOCK_V2_PATH != _LOCK_V1_PATH
        # Both live in the same directory but with different filenames.
        assert LOCK_V2_PATH.parent == _LOCK_V1_PATH.parent
        assert LOCK_V2_PATH.name == "lock_v2.json"
        assert _LOCK_V1_PATH.name == "lock.json"


class TestForcedEmitReason:
    def test_reason_documents_fail_verdict(self) -> None:
        assert "FAIL verdict" in FORCED_EMIT_REASON
        assert "lock_v1" in FORCED_EMIT_REASON
        assert "audit trail" in FORCED_EMIT_REASON

    def test_overrides_are_non_empty_strings(self) -> None:
        for override in OPERATOR_OVERRIDES:
            assert isinstance(override, str)
            assert len(override) > 30  # not stubs
