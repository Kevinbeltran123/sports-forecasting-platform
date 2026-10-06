"""Tests for lock_emitter.py — Phase 6 final lock JSON artifact.

Layer-1: emit + verify lock JSONs from synthetic LockDecision + fixtures.
Layer-2 (post-WC2026 scoring): real lock JSONs verified against actual
tournament outcomes; out of scope here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from bip.evaluation.tournaments.backtest.calibration_report import (
    MARKET_1X2,
    MARKET_BTTS,
    CalibrationReport,
)
from bip.evaluation.tournaments.backtest.lock_emitter import (
    DEFAULT_LOCK_DIR,
    LOCK_SCHEMA_VERSION,
    FixtureLockedPredictions,
    LockJSON,
    _canonical_payload,
    emit_lock_json,
    verify_lock_json,
)
from bip.evaluation.tournaments.backtest.lock_gate import evaluate_lock

# ── shared fixtures ──────────────────────────────────────────────────────────


def _passing_report(predictor: str = "bv", market: str = MARKET_1X2) -> CalibrationReport:
    is_multi = market == MARKET_1X2
    class_names = ("home", "draw", "away") if is_multi else ("no", "yes")
    return CalibrationReport(
        predictor_name=predictor,
        market=market,
        class_names=class_names,
        n_samples=200,
        n_bins=20,
        classwise_ece=0.02,
        per_class_ece=(0.02,) * len(class_names),
        bin_fill_pct=0.85,
        bin_fill_passes=True,
        brier=0.18 * len(class_names) if is_multi else 0.18,
        log_loss_value=0.5,
    )


def _below_gate_report(predictor: str = "bv", market: str = MARKET_1X2) -> CalibrationReport:
    is_multi = market == MARKET_1X2
    class_names = ("home", "draw", "away") if is_multi else ("no", "yes")
    return CalibrationReport(
        predictor_name=predictor,
        market=market,
        class_names=class_names,
        n_samples=200,
        n_bins=20,
        classwise_ece=0.10,  # well above 5%
        per_class_ece=(0.10,) * len(class_names),
        bin_fill_pct=0.85,
        bin_fill_passes=True,
        brier=0.30 * len(class_names) if is_multi else 0.30,
        log_loss_value=0.8,
    )


def _passing_decision():
    return evaluate_lock(
        [_passing_report("bv", MARKET_1X2), _passing_report("bv", MARKET_BTTS)],
        held_out_tournaments=("copa_2024", "euro_2024"),
        n_fixtures_total=100,
        n_fixtures_with_predictions=100,
        git_sha="abc1234",
    )


def _below_gate_decision():
    return evaluate_lock(
        [_below_gate_report("bv", MARKET_1X2)],
        held_out_tournaments=("copa_2024",),
        n_fixtures_total=100,
        n_fixtures_with_predictions=100,
    )


def _fixture(match_id: str = "wc_001") -> FixtureLockedPredictions:
    return FixtureLockedPredictions(
        match_id=match_id,
        tournament_phase="group_a",
        kickoff_utc=datetime(2026, 6, 11, 20, 0, tzinfo=UTC),
        home_team_id=10,
        away_team_id=20,
        home_team_name="Mexico",
        away_team_name="USA",
        predictions={
            "bivariate_poisson": {
                "p_home_win": 0.42,
                "p_draw": 0.28,
                "p_away_win": 0.30,
                "p_btts": 0.55,
                "p_over_2_5": 0.52,
            },
        },
    )


_FIXED_LOCK_TIME = datetime(2026, 6, 8, 18, 0, tzinfo=UTC)


# ── FixtureLockedPredictions validation ─────────────────────────────────────


class TestFixtureValidation:
    def test_valid_predictions_accepted(self):
        f = _fixture()
        assert f.match_id == "wc_001"
        assert f.predictions["bivariate_poisson"]["p_home_win"] == 0.42

    def test_probability_below_zero_rejected(self):
        with pytest.raises(ValueError, match="out of"):
            FixtureLockedPredictions(
                match_id="x",
                tournament_phase="group_a",
                kickoff_utc=_FIXED_LOCK_TIME,
                home_team_id=1,
                away_team_id=2,
                home_team_name="A",
                away_team_name="B",
                predictions={"bv": {"p_home_win": -0.1}},
            )

    def test_probability_above_one_rejected(self):
        with pytest.raises(ValueError, match="out of"):
            FixtureLockedPredictions(
                match_id="x",
                tournament_phase="group_a",
                kickoff_utc=_FIXED_LOCK_TIME,
                home_team_id=1,
                away_team_id=2,
                home_team_name="A",
                away_team_name="B",
                predictions={"bv": {"p_home_win": 1.5}},
            )

    def test_extra_fields_forbidden(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FixtureLockedPredictions(
                match_id="x",
                tournament_phase="group_a",
                kickoff_utc=_FIXED_LOCK_TIME,
                home_team_id=1,
                away_team_id=2,
                home_team_name="A",
                away_team_name="B",
                predictions={},
                rogue_field="hack",  # type: ignore[call-arg]
            )


# ── emit_lock_json — happy path ──────────────────────────────────────────────


class TestEmitLockJson:
    def test_passing_decision_emits_with_pass_status(self):
        lock = emit_lock_json(
            _passing_decision(),
            [_fixture()],
            locked_at=_FIXED_LOCK_TIME,
        )
        assert lock.calibration_status == "pass"
        assert lock.tournament_slug == "world_cup_2026"
        assert lock.locked_at == _FIXED_LOCK_TIME
        assert lock.schema_version == LOCK_SCHEMA_VERSION
        assert len(lock.fixtures) == 1
        assert lock.content_hash != ""
        assert len(lock.content_hash) == 64  # sha256 hex length

    def test_git_sha_falls_back_to_decision(self):
        lock = emit_lock_json(
            _passing_decision(),
            [_fixture()],
            locked_at=_FIXED_LOCK_TIME,
        )
        assert lock.git_sha == "abc1234"  # from _passing_decision

    def test_explicit_git_sha_overrides_decision(self):
        lock = emit_lock_json(
            _passing_decision(),
            [_fixture()],
            git_sha="override9",
            locked_at=_FIXED_LOCK_TIME,
        )
        assert lock.git_sha == "override9"

    def test_default_locked_at_uses_utc_now(self):
        before = datetime.now(UTC)
        lock = emit_lock_json(_passing_decision(), [_fixture()])
        after = datetime.now(UTC)
        assert before <= lock.locked_at <= after

    def test_writes_to_disk_when_output_path_given(self, tmp_path):
        out = tmp_path / "lock.json"
        emit_lock_json(
            _passing_decision(),
            [_fixture()],
            output_path=out,
            locked_at=_FIXED_LOCK_TIME,
        )
        assert out.exists()
        assert out.read_text().startswith("{")

    def test_marginal_decision_passes_lock_emits_without_force(self):
        from bip.evaluation.tournaments.backtest.calibration_report import MARKET_BTTS

        marginal_report = CalibrationReport(
            predictor_name="bv",
            market=MARKET_BTTS,
            class_names=("no", "yes"),
            n_samples=200,
            n_bins=20,
            classwise_ece=0.052,  # marginal
            per_class_ece=(0.052, 0.052),
            bin_fill_pct=0.85,
            bin_fill_passes=True,
            brier=0.20,
            log_loss_value=0.5,
        )
        decision = evaluate_lock(
            [marginal_report],
            held_out_tournaments=("copa_2024",),
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "marginal"
        # marginal still emits without force
        lock = emit_lock_json(decision, [_fixture()], locked_at=_FIXED_LOCK_TIME)
        assert lock.calibration_status == "marginal"
        assert lock.forced_emit_reason is None

    def test_structural_only_passes_lock_emits_without_force(self):
        from bip.evaluation.tournaments.backtest.lock_gate import evaluate_lock

        decision = evaluate_lock(
            [],
            held_out_tournaments=("copa_2024",),
            n_fixtures_total=0,
            n_fixtures_with_predictions=0,
            structural_only=True,
        )
        assert decision.calibration_status == "structural-only"
        lock = emit_lock_json(decision, [_fixture()], locked_at=_FIXED_LOCK_TIME)
        assert lock.calibration_status == "structural-only"


# ── force / R-08 slip plan ───────────────────────────────────────────────────


class TestForceEmit:
    def test_below_gate_without_force_raises(self):
        with pytest.raises(ValueError, match="does not pass"):
            emit_lock_json(_below_gate_decision(), [_fixture()])

    def test_force_without_reason_raises(self):
        with pytest.raises(ValueError, match="forced_emit_reason"):
            emit_lock_json(
                _below_gate_decision(),
                [_fixture()],
                force=True,
            )

    def test_force_with_reason_succeeds_and_records_audit(self):
        lock = emit_lock_json(
            _below_gate_decision(),
            [_fixture()],
            force=True,
            forced_emit_reason="R-08 slip plan: gate missed by 0.3pp on Copa 2024 only",
            locked_at=_FIXED_LOCK_TIME,
        )
        assert lock.calibration_status == "below-gate"
        assert "R-08 slip plan" in (lock.forced_emit_reason or "")

    def test_force_no_op_when_decision_already_passes(self):
        """force=True with reason on a passing decision still works (just records reason)."""
        lock = emit_lock_json(
            _passing_decision(),
            [_fixture()],
            force=True,
            forced_emit_reason="documenting why we shipped",
            locked_at=_FIXED_LOCK_TIME,
        )
        # The decision passed, so force was unnecessary but not harmful;
        # the reason is preserved verbatim for traceability.
        assert lock.calibration_status == "pass"
        assert lock.forced_emit_reason == "documenting why we shipped"


# ── content hash + tamper detection ──────────────────────────────────────────


class TestContentHash:
    def test_hash_is_deterministic_for_same_inputs(self):
        lock_a = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=_FIXED_LOCK_TIME
        )
        lock_b = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=_FIXED_LOCK_TIME
        )
        assert lock_a.content_hash == lock_b.content_hash

    def test_hash_differs_when_fixtures_differ(self):
        lock_a = emit_lock_json(
            _passing_decision(), [_fixture("wc_001")], locked_at=_FIXED_LOCK_TIME
        )
        lock_b = emit_lock_json(
            _passing_decision(), [_fixture("wc_002")], locked_at=_FIXED_LOCK_TIME
        )
        assert lock_a.content_hash != lock_b.content_hash

    def test_hash_differs_when_locked_at_differs(self):
        lock_a = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=_FIXED_LOCK_TIME
        )
        other_time = datetime(2026, 6, 7, 18, 0, tzinfo=UTC)
        lock_b = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=other_time
        )
        assert lock_a.content_hash != lock_b.content_hash

    def test_canonical_payload_excludes_content_hash(self):
        lock = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=_FIXED_LOCK_TIME
        )
        payload = _canonical_payload(lock.model_dump(mode="json"))
        assert "content_hash" not in payload

    def test_canonical_payload_keys_are_sorted(self):
        lock = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=_FIXED_LOCK_TIME
        )
        payload = _canonical_payload(lock.model_dump(mode="json"))
        # If keys are sorted, schema_version comes alphabetically before
        # tournament_slug (and so on); just verify a known-sortable position.
        # (Both top-level keys present; sorting is the property tested.)
        loaded = json.loads(payload)
        assert list(loaded.keys()) == sorted(loaded.keys())


class TestVerifyLockJson:
    def test_round_trip_verifies(self, tmp_path):
        out = tmp_path / "lock.json"
        emit_lock_json(
            _passing_decision(),
            [_fixture()],
            output_path=out,
            locked_at=_FIXED_LOCK_TIME,
        )
        lock, is_valid = verify_lock_json(out)
        assert is_valid
        assert lock.calibration_status == "pass"

    def test_tampered_fixture_detected(self, tmp_path):
        out = tmp_path / "lock.json"
        emit_lock_json(
            _passing_decision(),
            [_fixture()],
            output_path=out,
            locked_at=_FIXED_LOCK_TIME,
        )
        # Edit a probability post-lock
        data = json.loads(out.read_text())
        data["fixtures"][0]["predictions"]["bivariate_poisson"]["p_home_win"] = 0.99
        out.write_text(json.dumps(data, indent=2))

        _, is_valid = verify_lock_json(out)
        assert not is_valid

    def test_tampered_calibration_status_detected(self, tmp_path):
        out = tmp_path / "lock.json"
        emit_lock_json(
            _below_gate_decision(),
            [_fixture()],
            output_path=out,
            force=True,
            forced_emit_reason="R-08 slip plan",
            locked_at=_FIXED_LOCK_TIME,
        )
        data = json.loads(out.read_text())
        data["calibration_status"] = "pass"  # whitewash the verdict
        out.write_text(json.dumps(data, indent=2))

        _, is_valid = verify_lock_json(out)
        assert not is_valid

    def test_tampered_team_name_detected(self, tmp_path):
        out = tmp_path / "lock.json"
        emit_lock_json(
            _passing_decision(),
            [_fixture()],
            output_path=out,
            locked_at=_FIXED_LOCK_TIME,
        )
        data = json.loads(out.read_text())
        data["fixtures"][0]["home_team_name"] = "Argentina"  # arbitrary edit
        out.write_text(json.dumps(data, indent=2))

        _, is_valid = verify_lock_json(out)
        assert not is_valid

    def test_unmodified_file_with_whitespace_diff_still_verifies(self, tmp_path):
        """Re-serializing without changing content should preserve the hash."""
        out = tmp_path / "lock.json"
        emit_lock_json(
            _passing_decision(),
            [_fixture()],
            output_path=out,
            locked_at=_FIXED_LOCK_TIME,
        )
        # Re-write with different indentation but same data
        data = json.loads(out.read_text())
        out.write_text(json.dumps(data, indent=4))
        _, is_valid = verify_lock_json(out)
        # Hash is over canonical form (sort_keys + no whitespace), so
        # cosmetic re-indenting should still verify.
        assert is_valid


# ── coverage_pct + LockJSON shape ────────────────────────────────────────────


class TestLockJsonShape:
    def test_coverage_pct_derived(self):
        lock = emit_lock_json(
            _passing_decision(), [_fixture()], locked_at=_FIXED_LOCK_TIME
        )
        assert lock.coverage_pct == 1.0

    def test_zero_total_fixtures_yields_zero_coverage(self):
        from bip.evaluation.tournaments.backtest.lock_gate import evaluate_lock

        decision = evaluate_lock(
            [],
            held_out_tournaments=("copa_2024",),
            n_fixtures_total=0,
            n_fixtures_with_predictions=0,
            structural_only=True,
        )
        lock = emit_lock_json(decision, [_fixture()], locked_at=_FIXED_LOCK_TIME)
        assert lock.coverage_pct == 0.0

    def test_extra_top_level_field_rejected(self):
        from pydantic import ValidationError

        # Cannot add arbitrary fields — model is frozen with extra='forbid'
        with pytest.raises(ValidationError):
            LockJSON(
                schema_version=LOCK_SCHEMA_VERSION,
                tournament_slug="world_cup_2026",
                locked_at=_FIXED_LOCK_TIME,
                calibration_status="pass",
                held_out_tournaments=("copa_2024",),
                n_fixtures_total=1,
                n_fixtures_with_predictions=1,
                coverage_threshold=0.9,
                predictor_verdicts=(),
                operator_overrides=(),
                fixtures=(),
                content_hash="x",
                rogue="bad",  # type: ignore[call-arg]
            )

    def test_default_lock_dir_is_inside_package(self):
        """Sanity: the default destination matches spike doc §10."""
        s = str(DEFAULT_LOCK_DIR)
        assert "locked_predictions" in s
        assert "world_cup_2026" in s


# ── schema versioning anchor ────────────────────────────────────────────────


def test_schema_version_anchor():
    """Bump LOCK_SCHEMA_VERSION on breaking changes. This test prevents
    accidental silent changes — update both the constant AND this test."""
    assert LOCK_SCHEMA_VERSION == "1.0"
