"""Tamper-verify tests for lock_v3.json (Ola 3.F).

Locks in the audit-artifact invariants. Mirrors v2's lock_v2 integrity
test pattern but adds v3-specific checks (predictor_name, A+C verdict
strings in operator_overrides).

Skips if lock_v3.json doesn't exist yet — operator runs the emit
script once to produce it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


_LOCK_V1_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "bip"
    / "evaluation"
    / "tournaments"
    / "locked_predictions"
    / "world_cup_2026"
    / "lock.json"
)
_LOCK_V3_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "bip"
    / "evaluation"
    / "tournaments"
    / "locked_predictions"
    / "world_cup_2026"
    / "lock_v3.json"
)


@pytest.fixture(scope="module")
def lock_v3() -> dict:
    if not _LOCK_V3_PATH.exists():
        pytest.skip(
            f"lock_v3.json not present at {_LOCK_V3_PATH}. Run the "
            "(unpublished) lock-v3 emitter to produce it."
        )
    with _LOCK_V3_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def lock_v1() -> dict:
    with _LOCK_V1_PATH.open() as f:
        return json.load(f)


def test_lock_v3_calibration_status_below_gate(lock_v3) -> None:
    """A+C both FAIL → calibration_status must be 'below-gate'."""
    assert lock_v3["calibration_status"] == "below-gate"


def test_lock_v3_predictor_name_v3(lock_v3) -> None:
    """Verifies the v3 predictor name appears in fixture predictions."""
    fixtures = lock_v3["fixtures"]
    assert fixtures, "lock_v3 has no fixtures"
    pred = fixtures[0]["predictions"]
    assert (
        "wc2026_v3_calibrated_dibp_no_mv_no_hierarchy" in pred
    ), f"expected v3 predictor key, got {list(pred.keys())}"


def test_lock_v3_same_72_fixtures_as_v1(lock_v3, lock_v1) -> None:
    """Schema invariant: v3 must predict the SAME 72 group-stage fixtures
    as v1 (set equality on (home, away, kickoff)). v3 != v1 only in
    predictor / metadata."""
    v1_keys = {
        (fx["home_team_name"], fx["away_team_name"], fx["kickoff_utc"])
        for fx in lock_v1["fixtures"]
    }
    v3_keys = {
        (fx["home_team_name"], fx["away_team_name"], fx["kickoff_utc"])
        for fx in lock_v3["fixtures"]
    }
    assert v3_keys == v1_keys
    assert len(v3_keys) == 72


def test_lock_v3_predictions_sum_to_one(lock_v3) -> None:
    """For each fixture, P(home) + P(draw) + P(away) ≈ 1.0."""
    pred_key = "wc2026_v3_calibrated_dibp_no_mv_no_hierarchy"
    for fx in lock_v3["fixtures"]:
        p = fx["predictions"][pred_key]
        s = p["p_home_win"] + p["p_draw"] + p["p_away_win"]
        assert s == pytest.approx(1.0, abs=1e-6), (
            f"{fx['home_team_name']} vs {fx['away_team_name']}: "
            f"1X2 sum={s}, not 1.0"
        )


def test_lock_v3_btts_and_ou_in_unit_interval(lock_v3) -> None:
    pred_key = "wc2026_v3_calibrated_dibp_no_mv_no_hierarchy"
    for fx in lock_v3["fixtures"]:
        p = fx["predictions"][pred_key]
        assert 0.0 <= p["p_btts"] <= 1.0
        assert 0.0 <= p["p_over_2_5"] <= 1.0


def test_lock_v3_operator_overrides_document_a_and_c_verdicts(lock_v3) -> None:
    """v3 audit must explicitly mention A FAIL + C NO-GO in overrides."""
    overrides = " ".join(lock_v3.get("operator_overrides", []))
    # C verdict (hierarchical, ESS 299, Gelman 2005, divergences)
    assert "Ola 1.C" in overrides
    assert "hierarchical" in overrides.lower()
    assert "299" in overrides  # the ESS value
    # A verdict (Transfermarkt, monotonic, ruled out in both modes)
    assert "Ola 2.A" in overrides
    assert "Transfermarkt" in overrides
    assert "Peeters" in overrides
    assert "symmetric" in overrides.lower()  # the rescue
    # FAIL branch + lock_v1 ships
    assert "FAIL" in overrides
    assert "lock_v1" in overrides.lower() or "lock.json" in overrides.lower()


def test_lock_v3_forced_emit_reason_explains_audit(lock_v3) -> None:
    reason = lock_v3.get("forced_emit_reason", "")
    assert "Ola 3.F" in reason
    assert "FAIL" in reason
    assert "v3" in reason.lower()


def test_lock_v3_content_hash_matches_recompute(lock_v3) -> None:
    """Tamper-verify: re-hash the lock and compare to stored content_hash."""
    stored = lock_v3.get("content_hash")
    assert stored, "lock_v3 missing content_hash field"

    # Recompute hash over the lock body MINUS the content_hash field.
    lock_no_hash = {k: v for k, v in lock_v3.items() if k != "content_hash"}
    serialized = json.dumps(lock_no_hash, sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert stored == recomputed, (
        f"content_hash mismatch:\n  stored:  {stored}\n  recompute: {recomputed}\n"
        "Lock_v3 has been tampered with OR the serialization convention has "
        "changed."
    )


def test_lock_v3_tamper_detection(lock_v3) -> None:
    """Mutating any prediction value MUST invalidate the recomputed hash."""
    stored = lock_v3["content_hash"]
    tampered = json.loads(json.dumps(lock_v3))  # deep copy
    pred_key = "wc2026_v3_calibrated_dibp_no_mv_no_hierarchy"
    tampered["fixtures"][0]["predictions"][pred_key]["p_home_win"] += 0.01

    lock_no_hash = {k: v for k, v in tampered.items() if k != "content_hash"}
    serialized = json.dumps(lock_no_hash, sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert recomputed != stored, "tamper detection broken — hash unchanged after mutation"


def test_lock_v3_n_fixtures_field_consistent(lock_v3) -> None:
    assert lock_v3["n_fixtures_total"] == 72
    assert lock_v3["n_fixtures_with_predictions"] == 72
    assert len(lock_v3["fixtures"]) == 72


def test_lock_v3_held_out_tournaments_inherited_from_v2(lock_v3) -> None:
    """v3 inherits the immovable 4-tournament hold-out per PLAN.md §"Data split"."""
    held = lock_v3["held_out_tournaments"]
    assert set(held) == {"wc_2022", "afcon_2023", "copa_2024", "euro_2024"}
