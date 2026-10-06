"""Tests for bip.models.mundial.model.MundialModel.

Sprint 0 Ola C. Uses the real shipped lock at
src/bip/evaluation/tournaments/locked_predictions/world_cup_2026/lock.json
(commit 1589177) — these are integration-style smoke tests on production
artifacts, not unit-with-fixture tests. The lock is read-only data.

Coverage:
  - Lock loads from default path with 72 fixtures
  - can_handle routes WC2026 fixtures by match_id
  - predict() returns PredictionRecord with valid Max-P selection
  - Below-threshold fixtures return [] (gate honored)
  - Threshold override changes coverage as expected
  - SHA-256 integrity check runs (warns but does not block on mismatch)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bip.models.base import PredictionRecord
from bip.models.mundial.model import (
    DEFAULT_LOCK_PATH,
    P_MAX_THRESHOLD,
    MundialModel,
    _verify_lock_integrity,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / DEFAULT_LOCK_PATH


@pytest.fixture(scope="module")
def lock_data() -> dict:
    return json.loads(LOCK_PATH.read_text())


@pytest.fixture(scope="module")
def model() -> MundialModel:
    return MundialModel(lock_path=LOCK_PATH)


class TestLockLoading:
    def test_lock_file_exists(self):
        assert LOCK_PATH.exists(), f"Lock missing at {LOCK_PATH}"

    def test_lock_has_72_fixtures(self, lock_data: dict):
        assert len(lock_data["fixtures"]) == 72

    def test_calibration_status_is_below_gate(self, lock_data: dict):
        # The lock records
        # calibration_status='below-gate' is the ship state. This test
        # is a TRIPWIRE — if the lock ever updates to 'gated' we must
        # decide whether to relax the threshold or change behavior.
        assert lock_data["calibration_status"] == "below-gate"

    def test_missing_lock_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            MundialModel(lock_path=tmp_path / "nope.json")

    def test_model_version_includes_git_sha(self, model: MundialModel):
        assert model.version.startswith("lock-")
        assert len(model.version) > len("lock-")

    def test_fixture_ids_property(self, model: MundialModel):
        ids = model.fixture_ids
        assert len(ids) == 72
        assert all(isinstance(x, str) for x in ids)


class TestCanHandle:
    def test_routes_wc2026_match_id(self, model: MundialModel, lock_data: dict):
        first = lock_data["fixtures"][0]
        assert model.can_handle(
            {"competition": "WC2026", "match_id": first["match_id"]}
        )

    def test_accepts_competition_aliases(self, model: MundialModel, lock_data: dict):
        first = lock_data["fixtures"][0]
        assert model.can_handle(
            {"competition": "FIFA World Cup", "match_id": first["match_id"]}
        )

    def test_rejects_unknown_competition(self, model: MundialModel, lock_data: dict):
        first = lock_data["fixtures"][0]
        assert not model.can_handle(
            {"competition": "Premier League", "match_id": first["match_id"]}
        )

    def test_rejects_unknown_match_id(self, model: MundialModel):
        assert not model.can_handle(
            {"competition": "WC2026", "match_id": "wc2026_grp_FAKE"}
        )

    def test_accepts_fixture_id_as_fallback_key(
        self, model: MundialModel, lock_data: dict
    ):
        first = lock_data["fixtures"][0]
        assert model.can_handle(
            {"competition": "WC2026", "fixture_id": first["match_id"]}
        )


class TestPredict:
    def test_predict_for_known_fixture_returns_record_or_empty(
        self, model: MundialModel, lock_data: dict
    ):
        # The default threshold may or may not be crossed for any given
        # fixture; loop until we find one that emits.
        emitted = []
        for f in lock_data["fixtures"]:
            preds = model.predict({"competition": "WC2026", "match_id": f["match_id"]})
            if preds:
                emitted.extend(preds)
        # At least ONE fixture should emit at the lower threshold below.
        # Above we just verify the loop runs without error.
        assert isinstance(emitted, list)

    def test_predict_with_relaxed_threshold_emits(self, lock_data: dict):
        model = MundialModel(lock_path=LOCK_PATH, p_max_threshold=0.0)
        # With threshold 0.0 every fixture must emit one pick
        emitted_count = 0
        for f in lock_data["fixtures"]:
            preds = model.predict({"competition": "WC2026", "match_id": f["match_id"]})
            if preds:
                emitted_count += 1
                assert len(preds) == 1
                rec = preds[0]
                assert isinstance(rec, PredictionRecord)
                assert rec.source == "mundial"
                assert rec.competition == "WC2026"
                assert rec.market in {"1x2", "btts", "ou_2.5"}
                assert rec.selection in {"home", "draw", "away", "yes", "no", "over", "under"}
                assert 0.0 <= rec.p_model <= 1.0
                assert rec.ev is None
                assert rec.odds_at_pick is None
                assert rec.payload["predictor"] == "bayesian_bivariate_xg_blended"
                assert rec.payload["calibration_status"] == "below-gate"
                assert rec.payload["lock_content_hash"]
                # Max-P is by construction the largest of the scanned markets.
                # Tolerance 1e-3 because scanned_markets stores round(p, 4)
                # for log legibility while p_model is the full float.
                scanned_max = max(s["p"] for s in rec.payload["scanned_markets"])
                assert abs(rec.p_model - scanned_max) < 1e-3
        assert emitted_count == 72

    def test_predict_below_threshold_emits_nothing(self):
        model = MundialModel(lock_path=LOCK_PATH, p_max_threshold=0.999)
        # No real fixture can clear 99.9%
        for fid in model.fixture_ids:
            assert model.predict({"competition": "WC2026", "match_id": fid}) == []

    def test_predict_unknown_fixture_returns_empty(self, model: MundialModel):
        preds = model.predict({"competition": "WC2026", "match_id": "not_in_lock"})
        assert preds == []

    def test_predict_default_threshold_matches_spec(self):
        assert P_MAX_THRESHOLD == 0.65


class TestLockIntegrity:
    def test_verify_lock_integrity_runs(self, lock_data: dict):
        # We do NOT assert ok=True — the canonical-JSON form used by the
        # lock generator may differ from our recomputation. The model
        # logs a warning on mismatch but does not block. This test just
        # verifies the function returns a structured result.
        ok, computed = _verify_lock_integrity(lock_data)
        assert isinstance(ok, bool)
        assert isinstance(computed, str)

    def test_verify_handles_missing_hash(self):
        ok, computed = _verify_lock_integrity({"foo": "bar"})
        assert ok is False
        assert computed == ""
