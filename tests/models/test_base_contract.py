"""Tests for bip.models.base — BaseModel ABC + PredictionRecord schema.

Sprint 0 Ola B. Verifies:
  - PredictionRecord round-trips through to_supabase_dict (ISO timestamps)
  - PredictionRecord rejects out-of-range p_model (Pydantic validation)
  - BaseModel.train/evaluate raise NotImplementedError by default
  - A concrete StubModel subclass can be instantiated when can_handle +
    predict are implemented
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from bip.models.base import BaseModel, ModelMetrics, PredictionRecord


def _make_record(**overrides: Any) -> PredictionRecord:
    defaults: dict[str, Any] = {
        "source": "ligas",
        "fixture_id": "fx-001",
        "competition": "PL",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_datetime": datetime(2026, 5, 25, 19, 0, 0, tzinfo=UTC),
        "market": "1x2",
        "selection": "home",
        "p_model": 0.55,
        "ev": 0.072,
        "odds_at_pick": 1.95,
        "payload": {"features_snapshot": {"home_elo": 1850}},
        "model_version": "ligas-1x2-0.1.0",
    }
    defaults.update(overrides)
    return PredictionRecord(**defaults)


class TestPredictionRecord:
    def test_to_supabase_dict_iso_format(self):
        rec = _make_record()
        d = rec.to_supabase_dict()
        assert d["match_datetime"] == "2026-05-25T19:00:00+00:00"
        # created_at is auto-filled with now()
        assert d["created_at"].endswith("+00:00")
        assert d["source"] == "ligas"
        assert d["p_model"] == 0.55

    def test_p_model_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            _make_record(p_model=1.5)
        with pytest.raises(ValidationError):
            _make_record(p_model=-0.1)

    def test_ev_can_be_none(self):
        rec = _make_record(ev=None, odds_at_pick=None)
        assert rec.ev is None
        assert rec.odds_at_pick is None

    def test_payload_defaults_to_empty_dict(self):
        rec = PredictionRecord(
            source="mundial",
            fixture_id="wc-001",
            competition="WC2026",
            home_team="Argentina",
            away_team="Arabia Saudita",
            match_datetime=datetime(2026, 6, 12, 18, 0, 0, tzinfo=UTC),
            market="ou_1.5",
            selection="over",
            p_model=0.81,
            model_version="mundial-lock-1589177",
        )
        assert rec.payload == {}
        assert rec.status == "pending"

    def test_status_default_is_pending(self):
        assert _make_record().status == "pending"


class TestBaseModelContract:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_subclass_must_implement_can_handle_and_predict(self):
        class Incomplete(BaseModel):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        class StubModel(BaseModel):
            name = "stub"
            version = "0.1.0"

            def can_handle(self, fixture: dict[str, Any]) -> bool:
                return fixture.get("competition") == "stub"

            def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
                return [_make_record(competition="stub")]

        m = StubModel()
        assert m.name == "stub"
        assert m.can_handle({"competition": "stub"}) is True
        assert m.can_handle({"competition": "other"}) is False
        preds = m.predict({"competition": "stub"})
        assert len(preds) == 1
        assert preds[0].competition == "stub"

    def test_train_raises_not_implemented_by_default(self):
        class StubModel(BaseModel):
            name = "stub"

            def can_handle(self, fixture: dict[str, Any]) -> bool:
                return False

            def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
                return []

        m = StubModel()
        with pytest.raises(NotImplementedError):
            m.train(None)
        with pytest.raises(NotImplementedError):
            m.evaluate(None)


class TestModelMetrics:
    def test_minimal_construction(self):
        mm = ModelMetrics(n_picks=42, hit_rate=0.58, roi=0.034)
        assert mm.n_picks == 42
        assert mm.brier_score is None
        assert mm.clv_mean is None
        assert mm.notes == ""

    def test_full_construction(self):
        mm = ModelMetrics(
            n_picks=100, hit_rate=0.6, roi=0.05,
            brier_score=0.21, clv_mean=0.012, notes="bootstrap CI: [0.02, 0.08]",
        )
        assert mm.brier_score == 0.21
        assert mm.clv_mean == 0.012
