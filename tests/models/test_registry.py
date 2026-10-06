"""Tests for bip.models.registry — ModelRegistry routing.

Sprint 0 Ola B. Verifies:
  - register() rejects non-BaseModel instances
  - get() / __contains__ / __len__ behave correctly
  - iter_applicable() routes via can_handle()
  - A model that raises in can_handle does not block others
  - Re-registering by name replaces (last write wins)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from bip.models.base import BaseModel, PredictionRecord
from bip.models.registry import ModelRegistry


def _rec(source: str, competition: str) -> PredictionRecord:
    return PredictionRecord(
        source=source,
        fixture_id="fx-001",
        competition=competition,
        home_team="A",
        away_team="B",
        match_datetime=datetime(2026, 6, 1, tzinfo=UTC),
        market="1x2",
        selection="home",
        p_model=0.5,
        model_version="test-0.0.0",
    )


class LigasStub(BaseModel):
    name = "ligas"
    version = "0.1.0"

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        return fixture.get("competition") in {"PL", "La Liga", "Serie A"}

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        return [_rec("ligas", fixture["competition"])]


class MundialStub(BaseModel):
    name = "mundial"
    version = "0.1.0"

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        return fixture.get("competition") == "WC2026"

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        return [_rec("mundial", "WC2026")]


class BrokenCanHandle(BaseModel):
    name = "broken"
    version = "0.0.1"

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        raise RuntimeError("boom")

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        return []


class TestModelRegistryBasics:
    def test_empty_registry_is_falsy_via_len(self):
        r = ModelRegistry()
        assert len(r) == 0
        assert r.names() == []

    def test_register_rejects_non_basemodel(self):
        r = ModelRegistry()
        with pytest.raises(TypeError):
            r.register("not a model")  # type: ignore[arg-type]

    def test_register_and_get(self):
        r = ModelRegistry()
        m = LigasStub()
        r.register(m)
        assert "ligas" in r
        assert r.get("ligas") is m
        assert len(r) == 1
        assert r.names() == ["ligas"]

    def test_get_missing_raises_key_error(self):
        r = ModelRegistry()
        with pytest.raises(KeyError):
            r.get("nope")

    def test_re_register_replaces(self):
        r = ModelRegistry()
        m1 = LigasStub()
        m2 = LigasStub()
        r.register(m1)
        r.register(m2)
        assert len(r) == 1
        assert r.get("ligas") is m2


class TestIterApplicable:
    def test_routes_by_can_handle(self):
        r = ModelRegistry()
        r.register(LigasStub())
        r.register(MundialStub())

        applicable = list(r.iter_applicable({"competition": "PL"}))
        assert [m.name for m in applicable] == ["ligas"]

        applicable = list(r.iter_applicable({"competition": "WC2026"}))
        assert [m.name for m in applicable] == ["mundial"]

        applicable = list(r.iter_applicable({"competition": "Eredivisie"}))
        assert applicable == []

    def test_multiple_applicable_models_yielded_in_insertion_order(self):
        # Both models match
        class AllMatch(BaseModel):
            name = "all"

            def can_handle(self, fixture: dict[str, Any]) -> bool:
                return True

            def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
                return []

        class AlsoAll(BaseModel):
            name = "also"

            def can_handle(self, fixture: dict[str, Any]) -> bool:
                return True

            def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
                return []

        r = ModelRegistry()
        r.register(AllMatch())
        r.register(AlsoAll())
        names = [m.name for m in r.iter_applicable({"competition": "anything"})]
        assert names == ["all", "also"]

    def test_broken_can_handle_is_swallowed_and_other_models_still_yield(self):
        r = ModelRegistry()
        r.register(BrokenCanHandle())
        r.register(LigasStub())

        applicable = list(r.iter_applicable({"competition": "PL"}))
        # broken model swallowed; ligas still routes
        assert [m.name for m in applicable] == ["ligas"]

    def test_predict_returns_records(self):
        r = ModelRegistry()
        r.register(MundialStub())
        applicable = list(r.iter_applicable({"competition": "WC2026"}))
        preds = applicable[0].predict({"competition": "WC2026"})
        assert len(preds) == 1
        assert preds[0].source == "mundial"
