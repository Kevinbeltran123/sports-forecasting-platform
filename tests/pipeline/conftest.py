"""Shared mock factories for the v4 pipeline tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from bip.models.base import BaseModel, PredictionRecord


# ──────────────────────────────────────────────────────────────────────
# Supabase client mock
# ──────────────────────────────────────────────────────────────────────


class FakeQuery:
    """Builder mimicking supabase-py's chained query interface."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.inserts: list[dict] = []
        self.updates: list[tuple[dict, dict]] = []  # (filter, payload)
        self._pending_filter: dict[str, Any] = {}

    def insert(self, payload: dict) -> "FakeQuery":
        self.inserts.append(payload)
        return self

    def update(self, payload: dict) -> "FakeQuery":
        self._pending_payload = payload
        return self

    def select(self, _cols: str = "*") -> "FakeQuery":
        return self

    def eq(self, col: str, val: Any) -> "FakeQuery":
        self._pending_filter[col] = val
        return self

    def gte(self, col: str, val: Any) -> "FakeQuery":
        self._pending_filter[f">={col}"] = val
        return self

    def lte(self, col: str, val: Any) -> "FakeQuery":
        self._pending_filter[f"<={col}"] = val
        return self

    def lt(self, col: str, val: Any) -> "FakeQuery":
        self._pending_filter[f"<{col}"] = val
        return self

    def is_(self, col: str, val: Any) -> "FakeQuery":
        self._pending_filter[f"is:{col}"] = val
        return self

    def in_(self, col: str, vals: list) -> "FakeQuery":
        self._pending_filter[f"in:{col}"] = vals
        return self

    def execute(self) -> Any:
        # For inserts: echo back with synthetic id
        if self.inserts:
            last = self.inserts[-1]
            return MagicMock(data=[{**last, "id": f"uuid-{len(self.inserts)}"}])
        # For updates: record + ack
        if getattr(self, "_pending_payload", None) is not None:
            self.updates.append((dict(self._pending_filter), self._pending_payload))
            self._pending_payload = None
            self._pending_filter = {}
            return MagicMock(data=[{"status": "updated"}])
        # For selects: return rows
        result = list(self.rows)
        self._pending_filter = {}
        return MagicMock(data=result)


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.tables: dict[str, FakeQuery] = {}

    def table(self, name: str) -> FakeQuery:
        if name not in self.tables:
            self.tables[name] = FakeQuery()
        return self.tables[name]


@pytest.fixture
def fake_client() -> FakeSupabaseClient:
    return FakeSupabaseClient()


# ──────────────────────────────────────────────────────────────────────
# Stub BaseModel implementations
# ──────────────────────────────────────────────────────────────────────


class StubLigasModel(BaseModel):
    name = "stub_ligas"
    version = "test-0.0"

    def __init__(self, emit_per_fixture: int = 1) -> None:
        self._emit_per_fixture = emit_per_fixture

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        return fixture.get("competition") in {"PL", "La Liga"}

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        out: list[PredictionRecord] = []
        for i in range(self._emit_per_fixture):
            out.append(
                PredictionRecord(
                    source="ligas",
                    fixture_id=str(fixture.get("fixture_id")),
                    competition=str(fixture.get("competition")),
                    home_team=fixture.get("home_team", "Home"),
                    away_team=fixture.get("away_team", "Away"),
                    match_datetime=fixture.get("match_datetime")
                    or datetime(2026, 6, 1, 19, 0, 0, tzinfo=UTC),
                    market="1x2",
                    selection=("home", "draw", "away")[i % 3],
                    p_model=0.5,
                    ev=0.08,
                    odds_at_pick=2.0,
                    payload={"stub": True},
                    model_version=self.version,
                )
            )
        return out


class StubMundialModel(BaseModel):
    name = "stub_mundial"
    version = "test-0.0"

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        return fixture.get("competition") == "WC2026"

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        return [
            PredictionRecord(
                source="mundial",
                fixture_id=str(fixture.get("fixture_id")),
                competition="WC2026",
                home_team=fixture.get("home_team", "A"),
                away_team=fixture.get("away_team", "B"),
                match_datetime=fixture.get("match_datetime")
                or datetime(2026, 6, 11, 18, 0, 0, tzinfo=UTC),
                market="1x2",
                selection="home",
                p_model=0.7,
                ev=None,
                odds_at_pick=None,
                payload={"stub": True, "predictor": "stub"},
                model_version=self.version,
            )
        ]


class BrokenModel(BaseModel):
    name = "broken"
    version = "0.0.0"

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        return True

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        raise RuntimeError("predict() exploded")
