"""LigasModel BaseModel contract conformance.

Sprint 1 Ola A. No real data — synthetic fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from bip.models.base import BaseModel, PredictionRecord
from bip.models.ligas import (
    LEAGUE_ID_MAP,
    MARKETS,
    SUPPORTED_LEAGUES,
    LigasModel,
)


class TestContract:
    def test_is_basemodel(self):
        assert issubclass(LigasModel, BaseModel)

    def test_instantiable(self):
        m = LigasModel()
        assert m.name == "ligas_phase1"
        assert m.version
        assert m.ev_threshold == 0.05

    def test_custom_ev_threshold(self):
        m = LigasModel(ev_threshold=0.0)
        assert m.ev_threshold == 0.0


class TestCanHandle:
    @pytest.fixture
    def model(self) -> LigasModel:
        return LigasModel()

    def test_routes_all_5_supported_leagues(self, model: LigasModel):
        for league in SUPPORTED_LEAGUES:
            assert model.can_handle({"competition": league})
            assert model.can_handle({"league": league})

    def test_rejects_unsupported_competition(self, model: LigasModel):
        assert not model.can_handle({"competition": "WC2026"})
        assert not model.can_handle({"competition": "Eredivisie"})
        assert not model.can_handle({"competition": "MLS"})
        assert not model.can_handle({})


class TestConstants:
    def test_supported_leagues_are_5(self):
        assert len(SUPPORTED_LEAGUES) == 5

    def test_markets_are_phase1_only(self):
        # Spec §4.8 Phase 1: 1x2 + ou_2.5. BTTS / corners are Phase 2.
        assert MARKETS == ("1x2", "ou_2.5")

    def test_league_id_map_covers_all(self):
        assert set(LEAGUE_ID_MAP.keys()) == SUPPORTED_LEAGUES
        assert LEAGUE_ID_MAP["PL"] == 39
        assert LEAGUE_ID_MAP["La Liga"] == 140


class TestPredictWithoutTrainedSubmodels:
    def test_returns_empty_when_no_submodels_trained(self):
        m = LigasModel()
        fixture = {
            "competition": "PL",
            "fixture_id": "fx-1",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_datetime": datetime(2026, 6, 1, 19, 0, 0, tzinfo=UTC),
            "features": np.zeros(5),
        }
        assert m.predict(fixture) == []

    def test_returns_empty_when_features_missing(self):
        m = LigasModel()
        fixture = {
            "competition": "PL",
            "fixture_id": "fx-1",
            "match_datetime": datetime(2026, 6, 1, tzinfo=UTC),
        }
        assert m.predict(fixture) == []

    def test_returns_empty_when_unsupported_competition(self):
        m = LigasModel()
        fixture = {"competition": "WC2026", "features": np.zeros(5)}
        assert m.predict(fixture) == []


class TestPredictWithSyntheticTraining:
    """Train tiny synthetic sub-models and verify predict produces records."""

    @pytest.fixture
    def trained_model(self) -> LigasModel:
        rng = np.random.default_rng(seed=42)
        n = 200
        n_features = 6

        m = LigasModel(ev_threshold=0.0)  # no EV gate for this synthetic test

        # 1x2 with 3 classes
        X = rng.normal(size=(n, n_features))
        # synthetic signal: y = argmax of linear combo of first 3 features
        scores = X[:, :3] + rng.normal(size=(n, 3)) * 0.5
        y_3 = scores.argmax(axis=1).astype(int)
        m.train({"league": "PL", "market": "1x2", "X": X, "y": y_3})

        # ou_2.5 binary
        y_ou = (X[:, 0] + rng.normal(size=n) * 0.5 > 0).astype(int)
        m.train({"league": "PL", "market": "ou_2.5", "X": X, "y": y_ou})

        return m

    def test_predict_emits_records_for_trained_market(self, trained_model: LigasModel):
        fixture = {
            "competition": "PL",
            "fixture_id": "fx-1",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_datetime": datetime(2026, 6, 1, 19, 0, 0, tzinfo=UTC),
            "features": np.zeros(6),
        }
        records = trained_model.predict(fixture)
        # Each market contributes up to its n_selections records (3 for 1x2, 2 for ou)
        # With ev_threshold=0 and no odds, all are emitted unfiltered
        markets_seen = {r.market for r in records}
        # Without odds the EV gate path skips (ev_threshold=0 still skips when odds missing)
        # so we instead pass odds_at_pick to allow emission:
        fixture["odds_at_pick"] = {
            "1x2": {"home": 2.0, "draw": 3.5, "away": 4.0},
            "ou_2.5": {"over": 1.9, "under": 2.0},
        }
        records = trained_model.predict(fixture)
        assert len(records) > 0
        markets_seen = {r.market for r in records}
        assert markets_seen <= {"1x2", "ou_2.5"}
        for r in records:
            assert isinstance(r, PredictionRecord)
            assert r.source == "ligas"
            assert r.competition == "PL"
            assert 0.0 <= r.p_model <= 1.0

    def test_trained_submodels_tracked(self, trained_model: LigasModel):
        keys = trained_model.trained_submodels
        assert ("PL", "1x2") in keys
        assert ("PL", "ou_2.5") in keys


class TestTrainValidation:
    def test_rejects_unsupported_league(self):
        m = LigasModel()
        with pytest.raises(ValueError, match="Unsupported league"):
            m.train(
                {"league": "MLS", "market": "1x2", "X": np.zeros((5, 3)), "y": np.zeros(5)}
            )

    def test_rejects_unsupported_market(self):
        m = LigasModel()
        with pytest.raises(ValueError, match="Unsupported market"):
            m.train(
                {"league": "PL", "market": "btts", "X": np.zeros((5, 3)), "y": np.zeros(5)}
            )

    def test_rejects_missing_xy(self):
        m = LigasModel()
        with pytest.raises(ValueError, match="X and y are required"):
            m.train({"league": "PL", "market": "1x2"})

    def test_rejects_non_dict_payload(self):
        m = LigasModel()
        with pytest.raises(TypeError):
            m.train([1, 2, 3])
