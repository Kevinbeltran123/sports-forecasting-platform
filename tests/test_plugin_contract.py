"""Contract tests for SportPlugin ABC — CORE-01, CORE-04."""

import abc
import pytest


class TestSportPluginABC:
    """CORE-01: SportPlugin ABC has all required abstract methods."""

    def test_sport_plugin_has_all_abstract_methods(self):
        """SportPlugin must define get_fixtures, build_features, predict, build_claude_context, get_available_markets, get_opening_odds (G-CODE-01)."""
        from bip.sports import SportPlugin
        abstract_methods = getattr(SportPlugin, "__abstractmethods__", set())
        required = {
            "get_fixtures",
            "build_features",
            "predict",
            "build_claude_context",
            "get_available_markets",
            "get_opening_odds",  # G-CODE-01: orchestrator pipeline contract
        }
        assert required == abstract_methods, (
            f"Missing abstract methods: {required - abstract_methods}"
        )

    def test_sport_plugin_is_abc(self):
        """SportPlugin must be an ABC, not a Protocol."""
        from bip.sports import SportPlugin
        assert issubclass(SportPlugin, abc.ABC)

    def test_cannot_instantiate_sport_plugin_directly(self):
        """Attempting to instantiate SportPlugin directly must raise TypeError."""
        from bip.sports import SportPlugin
        with pytest.raises(TypeError):
            SportPlugin()  # type: ignore

    def test_probability_map_is_pydantic_model(self):
        """ProbabilityMap must be a Pydantic BaseModel with required fields — D-02c."""
        from bip.sports import ProbabilityMap
        pm = ProbabilityMap(
            fixture_id=1,
            market="onextwo",
            probabilities={"1": 0.45, "X": 0.28, "2": 0.27},
            model_version="v1.0",
            computed_at=__import__("datetime").datetime(2026, 4, 22, 15, 0, 0),
        )
        assert pm.fixture_id == 1
        assert pm.probabilities["1"] == pytest.approx(0.45)

    def test_fixture_data_is_pydantic_model(self):
        """FixtureData must be a Pydantic BaseModel — D-02d."""
        from bip.sports import FixtureData
        fd = FixtureData(
            fixture_id=99,
            league="premier_league",
            sport="football",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff_utc=__import__("datetime").datetime(2026, 4, 22, 15, 0, 0),
        )
        assert fd.sport == "football"

    def test_feature_matrix_has_computed_at(self):
        """FeatureMatrix.computed_at is required for DATA-05 point-in-time correctness."""
        from bip.sports import FeatureMatrix
        import inspect
        fields = FeatureMatrix.model_fields
        assert "computed_at" in fields, "FeatureMatrix must have computed_at field"


class TestFootballPluginContract:
    """CORE-04: FootballPlugin implements all SportPlugin methods."""

    def test_football_plugin_is_sport_plugin_subclass(self):
        """FootballPlugin must be a concrete subclass of SportPlugin."""
        from bip.sports import SportPlugin
        from bip.sports.football.plugin import FootballPlugin
        assert issubclass(FootballPlugin, SportPlugin)

    def test_football_plugin_implements_all_methods(self):
        """FootballPlugin must not have any abstract methods remaining."""
        from bip.sports.football.plugin import FootballPlugin
        remaining = getattr(FootballPlugin, "__abstractmethods__", set())
        assert len(remaining) == 0, (
            f"FootballPlugin still has unimplemented abstract methods: {remaining}"
        )

    def test_football_plugin_get_available_markets_returns_list(self, settings):
        """FootballPlugin.get_available_markets() must return a non-empty list of strings."""
        from bip.sports.football.plugin import FootballPlugin
        plugin = FootballPlugin(settings=settings)
        markets = plugin.get_available_markets()
        assert isinstance(markets, list)
        assert len(markets) > 0
        assert all(isinstance(m, str) for m in markets)
