"""Market YAML config tests — CORE-05 (no hardcoded enum in core)."""

import pytest


class TestMarketConfig:
    """CORE-05: Markets loaded from YAML, not from hardcoded enum."""

    def test_core_types_has_no_market_enum(self):
        """core/types.py must NOT contain a Market enum — CORE-05."""
        import bip.core.types as types
        assert not hasattr(types, "Market"), (
            "Market enum found in core/types.py — violates CORE-05. "
            "Markets must be defined in YAML config."
        )

    def test_football_markets_yaml_exists(self):
        """markets.yaml must exist in sports/football/config/."""
        from pathlib import Path
        markets_path = (
            Path(__file__).parent.parent
            / "src" / "bip" / "sports" / "football" / "config" / "markets.yaml"
        )
        assert markets_path.exists(), f"Missing markets.yaml at {markets_path}"

    def test_market_config_loads_5_markets(self):
        """markets.yaml must define at least 5 markets: btts, ou, onextwo, ah, corners."""
        from bip.sports.football.config.market_config import load_markets
        markets = load_markets()
        keys = [m["key"] for m in markets]
        required = {"btts", "ou", "onextwo", "ah", "corners"}
        assert required.issubset(set(keys)), (
            f"Missing markets: {required - set(keys)}"
        )

    def test_each_market_has_required_fields(self):
        """Each market entry must have: key, display, edge_threshold, kelly_max."""
        from bip.sports.football.config.market_config import load_markets
        for market in load_markets():
            assert "key" in market
            assert "display" in market
            assert "edge_threshold" in market
            assert "kelly_max" in market

    def test_storage_models_use_str_not_market_enum(self, sample_prediction):
        """Prediction.market must be str, not Market enum — CORE-05 / RESEARCH Pitfall 7."""
        assert isinstance(sample_prediction.market, str)
