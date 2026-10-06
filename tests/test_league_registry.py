"""League registry tests — DATA-03."""

import pytest


class TestLeagueRegistry:
    """DATA-03: LeagueRegistry loads all 5 leagues from YAML."""

    def test_loads_all_5_leagues(self, league_registry):
        """Registry must contain exactly 5 leagues."""
        assert len(league_registry.all_leagues()) == 5

    def test_all_required_slugs_present(self, league_registry):
        """All 5 PL/La Liga/Bundesliga/Serie A/Ligue 1 slugs must be present."""
        slugs = set(league_registry.slugs())
        required = {
            "premier_league",
            "la_liga",
            "bundesliga",
            "serie_a",
            "ligue_1",
        }
        assert required == slugs

    def test_get_by_slug_returns_league_config(self, league_registry):
        """get() by slug must return a LeagueConfig with api_football_league_id."""
        cfg = league_registry.get("premier_league")
        assert cfg.api_mappings.api_football_league_id == 39

    def test_missing_slug_raises_key_error(self, league_registry):
        """get() with unknown slug raises KeyError."""
        with pytest.raises(KeyError, match="tennis"):
            league_registry.get("tennis")

    def test_league_config_has_odds_api_sport_key(self, league_registry):
        """Each league must have odds_api_sport_key for CLV mapping."""
        for league in league_registry.all_leagues():
            assert league.api_mappings.odds_api_sport_key != "", (
                f"{league.slug} missing odds_api_sport_key"
            )
