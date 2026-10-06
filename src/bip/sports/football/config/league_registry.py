"""Football league registry -- loads all leagues from YAML configs."""

from pathlib import Path

import structlog
import yaml

from bip.sports.football.config.league_config import LeagueConfig

logger = structlog.get_logger(__name__)


class LeagueRegistry:
    """Registry of football leagues, loaded from a directory of YAML files."""

    def __init__(self, config_dir: Path) -> None:
        if not config_dir.is_dir():
            raise FileNotFoundError(f"League config directory not found: {config_dir}")

        self._leagues: dict[str, LeagueConfig] = {}

        for yaml_file in sorted(config_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            config = LeagueConfig(**data)
            self._leagues[config.slug] = config
            logger.info("league_loaded", name=config.name, slug=config.slug)

    def get(self, slug: str) -> LeagueConfig:
        """Get a league config by slug. Raises KeyError if not found."""
        if slug not in self._leagues:
            available = ", ".join(sorted(self._leagues.keys()))
            raise KeyError(f"League '{slug}' not configured. Available: {available}")
        return self._leagues[slug]

    def all_leagues(self) -> list[LeagueConfig]:
        """Return all loaded league configs."""
        return list(self._leagues.values())

    def slugs(self) -> list[str]:
        """Return all loaded league slugs."""
        return list(self._leagues.keys())
