"""Football market config loader -- CORE-05."""

from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger(__name__)

_MARKETS_FILE = Path(__file__).parent / "markets.yaml"


def load_markets(markets_file: Path | None = None) -> list[dict]:
    """Load market definitions from markets.yaml.

    Returns list of dicts with keys: key, display, edge_threshold, kelly_max, odds_api_key.
    No Market enum -- markets are runtime YAML config (CORE-05).
    """
    path = markets_file or _MARKETS_FILE
    with open(path) as f:
        data = yaml.safe_load(f)
    markets = data.get("markets", [])
    logger.info("markets_loaded", count=len(markets), path=str(path))
    return markets
