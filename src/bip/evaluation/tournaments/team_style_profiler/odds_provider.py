"""API-Football odds parsing — Pinnacle filter + canonical market names.

API-Football's /odds endpoint returns a deeply-nested payload with all
bookmakers and all markets. This module:
  1. Parses the response to typed Pydantic models.
  2. Filters to Pinnacle only (per operator decision: market-efficient
     reference for true edge detection).
  3. Maps API-Football's market names to TSP's canonical names.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

PINNACLE_NAME = "Pinnacle"


# ─── API-Football /odds parsers ─────────────────────────────────────────


class OddValue(BaseModel):
    """A single odd entry (e.g., 'Over' with cuota 1.85)."""

    model_config = ConfigDict(extra="ignore")
    value: str
    """Selection name. E.g. 'Home', 'Draw', 'Away', 'Over', 'Under',
    'Yes', 'No'. For O/U: 'Over 2.5'. For AH: 'Home -1.5'."""
    odd: str
    """Decimal odd as string. e.g. '1.85'."""

    @property
    def odd_float(self) -> float:
        return float(self.odd)


class OddBetMarket(BaseModel):
    """A market within a bookmaker (e.g., 'Match Winner', 'Goals Over/Under 2.5')."""

    model_config = ConfigDict(extra="ignore")
    id: int
    name: str
    values: list[OddValue] = Field(default_factory=list)


class OddBookmaker(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    name: str
    bets: list[OddBetMarket] = Field(default_factory=list)


class OddsFixture(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bookmakers: list[OddBookmaker] = Field(default_factory=list)


class OddsApiResponseItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fixture: dict
    bookmakers: list[OddBookmaker] = Field(default_factory=list)


class OddsApiResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: list[OddsApiResponseItem] = Field(default_factory=list)


# ─── Canonical market mapping ───────────────────────────────────────────


# Map (api_football_market_name, selection_value) -> TSP canonical name.
# Selection_value can be a parametrized string like 'Over 2.5' or 'Home'.
def _normalize_selection(value: str) -> str:
    return value.strip().lower()


CANONICAL_MARKET_MAP: dict[tuple[str, str], str] = {
    ("both teams to score", "yes"): "BTTS_yes",
    ("both teams to score", "no"): "BTTS_no",
    # Some bookmakers use "Both Teams Score" without "to"
    ("both teams score", "yes"): "BTTS_yes",
    ("both teams score", "no"): "BTTS_no",
    # Goals O/U — handled separately because line is part of the value
    # Cards O/U — handled separately
    # Corners O/U — handled separately
    # AH — handled separately
}


@dataclass(frozen=True)
class CanonicalOdd:
    """A single canonical odd: TSP market name + decimal odd."""

    market: str
    """e.g. 'BTTS_yes', 'O2.5', 'U2.5', 'corners_O9.5', 'cards_O4.5',
    'AH_home_-1.5', 'AH_away_-1.5', etc."""
    odd: float


def _parse_over_under_value(value: str, family: str) -> str | None:
    """Convert 'Over 2.5' / 'Under 2.5' to canonical 'O2.5' / 'U2.5'.

    family in {'goals', 'corners', 'cards'} determines the canonical prefix.
    """
    v = value.strip().lower()
    parts = v.split()
    if len(parts) != 2:
        return None
    side, line = parts[0], parts[1]
    if side not in {"over", "under"}:
        return None
    try:
        line_f = float(line)
    except ValueError:
        return None
    prefix_map = {
        "goals": "" if side == "over" else "U",
        "corners": "corners_O" if side == "over" else "corners_U",
        "cards": "cards_O" if side == "over" else "cards_U",
    }
    if family == "goals":
        prefix = "O" if side == "over" else "U"
    elif family == "corners":
        prefix = "corners_O" if side == "over" else "corners_U"
    elif family == "cards":
        prefix = "cards_O" if side == "over" else "cards_U"
    else:
        return None
    return f"{prefix}{line_f}".rstrip("0").rstrip(".") + (
        "" if "." in f"{prefix}{line_f}" else ""
    )


def _parse_ah_value(value: str) -> str | None:
    """Convert 'Home -1.5' / 'Away +0.5' to 'AH_home_-1.5' / 'AH_away_+0.5'."""
    v = value.strip().lower()
    parts = v.split()
    if len(parts) != 2:
        return None
    side, line_str = parts[0], parts[1]
    if side not in {"home", "away"}:
        return None
    try:
        line_f = float(line_str)
    except ValueError:
        return None
    sign = "+" if line_f >= 0 else "-"
    return f"AH_{side}_{sign}{abs(line_f)}"


def extract_pinnacle_canonical_odds(
    payload: dict,
) -> dict[str, float]:
    """Given an API-Football /odds response payload, extract Pinnacle's
    odds for canonical TSP markets only.

    Args:
        payload: parsed JSON dict from /odds (single fixture).

    Returns:
        Dict mapping canonical market name to decimal odd. Empty when
        Pinnacle isn't in the payload or no canonical markets matched.
    """
    parsed = OddsApiResponse.model_validate(payload)
    out: dict[str, float] = {}
    if not parsed.response:
        return out
    item = parsed.response[0]
    pinnacle: OddBookmaker | None = None
    for bm in item.bookmakers:
        if bm.name == PINNACLE_NAME:
            pinnacle = bm
            break
    if pinnacle is None:
        return out

    for bet in pinnacle.bets:
        name = bet.name.strip().lower()
        for v in bet.values:
            value_norm = _normalize_selection(v.value)
            # Direct map (BTTS)
            canonical = CANONICAL_MARKET_MAP.get((name, value_norm))
            if canonical is not None:
                out[canonical] = v.odd_float
                continue
            # O/U parsing — based on market name family
            if "goals over/under" in name or name == "over/under":
                c = _parse_over_under_value(v.value, "goals")
                if c is not None:
                    out[c] = v.odd_float
                    continue
            if "corners over/under" in name or "total corners" in name:
                c = _parse_over_under_value(v.value, "corners")
                if c is not None:
                    out[c] = v.odd_float
                    continue
            if "cards over/under" in name or "total cards" in name:
                c = _parse_over_under_value(v.value, "cards")
                if c is not None:
                    out[c] = v.odd_float
                    continue
            # AH
            if "asian handicap" in name:
                c = _parse_ah_value(v.value)
                if c is not None:
                    out[c] = v.odd_float
                    continue

    return out


def implied_probability(decimal_odd: float) -> float:
    """1/odd is the bookmaker-implied probability (no margin removal).

    For Pinnacle the margin is ~2-3% on 1X2 / 1-2% on totals. We do
    NOT remove margin here — Z-score gate handles the comparison
    directly with raw implied prob.
    """
    if decimal_odd <= 1.0:
        return 1.0
    return 1.0 / decimal_odd
