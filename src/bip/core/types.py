"""Shared enums and type aliases used across the system."""

from enum import StrEnum


class League(StrEnum):
    """Target leagues for prediction."""

    premier_league = "premier_league"
    la_liga = "la_liga"
    bundesliga = "bundesliga"
    serie_a = "serie_a"
    ligue_1 = "ligue_1"


class PickStatus(StrEnum):
    """Status of a betting pick."""

    pending = "pending"
    won = "won"
    lost = "lost"
    void = "void"
    push = "push"
    filtered = "filtered"   # D-03: did not clear edge / market_cap / claude_api_unavailable
    rejected = "rejected"   # D-03: Claude Role C returned REJECT (never sent)


class CalibrationMethod(StrEnum):
    """Probability calibration methods."""

    isotonic = "isotonic"
    platt = "platt"


class AggregationPeriod(StrEnum):
    """Time periods for performance aggregation."""

    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    season = "season"
    all_time = "all_time"


class MarketKey(StrEnum):
    """Canonical market identifier (G-MAINT-05).

    String-valued for backwards compatibility with existing YAML keys, DB
    columns, and Pydantic ``str`` fields (Prediction.market, Pick.market).

    The canonical form is the lowercase YAML ``key:`` field from
    ``src/bip/sports/football/config/markets.yaml``. External aliases
    (legacy 1X2, The Odds API h2h/totals/alternate_spreads/etc.) are mapped
    at boundaries via :meth:`from_str` and :meth:`to_odds_api`.

    Alias matrix (see :meth:`from_str` for the full lookup table):
      ONEXTWO   <- 1x2 / onextwo / h2h
      BTTS      <- btts
      OU        <- ou / totals / over_under
      AH        <- ah / alternate_spreads / asian_handicap
      CORNERS   <- corners
    """

    ONEXTWO = "onextwo"
    BTTS = "btts"
    OU = "ou"
    AH = "ah"
    CORNERS = "corners"

    @classmethod
    def from_str(cls, raw: str) -> "MarketKey":
        """Resolve a market identifier (canonical or alias) to its MarketKey member.

        Case-insensitive and whitespace-insensitive. Raises ValueError with the
        raw input embedded in the message when no alias matches.
        """
        normalized = raw.lower().strip()
        aliases: dict[str, MarketKey] = {
            "1x2": cls.ONEXTWO,
            "onextwo": cls.ONEXTWO,
            "h2h": cls.ONEXTWO,
            "btts": cls.BTTS,
            "ou": cls.OU,
            "totals": cls.OU,
            "over_under": cls.OU,
            "ah": cls.AH,
            "alternate_spreads": cls.AH,
            "asian_handicap": cls.AH,
            "corners": cls.CORNERS,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown market key: {raw!r}")

    def to_odds_api(self) -> str:
        """Map to The Odds API market key.

        Returns ``""`` for CORNERS (not supported by The Odds API). Callers
        that need a None-vs-empty distinction should check at the boundary
        (see ``OddsApiClient.map_market_key``).
        """
        mapping: dict[MarketKey, str] = {
            MarketKey.ONEXTWO: "h2h",
            MarketKey.BTTS: "btts",
            MarketKey.OU: "totals",
            MarketKey.AH: "alternate_spreads",
            MarketKey.CORNERS: "",
        }
        return mapping[self]

    def to_threshold_attr(self) -> str:
        """Bridge to the ``ModelParams.edge_threshold_<x>`` attribute name.

        Special-cases ONEXTWO -> ``edge_threshold_1x2`` because the Pydantic
        field uses the legacy ``1x2`` spelling. All other members compose
        directly with their canonical value.
        """
        if self == MarketKey.ONEXTWO:
            return "edge_threshold_1x2"
        return f"edge_threshold_{self.value}"
