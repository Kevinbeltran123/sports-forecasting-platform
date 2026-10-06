"""CLV (Closing Line Value) recorder.

CLV formula (CLV-02):
    fair = remove_vig(closing_odds_dict)
    clv_percentage = (odds_at_pick / fair[selection] - 1) * 100

Vig removal (PITFALLS.md Pitfall 7) is mandatory: comparing raw Pinnacle odds
against pick odds inflates CLV by the bookmaker overround (~2-4%) and biases
the project's primary success metric (CLV > +3%) systematically upward.

Persistence decision (frozen):
    ClvRecord.pinnacle_closing_odds = closing_odds_dict[selection]      # RAW
    ClvRecord.implied_prob_closing  = 1.0 / closing_odds_dict[selection]  # RAW
    ClvRecord.clv_percentage        = vig-removed CLV math                # FAIR
The audit trail keeps the actual book quote; the success metric uses fair odds.

Rolling average (CLV-03): tracks rolling 50-pick CLV for trend alerting.
If average drops below +1%, alert is triggered (Phase 4 sends Telegram warning).
"""

from datetime import UTC, datetime

import structlog

from bip.clv.odds_math import remove_vig
from bip.core.errors import ClvError
from bip.core.storage.models import ClvRecord
from bip.core.storage.repositories import ClvRecordRepository
from supabase import Client

logger = structlog.get_logger(__name__)


def calculate_clv_percentage(
    odds_at_pick: float,
    closing_odds_dict: dict[str, float],
    selection: str,
) -> float:
    """Calculate CLV percentage against the vig-removed (fair) closing price.

    Formula: (odds_at_pick / fair_odds[selection] - 1) * 100

    The closing odds dict is normalised via `remove_vig` first so that the
    comparison is against the bookmaker's true probability estimate, not the
    margin-inflated quote. See PITFALLS.md Pitfall 7.

    Args:
        odds_at_pick: Decimal odds we recorded when placing the pick.
        closing_odds_dict: All Pinnacle decimal closing odds for the market
            (e.g., {"1": 1.95, "X": 3.40, "2": 4.20}). Caller is responsible
            for projecting the Pinnacle h2h payload into this shape.
        selection: Key in `closing_odds_dict` identifying the picked outcome.

    Returns:
        CLV percentage. Positive = we beat the (fair) closing line.

    Raises:
        ClvError: If `odds_at_pick` is non-positive, the dict is invalid
            (empty / contains odd <= 1.0), or `selection` is not a key.
    """
    if odds_at_pick <= 0:
        raise ClvError(
            f"Invalid odds_at_pick={odds_at_pick} -- must be positive"
        )
    try:
        fair = remove_vig(closing_odds_dict)
    except ValueError as exc:
        raise ClvError(f"Vig removal failed: {exc}") from exc
    if selection not in fair:
        raise ClvError(
            f"Selection {selection!r} not in closing_odds_dict "
            f"(keys={sorted(fair.keys())})"
        )
    fair_odd = fair[selection]
    return (odds_at_pick / fair_odd - 1.0) * 100.0


def compute_rolling_clv_average(clv_values: list[float]) -> float:
    """Compute rolling average using the last 50 values.

    CLV-03: If this average drops below +1.0, the caller should trigger
    a Telegram warning to pause betting and audit.

    Args:
        clv_values: All historical CLV percentages in chronological order.

    Returns:
        Mean of the last 50 values (or all values if fewer than 50).
    """
    window = clv_values[-50:] if len(clv_values) > 50 else clv_values
    if not window:
        return 0.0
    return sum(window) / len(window)


class ClvRecorder:
    """Records CLV measurements to Supabase.

    Uses ClvRecordRepository for all database interactions.
    Captures odds_fetched_at timestamp for data-freshness monitoring (D-04c).
    """

    def __init__(self, client: Client) -> None:
        self._repo = ClvRecordRepository(client=client)

    def record(
        self,
        pick_id: int,
        fixture_id: int,
        sport: str,
        market: str,
        odds_at_pick: float,
        closing_odds_dict: dict[str, float],
        selection: str,
        odds_fetched_at: datetime | None = None,
    ) -> ClvRecord:
        """Calculate CLV (vig-removed) and persist to Supabase clv_records.

        Args:
            pick_id: FK to picks table.
            fixture_id: API-Football fixture ID.
            sport: Sport string (e.g., "football").
            market: Market key string (e.g., "btts").
            odds_at_pick: Decimal odds recorded at pick time.
            closing_odds_dict: Pinnacle decimal closing odds for the full
                market (e.g., {"1": 1.95, "X": 3.40, "2": 4.20}). Vig is
                removed proportionally before CLV is computed.
            selection: Key in `closing_odds_dict` identifying the picked
                outcome. Must be present.
            odds_fetched_at: When Pinnacle odds were fetched (D-04c).

        Returns:
            ClvRecord with vig-removed clv_percentage; pinnacle_closing_odds
            stores the RAW closing odd for the selection (frozen persistence
            decision -- audit trail preserves the actual book quote).

        Raises:
            ClvError: If `selection` is missing from `closing_odds_dict`,
                if vig removal fails on bad input, or if the underlying
                Supabase insert raises.
        """
        if selection not in closing_odds_dict:
            raise ClvError(
                f"selection {selection!r} not in closing_odds_dict "
                f"(keys={sorted(closing_odds_dict.keys())})"
            )

        if odds_fetched_at is None:
            odds_fetched_at = datetime.now(UTC)

        clv_pct = calculate_clv_percentage(
            odds_at_pick, closing_odds_dict, selection
        )

        raw_closing_odd = closing_odds_dict[selection]

        clv_record = ClvRecord(
            pick_id=pick_id,
            fixture_id=fixture_id,
            sport=sport,
            market=market,
            odds_at_pick=odds_at_pick,
            pinnacle_closing_odds=raw_closing_odd,
            implied_prob_at_pick=1.0 / odds_at_pick,
            implied_prob_closing=1.0 / raw_closing_odd,
            clv_percentage=clv_pct,
            odds_fetched_at=odds_fetched_at,
        )

        try:
            self._repo.insert(clv_record)
        except Exception as exc:
            raise ClvError(
                f"Failed to record CLV for pick_id={pick_id}: {exc}"
            ) from exc

        logger.info(
            "clv_recorded",
            pick_id=pick_id,
            fixture_id=fixture_id,
            market=market,
            selection=selection,
            odds_at_pick=odds_at_pick,
            pinnacle_closing_odds_raw=raw_closing_odd,
            closing_odds_dict=closing_odds_dict,
            clv_percentage=clv_pct,
        )

        return clv_record
