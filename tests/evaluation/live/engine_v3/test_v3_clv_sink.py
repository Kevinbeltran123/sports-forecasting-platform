"""Tests for src/bip/clv/v3_clv_sink.py.

All tests are OFFLINE — OddsApiClient is fully mocked. No network calls.

Coverage:
- A SENT goals pick with a mocked Pinnacle response yields a clv.parquet
  row with vig-removed clv_percentage.
- A corners pick writes a row with clv_percentage=None and
  reason="clv_unsupported_family".
- A skipped/failed delivery row is excluded from CLV computation.
- write_clv_records partitions correctly by delivery date.
- rolling_clv_from_parquet reads accumulated clv.parquet and returns a mean.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from bip.clv.v3_clv_sink import (
    V3ClvRecord,
    _compute_clv_for_row,
    _extract_closing_odds_dict,
    _resolve_selection_key,
    rolling_clv_from_parquet,
    run_clv_sink_for_delivered_picks,
    write_clv_records,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

FIXTURE_ID = 123456
THESIS_ID = "t-abc-123"
KICKOFF = datetime(2026, 5, 15, 19, 0, 0, tzinfo=UTC)
EVENT_ID = "odds-api-event-uuid"

_PINNACLE_H2H = {
    "key": "pinnacle",
    "markets": [
        {
            "key": "h2h",
            "outcomes": [
                {"name": "Home FC", "price": 2.10},
                {"name": "Draw", "price": 3.40},
                {"name": "Away FC", "price": 3.80},
            ],
        }
    ],
}

_PINNACLE_TOTALS = {
    "key": "pinnacle",
    "markets": [
        {
            "key": "totals",
            "outcomes": [
                {"name": "Over", "price": 1.95},
                {"name": "Under", "price": 1.95},
            ],
        }
    ],
}

_PINNACLE_BTTS = {
    "key": "pinnacle",
    "markets": [
        {
            "key": "btts",
            "outcomes": [
                {"name": "Yes", "price": 1.80},
                {"name": "No", "price": 2.05},
            ],
        }
    ],
}


def _mock_odds_client(event_id: str | None, pinnacle_dict: dict | None) -> AsyncMock:
    client = AsyncMock()
    client.find_event_by_fixture = AsyncMock(return_value=event_id)
    client.fetch_pinnacle_closing_odds = AsyncMock(return_value=pinnacle_dict)
    return client


def _goals_delivery_row(
    send_result: str = "sent",
    book_odd: float = 2.05,
    direction: str = "over",
) -> dict:
    return {
        "fixture_id": FIXTURE_ID,
        "thesis_id": THESIS_ID,
        "family": "goals",
        "market_id": "match_goals_over_2.5",
        "direction": direction,
        "bookmaker_odd": book_odd,
        "telegram_message_id": 42,
        "send_result": send_result,
        "timestamp_utc": KICKOFF,
    }


def _corners_delivery_row() -> dict:
    return {
        "fixture_id": FIXTURE_ID,
        "thesis_id": "t-corners-1",
        "family": "corners",
        "market_id": "match_corners_over_9.5",
        "direction": "over",
        "bookmaker_odd": 1.90,
        "telegram_message_id": 43,
        "send_result": "sent",
        "timestamp_utc": KICKOFF,
    }


FIXTURE_META = {
    FIXTURE_ID: {
        "home_team": "Home FC",
        "away_team": "Away FC",
        "kickoff_utc": KICKOFF,
    }
}


# ──────────────────────────────────────────────────────────────────────
# Unit tests — _extract_closing_odds_dict
# ──────────────────────────────────────────────────────────────────────


class TestExtractClosingOddsDict:
    def test_extracts_h2h_outcomes(self):
        result = _extract_closing_odds_dict(_PINNACLE_H2H, "h2h")
        assert result is not None
        assert "home fc" in result
        assert abs(result["home fc"] - 2.10) < 1e-6

    def test_returns_none_when_market_absent(self):
        result = _extract_closing_odds_dict(_PINNACLE_H2H, "totals")
        assert result is None

    def test_returns_none_for_single_outcome(self):
        single = {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [{"name": "Home", "price": 2.0}]}]}
        result = _extract_closing_odds_dict(single, "h2h")
        assert result is None

    def test_extracts_totals_over_under(self):
        result = _extract_closing_odds_dict(_PINNACLE_TOTALS, "totals")
        assert result is not None
        assert "over" in result
        assert "under" in result


class TestResolveSelectionKey:
    def test_direct_match(self):
        closing = {"over": 1.95, "under": 1.95}
        assert _resolve_selection_key("over", closing) == "over"

    def test_substring_match(self):
        closing = {"home fc": 2.10, "draw": 3.40, "away fc": 3.80}
        key = _resolve_selection_key("home fc", closing)
        assert key == "home fc"

    def test_direction_map_home(self):
        closing = {"home fc": 2.10, "draw": 3.40, "away fc": 3.80}
        key = _resolve_selection_key("home", closing)
        assert key == "home fc"

    def test_no_match_returns_none(self):
        closing = {"team_a": 2.0, "team_b": 2.5}
        assert _resolve_selection_key("draw", closing) is None


# ──────────────────────────────────────────────────────────────────────
# Unit tests — _compute_clv_for_row
# ──────────────────────────────────────────────────────────────────────


class TestComputeClvForRow:
    @pytest.mark.asyncio
    async def test_goals_pick_computes_clv(self):
        """A goals pick with a Pinnacle totals response yields vig-removed CLV."""
        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        row = _goals_delivery_row(book_odd=2.05, direction="over")
        rec = await _compute_clv_for_row(
            row,
            odds_client=client,
            sport_key="soccer_epl",
            home_team="Home FC",
            away_team="Away FC",
            kickoff_utc=KICKOFF,
        )
        assert rec.clv_percentage is not None
        # Pinnacle totals: Over=1.95, Under=1.95 → fair = remove_vig({over:1.95, under:1.95})
        # = {over:2.0, under:2.0} approximately (50/50 after vig removal)
        # CLV = (2.05 / fair_over - 1) * 100 > 0 because 2.05 > fair ~2.0
        assert rec.clv_percentage > 0, f"Expected positive CLV, got {rec.clv_percentage}"
        assert rec.reason is None
        assert rec.odds_fetched_at is not None
        assert rec.family == "goals"

    @pytest.mark.asyncio
    async def test_corners_pick_writes_unsupported_family(self):
        """A corners pick yields clv_percentage=None + reason=clv_unsupported_family."""
        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        row = _corners_delivery_row()
        rec = await _compute_clv_for_row(
            row,
            odds_client=client,
            sport_key="soccer_epl",
            home_team="Home FC",
            away_team="Away FC",
            kickoff_utc=KICKOFF,
        )
        assert rec.clv_percentage is None
        assert rec.reason == "clv_unsupported_family"
        # OddsApiClient should NOT be called for unsupported families
        client.find_event_by_fixture.assert_not_called()
        client.fetch_pinnacle_closing_odds.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_event_match_returns_no_event_match(self):
        """When find_event_by_fixture returns None, reason=no_event_match."""
        client = _mock_odds_client(None, _PINNACLE_TOTALS)
        row = _goals_delivery_row()
        rec = await _compute_clv_for_row(
            row,
            odds_client=client,
            sport_key="soccer_epl",
            home_team="Home FC",
            away_team="Away FC",
            kickoff_utc=KICKOFF,
        )
        assert rec.clv_percentage is None
        assert rec.reason == "no_event_match"

    @pytest.mark.asyncio
    async def test_no_pinnacle_data_returns_no_pinnacle_data(self):
        """When fetch_pinnacle_closing_odds returns None, reason=no_pinnacle_data."""
        client = _mock_odds_client(EVENT_ID, None)
        row = _goals_delivery_row()
        rec = await _compute_clv_for_row(
            row,
            odds_client=client,
            sport_key="soccer_epl",
            home_team="Home FC",
            away_team="Away FC",
            kickoff_utc=KICKOFF,
        )
        assert rec.clv_percentage is None
        assert rec.reason == "no_pinnacle_data"

    @pytest.mark.asyncio
    async def test_no_bookmaker_odd_returns_no_bookmaker_odd(self):
        """A row with no bookmaker_odd returns reason=no_bookmaker_odd."""
        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        row = _goals_delivery_row(book_odd=None)
        row["bookmaker_odd"] = None
        rec = await _compute_clv_for_row(
            row,
            odds_client=client,
            sport_key="soccer_epl",
            home_team="Home FC",
            away_team="Away FC",
            kickoff_utc=KICKOFF,
        )
        assert rec.clv_percentage is None
        assert rec.reason == "no_bookmaker_odd"

    @pytest.mark.asyncio
    async def test_btts_pick_computes_clv(self):
        """A btts/yes pick yields a valid CLV measurement."""
        client = _mock_odds_client(EVENT_ID, _PINNACLE_BTTS)
        row = {
            "fixture_id": FIXTURE_ID,
            "thesis_id": THESIS_ID,
            "family": "btts",
            "market_id": "btts_yes",
            "direction": "yes",
            "bookmaker_odd": 1.85,
            "send_result": "sent",
            "timestamp_utc": KICKOFF,
        }
        rec = await _compute_clv_for_row(
            row,
            odds_client=client,
            sport_key="soccer_epl",
            home_team="Home FC",
            away_team="Away FC",
            kickoff_utc=KICKOFF,
        )
        # Pinnacle BTTS: Yes=1.80, No=2.05 → fair_yes > 1.80 after vig removal
        # Our odd 1.85 vs fair may be positive or negative depending on vig fraction
        assert rec.clv_percentage is not None
        assert rec.reason is None


# ──────────────────────────────────────────────────────────────────────
# Integration test — run_clv_sink_for_delivered_picks
# ──────────────────────────────────────────────────────────────────────


class TestRunClvSinkForDeliveredPicks:
    @pytest.mark.asyncio
    async def test_goals_pick_writes_clv_parquet(self, tmp_path: Path):
        """A SENT goals pick → clv.parquet row with clv_percentage set."""
        # Write a synthetic deliveries.parquet
        date_iso = "2026-05-15"
        part_dir = tmp_path / f"dt={date_iso}"
        part_dir.mkdir(parents=True)
        rows = [_goals_delivery_row(send_result="sent")]
        pl.DataFrame(rows).write_parquet(part_dir / "deliveries.parquet")

        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        records, written = await run_clv_sink_for_delivered_picks(
            date_iso=date_iso,
            odds_client=client,
            fixture_metadata=FIXTURE_META,
            shadow_root=tmp_path,
            sport_key="soccer_epl",
        )

        assert len(records) == 1
        assert records[0].clv_percentage is not None
        assert records[0].reason is None

        # clv.parquet must be written
        clv_path = part_dir / "clv.parquet"
        assert clv_path.exists(), "clv.parquet not written"
        df = pl.read_parquet(clv_path)
        assert len(df) == 1
        assert "clv_percentage" in df.columns
        assert df["clv_percentage"][0] is not None

    @pytest.mark.asyncio
    async def test_corners_pick_writes_unsupported_row(self, tmp_path: Path):
        """A SENT corners pick → clv.parquet row with clv_percentage=null + reason."""
        date_iso = "2026-05-15"
        part_dir = tmp_path / f"dt={date_iso}"
        part_dir.mkdir(parents=True)
        rows = [_corners_delivery_row()]
        pl.DataFrame(rows).write_parquet(part_dir / "deliveries.parquet")

        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        records, written = await run_clv_sink_for_delivered_picks(
            date_iso=date_iso,
            odds_client=client,
            fixture_metadata=FIXTURE_META,
            shadow_root=tmp_path,
            sport_key="soccer_epl",
        )

        assert len(records) == 1
        assert records[0].clv_percentage is None
        assert records[0].reason == "clv_unsupported_family"

        clv_path = part_dir / "clv.parquet"
        assert clv_path.exists()
        df = pl.read_parquet(clv_path)
        assert df["reason"][0] == "clv_unsupported_family"
        assert df["clv_percentage"][0] is None

    @pytest.mark.asyncio
    async def test_skipped_delivery_excluded(self, tmp_path: Path):
        """A skipped/failed delivery is NOT processed — only send_result='sent'."""
        date_iso = "2026-05-15"
        part_dir = tmp_path / f"dt={date_iso}"
        part_dir.mkdir(parents=True)
        # One sent, one skipped, one failed
        rows = [
            _goals_delivery_row(send_result="sent"),
            _goals_delivery_row(send_result="skipped"),
            _goals_delivery_row(send_result="failed"),
        ]
        pl.DataFrame(rows).write_parquet(part_dir / "deliveries.parquet")

        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        records, _ = await run_clv_sink_for_delivered_picks(
            date_iso=date_iso,
            odds_client=client,
            fixture_metadata=FIXTURE_META,
            shadow_root=tmp_path,
            sport_key="soccer_epl",
        )

        # Only the sent row should be processed
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_no_deliveries_parquet_returns_empty(self, tmp_path: Path):
        """When deliveries.parquet is absent, returns empty lists."""
        client = _mock_odds_client(EVENT_ID, _PINNACLE_TOTALS)
        records, written = await run_clv_sink_for_delivered_picks(
            date_iso="2026-05-15",
            odds_client=client,
            fixture_metadata=FIXTURE_META,
            shadow_root=tmp_path,
        )
        assert records == []
        assert written == {}


# ──────────────────────────────────────────────────────────────────────
# write_clv_records — partition correctness
# ──────────────────────────────────────────────────────────────────────


class TestWriteClvRecords:
    def test_writes_single_partition(self, tmp_path: Path):
        ts = datetime(2026, 5, 15, 20, 0, 0, tzinfo=UTC)
        rec = V3ClvRecord(
            fixture_id=1,
            thesis_id="t1",
            family="goals",
            market_id="match_goals_over_2.5",
            direction="over",
            bookmaker_odd=2.05,
            delivery_ts=ts,
            odds_fetched_at=ts,
            clv_percentage=3.5,
            reason=None,
        )
        written = write_clv_records([rec], shadow_root=tmp_path)
        assert "dt=2026-05-15" in written
        path = written["dt=2026-05-15"]
        assert path.exists()
        df = pl.read_parquet(path)
        assert len(df) == 1
        assert abs(df["clv_percentage"][0] - 3.5) < 1e-6

    def test_appends_to_existing_parquet(self, tmp_path: Path):
        ts = datetime(2026, 5, 15, 20, 0, 0, tzinfo=UTC)
        rec1 = V3ClvRecord(
            fixture_id=1, thesis_id="t1", family="goals",
            market_id="match_goals_over_2.5", direction="over",
            bookmaker_odd=2.05, delivery_ts=ts, odds_fetched_at=ts,
            clv_percentage=3.5, reason=None,
        )
        write_clv_records([rec1], shadow_root=tmp_path)
        rec2 = V3ClvRecord(
            fixture_id=2, thesis_id="t2", family="btts",
            market_id="btts_yes", direction="yes",
            bookmaker_odd=1.85, delivery_ts=ts, odds_fetched_at=ts,
            clv_percentage=1.2, reason=None,
        )
        write_clv_records([rec2], shadow_root=tmp_path)
        path = tmp_path / "dt=2026-05-15" / "clv.parquet"
        df = pl.read_parquet(path)
        assert len(df) == 2

    def test_unsupported_family_row_written_with_null_clv(self, tmp_path: Path):
        ts = datetime(2026, 5, 15, 20, 0, 0, tzinfo=UTC)
        rec = V3ClvRecord(
            fixture_id=1, thesis_id="t-corners", family="corners",
            market_id="match_corners_over_9.5", direction="over",
            bookmaker_odd=1.90, delivery_ts=ts, odds_fetched_at=None,
            clv_percentage=None, reason="clv_unsupported_family",
        )
        write_clv_records([rec], shadow_root=tmp_path)
        df = pl.read_parquet(tmp_path / "dt=2026-05-15" / "clv.parquet")
        assert df["clv_percentage"][0] is None
        assert df["reason"][0] == "clv_unsupported_family"


# ──────────────────────────────────────────────────────────────────────
# rolling_clv_from_parquet
# ──────────────────────────────────────────────────────────────────────


class TestRollingClvFromParquet:
    def test_returns_none_when_no_data(self, tmp_path: Path):
        assert rolling_clv_from_parquet(shadow_root=tmp_path) is None

    def test_computes_average_from_parquet(self, tmp_path: Path):
        ts = datetime(2026, 5, 15, 20, 0, 0, tzinfo=UTC)
        records = [
            V3ClvRecord(
                fixture_id=i, thesis_id=f"t{i}", family="goals",
                market_id="m", direction="over", bookmaker_odd=2.0,
                delivery_ts=ts, odds_fetched_at=ts,
                clv_percentage=float(i), reason=None,
            )
            for i in range(1, 6)  # CLV values 1, 2, 3, 4, 5
        ]
        write_clv_records(records, shadow_root=tmp_path)
        avg = rolling_clv_from_parquet(shadow_root=tmp_path)
        assert avg is not None
        assert abs(avg - 3.0) < 1e-6  # mean of [1,2,3,4,5] = 3.0

    def test_excludes_null_clv_rows(self, tmp_path: Path):
        ts = datetime(2026, 5, 15, 20, 0, 0, tzinfo=UTC)
        records = [
            V3ClvRecord(
                fixture_id=1, thesis_id="t1", family="goals",
                market_id="m", direction="over", bookmaker_odd=2.0,
                delivery_ts=ts, odds_fetched_at=ts, clv_percentage=4.0, reason=None,
            ),
            V3ClvRecord(
                fixture_id=2, thesis_id="t2", family="corners",
                market_id="m2", direction="over", bookmaker_odd=1.9,
                delivery_ts=ts, odds_fetched_at=None, clv_percentage=None,
                reason="clv_unsupported_family",
            ),
        ]
        write_clv_records(records, shadow_root=tmp_path)
        avg = rolling_clv_from_parquet(shadow_root=tmp_path)
        assert avg is not None
        assert abs(avg - 4.0) < 1e-6  # only the goals row counted
