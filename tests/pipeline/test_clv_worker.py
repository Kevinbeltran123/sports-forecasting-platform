"""ClvWorker tests — Sprint 2 Ola C.

Tests:
  - Layer-1 path (no odds client) enumerates and logs "would measure"
  - Layer-2 path with mock OddsClient + ResultClient measures CLV + grade
  - No candidate rows → zero summary
  - Closing odds missing → skip with reason
  - Result client missing → CLV recorded without result/pl_units
  - _grade_result correctness across 1x2, ou_*, btts
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from bip.pipeline.clv_worker import (
    ClvRunSummary,
    ClvWorker,
    _grade_result,
    _pl_units,
)
from bip.pipeline.orchestrator import PREDICTIONS_RAW_TABLE
from tests.pipeline.conftest import FakeSupabaseClient


def _sent_row(
    *,
    row_id: str = "row-1",
    fixture_id: str = "fx-1",
    market: str = "1x2",
    selection: str = "home",
    odds: float = 2.0,
    stake_units: float = 1.0,
    kickoff_offset_h: float = -3.0,
) -> dict:
    kickoff = datetime.now(UTC) + timedelta(hours=kickoff_offset_h)
    return {
        "id": row_id,
        "source": "ligas",
        "fixture_id": fixture_id,
        "competition": "PL",
        "home_team": "A",
        "away_team": "B",
        "match_datetime": kickoff.isoformat(),
        "market": market,
        "selection": selection,
        "odds_at_pick": odds,
        "stake_units": stake_units,
        "status": "sent",
        "clv": None,
    }


def _seed_sent(client: FakeSupabaseClient, rows: list[dict]) -> None:
    client.table(PREDICTIONS_RAW_TABLE).rows = list(rows)


@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


# ──────────────────────────────────────────────────────────────────────
# Grading + P/L unit tests
# ──────────────────────────────────────────────────────────────────────


class TestGrading:
    def test_1x2_home_win(self):
        assert _grade_result(
            market="1x2", selection="home",
            result={"home_goals": 2, "away_goals": 1},
        ) == "win"

    def test_1x2_home_loss(self):
        assert _grade_result(
            market="1x2", selection="home",
            result={"home_goals": 0, "away_goals": 1},
        ) == "loss"

    def test_1x2_draw(self):
        assert _grade_result(
            market="1x2", selection="draw",
            result={"home_goals": 1, "away_goals": 1},
        ) == "win"

    def test_ou_25_over_win(self):
        assert _grade_result(
            market="ou_2.5", selection="over",
            result={"home_goals": 2, "away_goals": 2},
        ) == "win"

    def test_ou_25_over_loss(self):
        assert _grade_result(
            market="ou_2.5", selection="over",
            result={"home_goals": 1, "away_goals": 1},
        ) == "loss"

    def test_ou_25_under_win(self):
        assert _grade_result(
            market="ou_2.5", selection="under",
            result={"home_goals": 0, "away_goals": 1},
        ) == "win"

    def test_btts_yes(self):
        assert _grade_result(
            market="btts", selection="yes",
            result={"home_goals": 1, "away_goals": 2},
        ) == "win"
        assert _grade_result(
            market="btts", selection="no",
            result={"home_goals": 3, "away_goals": 0},
        ) == "win"

    def test_missing_goals_returns_none(self):
        assert _grade_result(
            market="1x2", selection="home",
            result={},
        ) is None

    def test_pl_units_win_at_1_95(self):
        assert _pl_units(graded="win", stake_units=1.0, odds=1.95) == pytest.approx(0.95)

    def test_pl_units_loss(self):
        assert _pl_units(graded="loss", stake_units=2.0, odds=1.9) == -2.0

    def test_pl_units_void_zero(self):
        assert _pl_units(graded="push", stake_units=1.0, odds=1.95) == 0.0


# ──────────────────────────────────────────────────────────────────────
# Worker behavior
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestClvWorker:
    async def test_no_candidates_returns_zero(self, fake_client, now_utc):
        worker = ClvWorker(supabase_client=fake_client)
        summary = await worker.run_once(now=now_utc)
        assert summary == ClvRunSummary()

    async def test_layer1_logs_without_measuring(self, fake_client, now_utc):
        _seed_sent(
            fake_client,
            [
                _sent_row(row_id="r1", kickoff_offset_h=-5),
                _sent_row(row_id="r2", kickoff_offset_h=-3),
            ],
        )
        worker = ClvWorker(supabase_client=fake_client)
        summary = await worker.run_once(now=now_utc)
        assert summary.n_candidate_rows == 2
        # Layer-1 doesn't write back
        assert summary.n_measured == 0
        assert summary.skip_reasons.get("no_odds_client_layer1") == 2

    async def test_layer2_records_clv_and_result(self, fake_client, now_utc):
        # Vig-removed closing: home win probability higher → smaller closing odds
        # P_taken(home @ 2.5) = 0.40 ; vig-removed P_closing depends on full slate
        # Using a 3-way closing slate that yields clean CLV math
        closing = {"home": 2.0, "draw": 3.5, "away": 4.0}
        odds_client = AsyncMock()
        odds_client.fetch_closing_odds.return_value = closing
        result_client = AsyncMock()
        result_client.fetch_result.return_value = {"home_goals": 2, "away_goals": 1}

        _seed_sent(
            fake_client,
            [_sent_row(row_id="r1", market="1x2", selection="home", odds=2.5,
                       stake_units=1.0, kickoff_offset_h=-3)],
        )
        worker = ClvWorker(
            supabase_client=fake_client,
            odds_client=odds_client,
            result_client=result_client,
        )
        summary = await worker.run_once(now=now_utc)

        assert summary.n_measured == 1
        # UPDATE payload was recorded
        updates = fake_client.tables[PREDICTIONS_RAW_TABLE].updates
        assert len(updates) == 1
        _filt, payload = updates[0]
        assert "clv" in payload
        assert payload["result"] == "win"
        assert payload["pl_units"] == pytest.approx(1.5)  # 1u * (2.5 - 1.0)
        assert payload["closing_odds"] == 2.0

    async def test_layer2_no_closing_odds_skipped(self, fake_client, now_utc):
        odds_client = AsyncMock()
        odds_client.fetch_closing_odds.return_value = None

        _seed_sent(fake_client, [_sent_row(kickoff_offset_h=-3)])
        worker = ClvWorker(supabase_client=fake_client, odds_client=odds_client)
        summary = await worker.run_once(now=now_utc)
        assert summary.n_measured == 0
        assert summary.skip_reasons.get("no_closing_odds") == 1

    async def test_layer2_no_result_client_records_clv_only(self, fake_client, now_utc):
        closing = {"home": 2.0, "draw": 3.5, "away": 4.0}
        odds_client = AsyncMock()
        odds_client.fetch_closing_odds.return_value = closing

        _seed_sent(
            fake_client,
            [_sent_row(row_id="r1", market="1x2", selection="home", odds=2.5,
                       kickoff_offset_h=-3)],
        )
        worker = ClvWorker(supabase_client=fake_client, odds_client=odds_client)
        summary = await worker.run_once(now=now_utc)

        assert summary.n_measured == 1
        _filt, payload = fake_client.tables[PREDICTIONS_RAW_TABLE].updates[0]
        assert "clv" in payload
        assert "result" not in payload
        assert "pl_units" not in payload

    async def test_row_error_does_not_block_others(self, fake_client, now_utc):
        closing = {"home": 2.0, "draw": 3.5, "away": 4.0}
        odds_client = AsyncMock()

        # First row raises, second succeeds
        async def maybe_raise(*, fixture_id, market):
            if fixture_id == "fx-bad":
                raise RuntimeError("odds API timeout")
            return closing

        odds_client.fetch_closing_odds.side_effect = maybe_raise

        _seed_sent(
            fake_client,
            [
                _sent_row(row_id="r1", fixture_id="fx-bad", kickoff_offset_h=-3),
                _sent_row(row_id="r2", fixture_id="fx-good", market="1x2",
                          selection="home", odds=2.5, kickoff_offset_h=-3),
            ],
        )
        worker = ClvWorker(supabase_client=fake_client, odds_client=odds_client)
        summary = await worker.run_once(now=now_utc)
        assert summary.n_errors == 1
        assert summary.n_measured == 1
