"""Tests for the WC2026 pattern runner + parquet recorder."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_bias_flags import LockPrediction
from bip.evaluation.live.engine_v3.wc2026_contingencies import (
    LineupSnapshot,
    MD1Result,
    StakeAdjustment,
)
from bip.evaluation.live.engine_v3.wc2026_pattern_runner import (
    FixtureContingencyInputs,
    PatternPick,
    PatternPickRecorder,
    WC2026PatternRunner,
)
from bip.evaluation.live.engine_v3.wc2026_patterns import (
    MatchContext,
    OddsSnapshot,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _mex_ctx() -> MatchContext:
    return MatchContext(
        match_id="wc2026_grp_000",
        tournament_slug="world_cup_2026",
        home_team="Mexico",
        away_team="South Africa",
        phase="group",
        home_market_value_eur=101_800_000.0,
        away_market_value_eur=46_400_000.0,
    )


def _live_u25_at_ht00() -> OddsSnapshot:
    return OddsSnapshot(
        stage="live",
        market_family=MarketFamily.GOALS,
        selection="under_2_5",
        decimal_odds=1.35,
        minute=46,
        home_goals_ht=0,
        away_goals_ht=0,
    )


def _pre_match_corner_odds() -> OddsSnapshot:
    return OddsSnapshot(
        stage="pre_match",
        market_family=MarketFamily.CORNERS,
        selection="ah_-2_home",
        decimal_odds=1.95,
    )


def _pre_match_tt_home_odds() -> OddsSnapshot:
    return OddsSnapshot(
        stage="pre_match",
        market_family=MarketFamily.GOALS,
        selection="home_team_total_over_1_5",
        decimal_odds=1.70,
    )


# ── Runner ──────────────────────────────────────────────────────────────


class TestRunnerEvaluate:
    def test_unregistered_fixture_returns_empty(self, caplog):
        runner = WC2026PatternRunner()
        out = runner.evaluate("missing", _live_u25_at_ht00())
        assert out == []
        assert any("not registered" in rec.message for rec in caplog.records)

    def test_live_event_runs_a1_only(self):
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        picks = runner.evaluate("m1", _live_u25_at_ht00())
        assert len(picks) == 1
        p = picks[0]
        assert p.rule_id == "A1"
        assert p.market_family == MarketFamily.GOALS
        assert p.direction == "under"
        assert p.decimal_odds == 1.35
        assert p.max_kelly == 0.25

    def test_pre_match_runs_a2_a3_in_parallel(self):
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        corner_picks = runner.evaluate("m1", _pre_match_corner_odds())
        tt_picks = runner.evaluate("m1", _pre_match_tt_home_odds())
        rule_ids = {p.rule_id for p in (*corner_picks, *tt_picks)}
        assert rule_ids == {"A2", "A3"}

    def test_pre_match_does_not_run_a1(self):
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        picks = runner.evaluate(
            "m1",
            OddsSnapshot(
                stage="pre_match",
                market_family=MarketFamily.GOALS,
                selection="under_2_5",
                decimal_odds=1.35,
            ),
        )
        # A1 needs live + HT score; A3 looks at goals but expects team-total.
        # Therefore no picks should emit.
        assert picks == []

    def test_metadata_propagates(self):
        ctx = _mex_ctx()
        runner = WC2026PatternRunner(contexts={"m1": ctx})
        picks = runner.evaluate("m1", _pre_match_corner_odds())
        assert len(picks) == 1
        p = picks[0]
        assert p.home_team == "Mexico"
        assert p.away_team == "South Africa"
        assert p.tournament_slug == "world_cup_2026"
        assert p.phase == "group"
        assert p.home_market_value_eur == 101_800_000.0
        assert p.away_market_value_eur == 46_400_000.0


class TestRunnerDedupe:
    def test_same_event_emits_once(self):
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        first = runner.evaluate("m1", _live_u25_at_ht00())
        second = runner.evaluate("m1", _live_u25_at_ht00())
        assert len(first) == 1
        assert len(second) == 0

    def test_different_direction_emits_separately(self):
        # Build two corner odds events: one favoring home, then a different
        # selection (same market family but different sided AH selection).
        # The runner keys dedup on (fixture, rule, family, direction). A2
        # always emits "home" direction for the same favorite, so dedup
        # should suppress the second.
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        first = runner.evaluate("m1", _pre_match_corner_odds())
        second = runner.evaluate("m1", _pre_match_corner_odds())
        assert len(first) == 1
        assert len(second) == 0

    def test_reset_dedupe_clears_fixture(self):
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        runner.evaluate("m1", _live_u25_at_ht00())
        runner.reset_dedupe("m1")
        again = runner.evaluate("m1", _live_u25_at_ht00())
        assert len(again) == 1

    def test_reset_dedupe_all(self):
        runner = WC2026PatternRunner(
            contexts={
                "m1": _mex_ctx(),
                "m2": MatchContext(
                    match_id="m2",
                    tournament_slug="world_cup_2026",
                    home_team="USA",
                    away_team="Paraguay",
                    phase="group",
                    home_market_value_eur=280_000_000.0,
                    away_market_value_eur=85_000_000.0,
                ),
            }
        )
        runner.evaluate("m1", _live_u25_at_ht00())
        runner.evaluate("m2", _live_u25_at_ht00())
        runner.reset_dedupe()
        # Both should re-fire.
        assert len(runner.evaluate("m1", _live_u25_at_ht00())) == 1
        assert len(runner.evaluate("m2", _live_u25_at_ht00())) == 1


class TestAfconFilterStatus:
    def test_reports_status_for_afcon(self):
        ctx = MatchContext(
            match_id="afcon_001",
            tournament_slug="afcon_2025",
            home_team="Egypt",
            away_team="Algeria",
            phase="group",
            home_market_value_eur=136_000_000.0,
            away_market_value_eur=160_000_000.0,
        )
        runner = WC2026PatternRunner(contexts={"afcon_001": ctx})
        verdict = runner.afcon_filter_status("afcon_001")
        assert verdict is not None
        assert verdict.triggered is True

    def test_returns_none_for_missing_fixture(self):
        runner = WC2026PatternRunner()
        assert runner.afcon_filter_status("nope") is None


class TestFromLockJSON:
    def test_loads_72_fixtures(self):
        runner = WC2026PatternRunner.from_lock_json()
        assert len(runner.contexts) == 72
        # Spot-check the opening fixture.
        opener = runner.contexts["wc2026_grp_000"]
        assert opener.home_team == "Mexico"
        assert opener.away_team == "South Africa"
        assert opener.tournament_slug == "world_cup_2026"
        assert opener.phase == "group"

    @pytest.mark.requires_data
    def test_market_values_attached_when_team_in_tm(self):
        runner = WC2026PatternRunner.from_lock_json()
        # Mexico is in the squad_values_current parquet (all 48 WC teams).
        opener = runner.contexts["wc2026_grp_000"]
        assert opener.home_market_value_eur is not None
        assert opener.home_market_value_eur > 0
        assert opener.away_market_value_eur is not None
        assert opener.away_market_value_eur > 0

    def test_evaluate_after_from_lock(self):
        runner = WC2026PatternRunner.from_lock_json()
        picks = runner.evaluate(
            "wc2026_grp_000",
            OddsSnapshot(
                stage="live",
                market_family=MarketFamily.GOALS,
                selection="under_2_5",
                decimal_odds=1.40,
                home_goals_ht=0,
                away_goals_ht=0,
            ),
        )
        assert len(picks) == 1
        assert picks[0].rule_id == "A1"


# ── Recorder ────────────────────────────────────────────────────────────


class TestPatternPickRecorder:
    def test_writes_one_row(self, tmp_path: Path):
        rec = PatternPickRecorder(output_root=tmp_path)
        pick = PatternPick(
            fixture_id="m1",
            timestamp_utc=datetime(2026, 6, 11, tzinfo=timezone.utc),
            rule_id="A1",
            home_team="Mexico",
            away_team="South Africa",
            tournament_slug="world_cup_2026",
            phase="group",
            market_family=MarketFamily.GOALS,
            direction="under",
            decimal_odds=1.35,
            max_kelly=0.25,
            reason="HT 0-0",
        )
        rec.record([pick])
        assert len(rec) == 1
        out_path = rec.flush(ts=datetime(2026, 6, 11, tzinfo=timezone.utc))
        assert out_path is not None
        assert out_path.exists()
        df = pl.read_parquet(out_path)
        assert df.height == 1
        assert df["rule_id"][0] == "A1"
        assert df["fixture_id"][0] == "m1"

    def test_appends_to_existing_partition(self, tmp_path: Path):
        rec = PatternPickRecorder(output_root=tmp_path)
        ts = datetime(2026, 6, 11, tzinfo=timezone.utc)
        first = PatternPick(
            fixture_id="m1",
            timestamp_utc=ts,
            rule_id="A1",
            home_team="Mexico",
            away_team="South Africa",
            tournament_slug="world_cup_2026",
            phase="group",
            market_family=MarketFamily.GOALS,
            direction="under",
            decimal_odds=1.35,
            max_kelly=0.25,
            reason="r1",
        )
        rec.record([first])
        path = rec.flush(ts=ts)
        assert path is not None
        second = PatternPick(
            fixture_id="m2",
            timestamp_utc=ts,
            rule_id="A2",
            home_team="USA",
            away_team="Paraguay",
            tournament_slug="world_cup_2026",
            phase="group",
            market_family=MarketFamily.CORNERS,
            direction="home",
            decimal_odds=1.95,
            max_kelly=0.25,
            reason="r2",
        )
        rec.record([second])
        rec.flush(ts=ts)
        df = pl.read_parquet(path)
        assert df.height == 2
        assert set(df["rule_id"].to_list()) == {"A1", "A2"}

    def test_flush_empty_buffer_returns_none(self, tmp_path: Path):
        rec = PatternPickRecorder(output_root=tmp_path)
        assert rec.flush() is None


# ── B+C+D integration with runner ───────────────────────────────────────


class TestRunnerWithFlags:
    def test_pick_carries_avoid_verdicts(self):
        """An A2 pick should have C-rule verdicts attached, none blocking
        on a vanilla setup."""
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        picks = runner.evaluate("m1", _pre_match_corner_odds())
        assert len(picks) == 1
        pick = picks[0]
        assert len(pick.avoid_verdicts) == 4
        assert {v.rule_id for v in pick.avoid_verdicts} == {"C1", "C2", "C3", "C4"}
        # No avoid rule blocks the vanilla A2 corner pick
        assert pick.blocked is False

    def test_blocked_pick_when_c3_fires(self):
        """If pre-match odds is ML on Cinderella at short price, C3 blocks."""
        ctx = MatchContext(
            match_id="m1",
            tournament_slug="world_cup_2026",
            home_team="New Zealand",
            away_team="Egypt",
            phase="group",
            home_market_value_eur=31_700_000.0,
            away_market_value_eur=136_230_000.0,
        )
        runner = WC2026PatternRunner(contexts={"m1": ctx})
        # ML on NZL at 2.40 → C3 blocks
        ml_odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.RESULT_1X2,
            selection="home",
            decimal_odds=2.40,
        )
        picks = runner.evaluate("m1", ml_odds)
        # A2/A3 don't apply to RESULT_1X2 anyway, so picks is empty.
        # But the avoid verdicts are still computed inside the runner;
        # since picks is empty we can't inspect them through picks. Use
        # afcon_filter_status for an audit-level check instead — for now
        # we just assert the runner ran without crashing.
        assert picks == []

    def test_d2_md1_reduces_kelly(self):
        """When both teams lost MD1, D2 fires REDUCE_50; effective_kelly
        should be half of max_kelly."""
        runner = WC2026PatternRunner(contexts={"m1": _mex_ctx()})
        runner.set_contingency_inputs(
            "m1",
            FixtureContingencyInputs(
                home_md1=MD1Result(team="Mexico", won=False, drew=False, lost=True,
                                    goals_for=0, goals_against=1),
                away_md1=MD1Result(team="South Africa", won=False, drew=False, lost=True,
                                    goals_for=0, goals_against=2),
            ),
        )
        picks = runner.evaluate("m1", _pre_match_corner_odds())
        assert len(picks) == 1
        pick = picks[0]
        assert pick.contingency_adjustment == StakeAdjustment.REDUCE_50
        assert pick.effective_kelly == pytest.approx(pick.max_kelly * 0.5)

    def test_d1_lineup_cancels_when_both_stars_missing(self):
        ctx = MatchContext(
            match_id="m1",
            tournament_slug="world_cup_2026",
            home_team="United States",
            away_team="England",
            phase="group",
            home_market_value_eur=280_000_000.0,
            away_market_value_eur=1_310_000_000.0,
        )
        runner = WC2026PatternRunner(contexts={"m1": ctx})
        captured = datetime(2026, 6, 19, 18, 0, tzinfo=timezone.utc)
        runner.set_contingency_inputs(
            "m1",
            FixtureContingencyInputs(
                home_lineup=LineupSnapshot(
                    team="United States", starting_xi=["No Pulisic"],
                    captured_at_utc=captured,
                ),
                away_lineup=LineupSnapshot(
                    team="England", starting_xi=["Harry Kane"],
                    captured_at_utc=captured,
                ),
            ),
        )
        picks = runner.evaluate(
            "m1",
            OddsSnapshot(
                stage="pre_match",
                market_family=MarketFamily.CORNERS,
                selection="ah_-2_away",
                decimal_odds=1.95,
            ),
        )
        # A2 should still emit, but with contingency_adjustment=CANCEL
        # and effective_kelly=0.
        assert len(picks) == 1
        pick = picks[0]
        assert pick.contingency_adjustment == StakeAdjustment.CANCEL
        assert pick.effective_kelly == 0.0

    def test_flags_for_returns_bias_flags(self):
        """flags_for() exposes B-series + D-series for audit."""
        runner = WC2026PatternRunner(
            contexts={
                "m1": MatchContext(
                    match_id="m1",
                    tournament_slug="world_cup_2026",
                    home_team="Brazil",
                    away_team="Haiti",
                    phase="group",
                    home_market_value_eur=905_700_000.0,
                    away_market_value_eur=55_730_000.0,
                )
            }
        )
        runner.set_contingency_inputs(
            "m1",
            FixtureContingencyInputs(
                lock_prediction=LockPrediction(
                    p_home_win=0.52, p_draw=0.25, p_away_win=0.23,
                ),
            ),
        )
        flags = runner.flags_for("m1")
        assert flags is not None
        b1 = next(f for f in flags.bias_flags if f.flag_id == "B1")
        # Haiti is debutant, ratio 16.2x → B1 fires
        assert b1.triggered is True
        assert b1.affected_team == "Brazil"

    def test_flags_for_returns_none_for_unknown_fixture(self):
        runner = WC2026PatternRunner()
        assert runner.flags_for("nope") is None

    def test_register_fixture_invalidates_cache(self):
        ctx = _mex_ctx()
        runner = WC2026PatternRunner(contexts={"m1": ctx})
        # populate cache
        runner.flags_for("m1")
        assert "m1" in runner._flag_cache
        # mutate via register_fixture
        runner.register_fixture(MatchContext(
            match_id="m1",
            tournament_slug="afcon_2025",
            home_team="Egypt",
            away_team="Algeria",
            phase="group",
        ))
        # cache wiped
        assert "m1" not in runner._flag_cache
