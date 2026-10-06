"""Tests for backtest runner — synthetic data drives the pipeline."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.backtest.runner import (
    HistoricalFixture,
    render_markdown_report,
    run_backtest,
)
from bip.evaluation.tournaments.team_style_profiler.bettable_profile import (
    BettableProfile,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    DistributionStat,
)


def _d(mean: float, width: float = 0.4, n: int = 20) -> DistributionStat:
    return DistributionStat(
        mean=mean, ci_low=mean - width / 2, ci_high=mean + width / 2, n=n
    )


def _profile(
    name: str, conf: str = "UEFA", gf: float = 1.5, ga: float = 1.0,
    btts: float = 0.55, over_25: float = 0.55,
    corners_for: float = 6.0, corners_against: float = 4.0,
    yellow: float = 2.0,
) -> BettableProfile:
    return BettableProfile(
        team_name=name, confederation=conf, source="own_tsv",  # type: ignore[arg-type]
        goals_for_per_match=_d(gf),
        goals_against_per_match=_d(ga),
        btts_rate=_d(btts),
        over_25_rate=_d(over_25),
        over_35_rate=_d(0.30),
        corners_for_per_match=_d(corners_for),
        corners_against_per_match=_d(corners_against),
        yellow_cards_per_match=_d(yellow),
        fouls_per_match=_d(12.0),
        mean_total_goals=_d(gf + ga),
        possession_avg=_d(50.0),
        sub_profile_vs_opponent=None,
    )


def _fixture(
    fid: int, hg: int, ag: int,
    h_profile: BettableProfile | None = None,
    a_profile: BettableProfile | None = None,
    yc: int | None = None, corners: int | None = None,
) -> HistoricalFixture:
    return HistoricalFixture(
        fixture_id=fid,
        home_profile=h_profile or _profile(f"H{fid}"),
        away_profile=a_profile or _profile(f"A{fid}"),
        home_goals=hg, away_goals=ag,
        total_yellow_cards=yc, total_corners=corners,
    )


class TestRunBacktest:
    def test_empty_fixtures(self) -> None:
        m = run_backtest([])
        assert m.n_fixtures == 0
        assert m.n_with_predictions == 0
        assert m.market_metrics == {}

    def test_basic_coverage(self) -> None:
        fixtures = [_fixture(i, 1, 1) for i in range(10)]
        m = run_backtest(fixtures)
        assert m.n_fixtures == 10
        assert m.n_with_predictions == 10
        assert m.coverage_pct == 100.0

    def test_missing_profile_skipped(self) -> None:
        # 5 fixtures with both profiles, 5 with home missing
        ok = [_fixture(i, 1, 1) for i in range(5)]
        broken = [
            HistoricalFixture(
                fixture_id=100 + i, home_profile=None,
                away_profile=_profile(f"A{i}"), home_goals=1, away_goals=1,
            )
            for i in range(5)
        ]
        m = run_backtest(ok + broken)
        assert m.n_fixtures == 10
        assert m.n_with_predictions == 5

    def test_btts_market_aggregates(self) -> None:
        # 10 fixtures where outcome is BTTS=yes (everyone scores 1-1)
        fixtures = [_fixture(i, 1, 1) for i in range(10)]
        m = run_backtest(fixtures)
        btts = m.market_metrics["BTTS_yes"]
        # All outcomes positive
        assert btts.mean_outcome == 1.0
        # Model predicts ~0.55 (per profile btts_rate); Brier = (0.55 - 1)^2 ~ 0.20
        assert btts.brier > 0.15
        assert btts.brier < 0.25

    def test_over25_market(self) -> None:
        # 5 fixtures with O2.5 outcome, 5 with U2.5
        fixtures = (
            [_fixture(i, 2, 1) for i in range(5)]      # 3 goals -> O2.5=1
            + [_fixture(i + 100, 1, 0) for i in range(5)]  # 1 goal -> O2.5=0
        )
        m = run_backtest(fixtures)
        o25 = m.market_metrics["O2.5"]
        assert o25.n == 10
        assert o25.mean_outcome == 0.5

    def test_corners_market_skipped_without_data(self) -> None:
        fixtures = [_fixture(i, 1, 1) for i in range(10)]  # no corners
        m = run_backtest(fixtures)
        # corners markets shouldn't appear since outcome can't be computed
        assert "corners_O9.5" not in m.market_metrics

    def test_corners_market_with_data(self) -> None:
        fixtures = [_fixture(i, 1, 1, corners=12) for i in range(10)]
        m = run_backtest(fixtures)
        # Profile has corners_for=6, corners_against=4; data exists
        assert "corners_O9.5" in m.market_metrics
        # All 12 > 9 -> outcome=1 for all
        assert m.market_metrics["corners_O9.5"].mean_outcome == 1.0


class TestRenderMarkdown:
    def test_renders_table(self) -> None:
        fixtures = [_fixture(i, 1, 1, yc=4) for i in range(10)]
        m = run_backtest(fixtures)
        report = render_markdown_report(m, title="Test Report")
        assert "Test Report" in report
        assert "BTTS_yes" in report
        assert "| Market |" in report
        assert "Coverage:" in report
