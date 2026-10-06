"""Backtest runner — orchestrates TSP predictions across historical fixtures.

LIMITATION: this runner takes pre-computed BettableProfiles and historical
fixture outcomes as input. It does NOT pull from API-Football directly
(that requires credentials + budget). The caller is expected to assemble
the input data structures from a separate ingest pass.

Workflow:
  1. For each historical fixture, the caller passes:
       - home_profile (BettableProfile)
       - away_profile (BettableProfile)
       - actual_home_goals, actual_away_goals
       - (optional) yellow_cards_total, corners_total
  2. Runner calls predict_markets() to get per-market probabilities
  3. Runner compares predictions to outcomes
  4. Produces BacktestMetrics + per-market Brier/ECE/hit_rate
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.backtest.metrics import (
    BacktestMetrics,
    MarketMetrics,
    brier_score,
    expected_calibration_error,
    hit_rate,
    market_passes_brier_gate,
)
from bip.evaluation.tournaments.team_style_profiler.bettable_profile import (
    BettableProfile,
)
from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    predict_markets,
)


@dataclass(frozen=True)
class HistoricalFixture:
    """A single fixture in the backtest set, with profiles + outcome."""

    fixture_id: int
    home_profile: BettableProfile | None
    away_profile: BettableProfile | None
    home_goals: int
    away_goals: int
    total_yellow_cards: int | None = None
    total_corners: int | None = None

    @property
    def is_predictable(self) -> bool:
        """True iff both teams have profiles (own_tsv or cohort)."""
        return self.home_profile is not None and self.away_profile is not None

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def btts(self) -> bool:
        return self.home_goals > 0 and self.away_goals > 0

    @property
    def home_minus_away(self) -> int:
        return self.home_goals - self.away_goals


def _market_outcome(market: str, fix: HistoricalFixture) -> int | None:
    """Compute the 0/1 outcome for a market from the fixture result.

    Returns None when the market's outcome can't be determined from
    available fixture data (e.g., corners not provided).
    """
    if market == "BTTS_yes":
        return 1 if fix.btts else 0
    if market == "BTTS_no":
        return 0 if fix.btts else 1
    if market == "O2.5":
        return 1 if fix.total_goals > 2 else 0
    if market == "U2.5":
        return 0 if fix.total_goals > 2 else 1
    if market == "O3.5":
        return 1 if fix.total_goals > 3 else 0
    if market == "corners_O9.5":
        if fix.total_corners is None:
            return None
        return 1 if fix.total_corners > 9 else 0
    if market == "corners_O8.5":
        if fix.total_corners is None:
            return None
        return 1 if fix.total_corners > 8 else 0
    if market == "corners_O10.5":
        if fix.total_corners is None:
            return None
        return 1 if fix.total_corners > 10 else 0
    if market == "cards_O3.5":
        if fix.total_yellow_cards is None:
            return None
        return 1 if fix.total_yellow_cards > 3 else 0
    if market == "cards_O4.5":
        if fix.total_yellow_cards is None:
            return None
        return 1 if fix.total_yellow_cards > 4 else 0
    if market == "cards_O5.5":
        if fix.total_yellow_cards is None:
            return None
        return 1 if fix.total_yellow_cards > 5 else 0
    if market == "AH_home_-0.5":
        return 1 if fix.home_minus_away > 0 else 0
    if market == "AH_home_-1.5":
        return 1 if fix.home_minus_away > 1 else 0
    if market == "AH_away_-0.5":
        return 1 if fix.home_minus_away < 0 else 0
    if market == "AH_away_-1.5":
        return 1 if fix.home_minus_away < -1 else 0
    return None


def run_backtest(
    fixtures: list[HistoricalFixture], rho: float = 0.0
) -> BacktestMetrics:
    """Compute per-market metrics across the fixture list.

    Skips fixtures where either profile is missing (counted toward
    coverage but not toward per-market stats).
    """
    n_total = len(fixtures)
    n_predicted = 0

    # market -> (list of probs, list of outcomes)
    market_data: dict[str, tuple[list[float], list[int]]] = {}

    for fix in fixtures:
        if not fix.is_predictable:
            continue
        n_predicted += 1
        predictions = predict_markets(
            fix.home_profile, fix.away_profile, rho=rho  # type: ignore[arg-type]
        )
        for mp in predictions.all_markets:
            outcome = _market_outcome(mp.market, fix)
            if outcome is None:
                continue
            if mp.market not in market_data:
                market_data[mp.market] = ([], [])
            market_data[mp.market][0].append(mp.probability)
            market_data[mp.market][1].append(outcome)

    # Aggregate per market
    market_metrics: dict[str, MarketMetrics] = {}
    for market, (probs_list, outcomes_list) in market_data.items():
        probs_arr = np.asarray(probs_list)
        outcomes_arr = np.asarray(outcomes_list)
        b = brier_score(probs_arr, outcomes_arr)
        market_metrics[market] = MarketMetrics(
            market=market,
            n=len(probs_arr),
            brier=b,
            ece=expected_calibration_error(probs_arr, outcomes_arr),
            hit_rate=hit_rate(probs_arr, outcomes_arr),
            mean_predicted=float(probs_arr.mean()),
            mean_outcome=float(outcomes_arr.mean()),
            passes_brier_gate=market_passes_brier_gate(market, b),
        )

    coverage = (n_predicted / n_total * 100.0) if n_total > 0 else 0.0

    return BacktestMetrics(
        n_fixtures=n_total,
        n_with_predictions=n_predicted,
        coverage_pct=coverage,
        market_metrics=market_metrics,
    )


def render_markdown_report(
    metrics: BacktestMetrics, title: str = "TSP Backtest Report"
) -> str:
    """Render BacktestMetrics as a markdown report string."""
    lines = [
        f"# {title}",
        "",
        f"- Total fixtures: {metrics.n_fixtures}",
        f"- With predictions: {metrics.n_with_predictions}",
        f"- Coverage: {metrics.coverage_pct:.1f}%",
        f"- All markets pass: **{metrics.all_markets_pass}**",
        "",
        "## Per-market metrics",
        "",
        "| Market | n | Brier | ECE | Hit rate | E[p] | E[y] | Verdict |",
        "|--------|---|-------|-----|----------|------|------|---------|",
    ]
    for market, mm in sorted(metrics.market_metrics.items()):
        verdict = "✅ PASS" if mm.passes_brier_gate else "❌ FAIL"
        lines.append(
            f"| {market} | {mm.n} | {mm.brier:.4f} | {mm.ece:.4f} | "
            f"{mm.hit_rate:.2%} | {mm.mean_predicted:.3f} | "
            f"{mm.mean_outcome:.3f} | {verdict} |"
        )
    return "\n".join(lines)
