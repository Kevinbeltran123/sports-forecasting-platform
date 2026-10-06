"""Walk-forward backtest harness for tournament predictions.

The harness replays a list of historical fixtures with known outcomes:

    for fixture in historical:
        prediction = runner.predict(...point-in-time inputs...)
        record_observed(prediction, fixture.actual_result)

Inputs are passed in fully resolved (pre-loaded baselines, lineup forms,
etc.) — the harness does NOT load data. This keeps it pure and trivially
unit-testable: the heavy I/O lives in the CLI script that builds the
input list.

Output: BacktestReport with per-model metrics across markets, plus a
JSON-serializable shape for storage.

Why "walk-forward" rather than simple backtest:
- For each historical match M, the inputs (team baselines, player form)
  must be the data that was AVAILABLE at the time of M's kickoff,
  NOT including M itself or any subsequent match. The CLI script
  constructs these point-in-time snapshots; the harness just iterates.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from bip.evaluation.tournaments.backtest.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    mae,
    multiclass_brier_score,
    multiclass_log_loss,
    poisson_deviance,
)
from bip.evaluation.tournaments.predict.match_runner import (
    MatchPredictionRunner,
    TeamMatchInputs,
)
from bip.evaluation.tournaments.predict.output import MatchPrediction

# Backtest replay does not care about kickoff time; we use a fixed sentinel
# so per-match predictions stay deterministic across runs.
_BACKTEST_KICKOFF_SENTINEL = datetime(2000, 1, 1)


@dataclass(frozen=True)
class HistoricalFixture:
    """A historical match with known outcome — input to the backtest harness.

    actual_outcome_1x2 is the observed result class:
        0 = home win, 1 = draw, 2 = away win.
    """

    match_id: str
    tournament_slug: str
    home_inputs: TeamMatchInputs
    away_inputs: TeamMatchInputs
    actual_home_goals: int
    actual_away_goals: int
    actual_home_corners: int | None = None
    actual_away_corners: int | None = None

    @property
    def actual_outcome_1x2(self) -> int:
        if self.actual_home_goals > self.actual_away_goals:
            return 0
        if self.actual_home_goals < self.actual_away_goals:
            return 2
        return 1

    @property
    def actual_btts(self) -> int:
        return int(self.actual_home_goals > 0 and self.actual_away_goals > 0)

    @property
    def actual_total_goals(self) -> int:
        return self.actual_home_goals + self.actual_away_goals

    @property
    def actual_over_2_5(self) -> int:
        return int(self.actual_total_goals >= 3)


@dataclass(frozen=True)
class BacktestReport:
    """Per-model metrics across the historical sample."""

    model_name: str
    n_matches: int

    # 1X2
    multiclass_log_loss: float
    multiclass_brier: float

    # BTTS
    btts_log_loss: float
    btts_brier: float
    btts_ece: float

    # Over/Under 2.5
    over25_log_loss: float
    over25_brier: float
    over25_ece: float

    # Goals (count)
    home_goals_mae: float
    away_goals_mae: float
    home_goals_poisson_dev: float
    away_goals_poisson_dev: float

    # Per-prediction trace (optional, for diagnostics)
    per_match: tuple[dict, ...] = field(default_factory=tuple)


def run_backtest(
    fixtures: Sequence[HistoricalFixture],
    runner: MatchPredictionRunner,
    *,
    keep_per_match: bool = False,
) -> BacktestReport:
    """Replay the fixtures through the runner and aggregate metrics.

    Args:
        fixtures: Historical fixtures with point-in-time inputs already attached.
        runner: A MatchPredictionRunner configured with one of the GoalsModel
            implementations (Independent, Bivariate, ELO+Logistic).
        keep_per_match: If True, records per-fixture predictions in the report
            for downstream inspection. Off by default to keep reports small.

    Returns:
        BacktestReport with all the metrics computed.
    """
    if not fixtures:
        raise ValueError("run_backtest requires at least 1 fixture")

    predictions: list[MatchPrediction] = []
    for fx in fixtures:
        pred = runner.predict(
            match_id=fx.match_id,
            tournament_slug=fx.tournament_slug,
            kickoff=_BACKTEST_KICKOFF_SENTINEL,
            home=fx.home_inputs,
            away=fx.away_inputs,
        )
        predictions.append(pred)

    # Collect per-market arrays.
    onextwo_probs = [
        (p.goals.p_home_win, p.goals.p_draw, p.goals.p_away_win) for p in predictions
    ]
    onextwo_outcomes = [fx.actual_outcome_1x2 for fx in fixtures]

    btts_probs = [p.goals.p_btts for p in predictions]
    btts_outcomes = [fx.actual_btts for fx in fixtures]

    over25_probs = [p.goals.p_over_2_5 for p in predictions]
    over25_outcomes = [fx.actual_over_2_5 for fx in fixtures]

    home_lambdas = [p.goals.lambda_home for p in predictions]
    away_lambdas = [p.goals.lambda_away for p in predictions]
    home_actual = [fx.actual_home_goals for fx in fixtures]
    away_actual = [fx.actual_away_goals for fx in fixtures]

    per_match: list[dict] = []
    if keep_per_match:
        for p, fx in zip(predictions, fixtures):
            per_match.append(
                {
                    "match_id": fx.match_id,
                    "lambda_home": p.goals.lambda_home,
                    "lambda_away": p.goals.lambda_away,
                    "actual_home_goals": fx.actual_home_goals,
                    "actual_away_goals": fx.actual_away_goals,
                    "p_home_win": p.goals.p_home_win,
                    "p_draw": p.goals.p_draw,
                    "p_away_win": p.goals.p_away_win,
                    "p_btts": p.goals.p_btts,
                    "p_over_2_5": p.goals.p_over_2_5,
                    "actual_1x2": fx.actual_outcome_1x2,
                    "actual_btts": fx.actual_btts,
                    "actual_over_2_5": fx.actual_over_2_5,
                }
            )

    # `runner` exposes the model_name via the goals_model.name attribute;
    # we read it directly because the runner does not (yet) re-publish it.
    model_name = predictions[0].goals.model_name

    return BacktestReport(
        model_name=model_name,
        n_matches=len(fixtures),
        multiclass_log_loss=multiclass_log_loss(onextwo_probs, onextwo_outcomes),
        multiclass_brier=multiclass_brier_score(onextwo_probs, onextwo_outcomes),
        btts_log_loss=log_loss(btts_probs, btts_outcomes),
        btts_brier=brier_score(btts_probs, btts_outcomes),
        btts_ece=expected_calibration_error(btts_probs, btts_outcomes),
        over25_log_loss=log_loss(over25_probs, over25_outcomes),
        over25_brier=brier_score(over25_probs, over25_outcomes),
        over25_ece=expected_calibration_error(over25_probs, over25_outcomes),
        home_goals_mae=mae(home_lambdas, home_actual),
        away_goals_mae=mae(away_lambdas, away_actual),
        home_goals_poisson_dev=poisson_deviance(home_lambdas, home_actual),
        away_goals_poisson_dev=poisson_deviance(away_lambdas, away_actual),
        per_match=tuple(per_match),
    )


def write_report(report: BacktestReport, path: Path) -> None:
    """Persist a BacktestReport as JSON for cross-model comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(report), f, indent=2)


def compare_reports(reports: Sequence[BacktestReport]) -> str:
    """Render a side-by-side Markdown table of multiple model reports."""
    if not reports:
        return "_No reports to compare._"

    headers = [r.model_name for r in reports]

    rows = [
        ("n_matches", [r.n_matches for r in reports]),
        ("1X2 log-loss", [r.multiclass_log_loss for r in reports]),
        ("1X2 Brier", [r.multiclass_brier for r in reports]),
        ("BTTS log-loss", [r.btts_log_loss for r in reports]),
        ("BTTS Brier", [r.btts_brier for r in reports]),
        ("BTTS ECE", [r.btts_ece for r in reports]),
        ("O2.5 log-loss", [r.over25_log_loss for r in reports]),
        ("O2.5 Brier", [r.over25_brier for r in reports]),
        ("O2.5 ECE", [r.over25_ece for r in reports]),
        ("home goals MAE", [r.home_goals_mae for r in reports]),
        ("away goals MAE", [r.away_goals_mae for r in reports]),
    ]

    lines = ["| Metric | " + " | ".join(headers) + " |",
             "|---" + "|---:" * len(headers) + "|"]
    for label, values in rows:
        formatted = [
            f"{v}" if isinstance(v, int) else f"{v:.4f}" for v in values
        ]
        lines.append(f"| {label} | " + " | ".join(formatted) + " |")
    return "\n".join(lines)
