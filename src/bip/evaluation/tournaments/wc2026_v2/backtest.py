"""Walk-forward backtest harness + bootstrap CI for the v2 predictor.

Walk-forward methodology (matching the existing rolling-origin CV in the
locked v1 spike — commit 220cedb):

- For each of the four held-out tournaments, refit the v2 pipeline using
  ALL train + calibration data dated strictly before the tournament's
  start. Match-importance weighted MLE handles the recency weighting
  naturally; the cutoff just removes leakage from inside the tournament.
- Predict each match in the tournament with that one fit (the prior is
  not updated mid-tournament — same convention as the v1 spike).
- Accumulate per-fixture Brier scores across all four tournaments.
- Report per-tournament + overall metrics with bootstrap CI.

Bootstrap CI: resample fixtures with replacement n_resamples times, take
the 2.5th and 97.5th percentiles of the resample-mean Brier distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl

from .pipeline import V2PipelineResult, fit_v2_pipeline


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float

    def __repr__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def bootstrap_ci(
    per_fixture: np.ndarray, n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05
) -> BootstrapCI:
    """95% bootstrap CI on the mean of a per-fixture metric vector."""
    if per_fixture.size == 0:
        return BootstrapCI(point=float("nan"), lower=float("nan"), upper=float("nan"))
    rng = np.random.default_rng(seed)
    n = per_fixture.size
    point = float(per_fixture.mean())
    sample_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_means[i] = per_fixture[idx].mean()
    lo = float(np.quantile(sample_means, alpha / 2))
    hi = float(np.quantile(sample_means, 1 - alpha / 2))
    return BootstrapCI(point=point, lower=lo, upper=hi)


def _brier_1x2(p_home: float, p_draw: float, p_away: float, outcome: int) -> float:
    """Gate-scale multiclass Brier = (1/K) · Σ_k (p_k − 1{y=k})^2 with K=3.

    Matches the convention used by the existing lock emitter (see
    ``tests/evaluation/test_lock_gate.py``: "brier_for_gate divides by K").
    Lock baseline (3-tournament CV, n=135): 0.2156 [CI 0.2027–0.2290].
    Uniform 1X2 baseline = 2/9 ≈ 0.222.
    """

    target = [0.0, 0.0, 0.0]
    target[outcome] = 1.0
    sum_sq = (p_home - target[0]) ** 2 + (p_draw - target[1]) ** 2 + (p_away - target[2]) ** 2
    return sum_sq / 3.0


def _brier_binary(p: float, y: int) -> float:
    return (p - y) ** 2


def _ece_binary(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        conf = float(probs[mask].mean())
        acc = float(outcomes[mask].mean())
        ece += abs(conf - acc) * mask.sum() / n
    return float(ece)


def _ece_multiclass(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE — bin by argmax confidence, compare top-class accuracy."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == outcomes).astype(float)
    return _ece_binary(confidences, correct, n_bins=n_bins)


@dataclass
class TournamentBacktestMetrics:
    tournament_slug: str
    n_matches: int
    n_cold_start: int
    brier_1x2: BootstrapCI
    brier_btts: BootstrapCI
    brier_ou_2_5: BootstrapCI
    ece_1x2: float
    ece_btts: float
    ece_ou_2_5: float
    per_fixture_brier_1x2: np.ndarray = field(repr=False)


@dataclass
class V2BacktestResult:
    per_tournament: dict[str, TournamentBacktestMetrics]
    overall_brier_1x2: BootstrapCI
    overall_brier_btts: BootstrapCI
    overall_brier_ou_2_5: BootstrapCI
    overall_ece_1x2: float
    overall_ece_btts: float
    overall_ece_ou_2_5: float
    n_matches: int
    n_cold_start: int


def _evaluate_predictor_on_tournament(
    fit: V2PipelineResult,
    tournament_matches: pl.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> TournamentBacktestMetrics:
    rows = list(tournament_matches.iter_rows(named=True))
    n = len(rows)
    n_cold = 0

    brier_1x2 = np.empty(n, dtype=np.float64)
    brier_btts = np.empty(n, dtype=np.float64)
    brier_ou = np.empty(n, dtype=np.float64)

    probs_1x2 = np.empty((n, 3), dtype=np.float64)
    probs_btts = np.empty(n, dtype=np.float64)
    probs_ou = np.empty(n, dtype=np.float64)
    outcomes_1x2 = np.empty(n, dtype=np.int64)
    outcomes_btts = np.empty(n, dtype=np.int64)
    outcomes_ou = np.empty(n, dtype=np.int64)

    for i, row in enumerate(rows):
        pred = fit.predictor.predict(row["home_team"], row["away_team"])
        if pred.is_cold_start:
            n_cold += 1
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        outcome_1x2 = 0 if hg > ag else (1 if hg == ag else 2)
        outcome_btts = 1 if (hg >= 1 and ag >= 1) else 0
        outcome_ou = 1 if (hg + ag) >= 3 else 0

        brier_1x2[i] = _brier_1x2(pred.p_home_win, pred.p_draw, pred.p_away_win, outcome_1x2)
        brier_btts[i] = _brier_binary(pred.p_btts, outcome_btts)
        brier_ou[i] = _brier_binary(pred.p_over_2_5, outcome_ou)
        probs_1x2[i] = [pred.p_home_win, pred.p_draw, pred.p_away_win]
        probs_btts[i] = pred.p_btts
        probs_ou[i] = pred.p_over_2_5
        outcomes_1x2[i] = outcome_1x2
        outcomes_btts[i] = outcome_btts
        outcomes_ou[i] = outcome_ou

    return TournamentBacktestMetrics(
        tournament_slug=str(tournament_matches["tournament_slug"][0]),
        n_matches=n,
        n_cold_start=n_cold,
        brier_1x2=bootstrap_ci(brier_1x2, n_resamples=n_bootstrap, seed=seed),
        brier_btts=bootstrap_ci(brier_btts, n_resamples=n_bootstrap, seed=seed + 1),
        brier_ou_2_5=bootstrap_ci(brier_ou, n_resamples=n_bootstrap, seed=seed + 2),
        ece_1x2=_ece_multiclass(probs_1x2, outcomes_1x2),
        ece_btts=_ece_binary(probs_btts, outcomes_btts),
        ece_ou_2_5=_ece_binary(probs_ou, outcomes_ou),
        per_fixture_brier_1x2=brier_1x2,
    )


def run_walk_forward_backtest(
    train: pl.DataFrame,
    calibration: pl.DataFrame,
    heldout: pl.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    cutoff_buffer_days: int = 7,
    use_dibp: bool = True,
    use_beta_calibration: bool = True,
    use_match_importance: bool = True,
    use_market_value: bool = False,
    market_values_path: str | None = None,
    market_value_beta: float | None = None,
    market_offset_mode: str = "attack",
) -> V2BacktestResult:
    """Walk-forward across the four held-out tournaments.

    For each tournament, build a strict pre-tournament training corpus
    (train rows with date < tournament_start − ``cutoff_buffer_days``) +
    pre-tournament calibration rows, fit the pipeline, evaluate on the
    tournament's fixtures.
    """

    held_dates = (
        heldout.group_by("tournament_slug")
        .agg(pl.col("match_date").min().alias("start"))
        .sort("start")
    )

    per_tournament_metrics: dict[str, TournamentBacktestMetrics] = {}
    all_brier_1x2: list[np.ndarray] = []
    all_brier_btts: list[np.ndarray] = []
    all_brier_ou: list[np.ndarray] = []
    all_probs_1x2: list[np.ndarray] = []
    all_probs_btts: list[np.ndarray] = []
    all_probs_ou: list[np.ndarray] = []
    all_outcomes_1x2: list[np.ndarray] = []
    all_outcomes_btts: list[np.ndarray] = []
    all_outcomes_ou: list[np.ndarray] = []
    total_n = 0
    total_cold = 0

    for row in held_dates.iter_rows(named=True):
        slug = row["tournament_slug"]
        start: date = row["start"]
        cutoff = start - timedelta(days=cutoff_buffer_days)
        train_window = train.filter(pl.col("date") < pl.lit(cutoff))
        # Calibration corpus: WC2018 + Euro2020. Both end well before any
        # held-out tournament starts, so the full calibration set is
        # leakage-safe. Still gate it defensively.
        cal_window = calibration.filter(pl.col("match_date") < pl.lit(cutoff))

        pipeline_kwargs: dict = {
            "train": train_window,
            "calibration": cal_window,
            "reference_date": cutoff,
            "use_dibp": use_dibp,
            "use_beta_calibration": use_beta_calibration,
            "use_match_importance": use_match_importance,
            "use_market_value": use_market_value,
            "market_values_path": market_values_path,
            "market_offset_mode": market_offset_mode,
        }
        if market_value_beta is not None:
            pipeline_kwargs["market_value_beta"] = market_value_beta
        fit = fit_v2_pipeline(**pipeline_kwargs)
        tournament_matches = heldout.filter(pl.col("tournament_slug") == slug).sort("match_date")
        metrics = _evaluate_predictor_on_tournament(
            fit, tournament_matches, n_bootstrap=n_bootstrap, seed=seed
        )
        per_tournament_metrics[slug] = metrics
        all_brier_1x2.append(metrics.per_fixture_brier_1x2)
        total_n += metrics.n_matches
        total_cold += metrics.n_cold_start

        # Re-evaluate full prob/outcome arrays (cheap, we already did the
        # predict loop, but we want overall ECE — easiest is to recompute
        # outcomes here).
        per_probs_1x2 = np.empty((metrics.n_matches, 3), dtype=np.float64)
        per_probs_btts = np.empty(metrics.n_matches, dtype=np.float64)
        per_probs_ou = np.empty(metrics.n_matches, dtype=np.float64)
        per_outc_1x2 = np.empty(metrics.n_matches, dtype=np.int64)
        per_outc_btts = np.empty(metrics.n_matches, dtype=np.int64)
        per_outc_ou = np.empty(metrics.n_matches, dtype=np.int64)
        brier_btts = np.empty(metrics.n_matches, dtype=np.float64)
        brier_ou = np.empty(metrics.n_matches, dtype=np.float64)
        for i, m_row in enumerate(tournament_matches.iter_rows(named=True)):
            pred = fit.predictor.predict(m_row["home_team"], m_row["away_team"])
            hg, ag = int(m_row["home_goals"]), int(m_row["away_goals"])
            outcome_1x2 = 0 if hg > ag else (1 if hg == ag else 2)
            outcome_btts = 1 if (hg >= 1 and ag >= 1) else 0
            outcome_ou = 1 if (hg + ag) >= 3 else 0
            per_probs_1x2[i] = [pred.p_home_win, pred.p_draw, pred.p_away_win]
            per_probs_btts[i] = pred.p_btts
            per_probs_ou[i] = pred.p_over_2_5
            per_outc_1x2[i] = outcome_1x2
            per_outc_btts[i] = outcome_btts
            per_outc_ou[i] = outcome_ou
            brier_btts[i] = _brier_binary(pred.p_btts, outcome_btts)
            brier_ou[i] = _brier_binary(pred.p_over_2_5, outcome_ou)
        all_brier_btts.append(brier_btts)
        all_brier_ou.append(brier_ou)
        all_probs_1x2.append(per_probs_1x2)
        all_probs_btts.append(per_probs_btts)
        all_probs_ou.append(per_probs_ou)
        all_outcomes_1x2.append(per_outc_1x2)
        all_outcomes_btts.append(per_outc_btts)
        all_outcomes_ou.append(per_outc_ou)

    pooled_1x2 = np.concatenate(all_brier_1x2)
    pooled_btts = np.concatenate(all_brier_btts)
    pooled_ou = np.concatenate(all_brier_ou)
    pooled_probs_1x2 = np.concatenate(all_probs_1x2)
    pooled_probs_btts = np.concatenate(all_probs_btts)
    pooled_probs_ou = np.concatenate(all_probs_ou)
    pooled_outc_1x2 = np.concatenate(all_outcomes_1x2)
    pooled_outc_btts = np.concatenate(all_outcomes_btts)
    pooled_outc_ou = np.concatenate(all_outcomes_ou)

    return V2BacktestResult(
        per_tournament=per_tournament_metrics,
        overall_brier_1x2=bootstrap_ci(pooled_1x2, n_resamples=n_bootstrap, seed=seed + 100),
        overall_brier_btts=bootstrap_ci(pooled_btts, n_resamples=n_bootstrap, seed=seed + 101),
        overall_brier_ou_2_5=bootstrap_ci(pooled_ou, n_resamples=n_bootstrap, seed=seed + 102),
        overall_ece_1x2=_ece_multiclass(pooled_probs_1x2, pooled_outc_1x2),
        overall_ece_btts=_ece_binary(pooled_probs_btts, pooled_outc_btts),
        overall_ece_ou_2_5=_ece_binary(pooled_probs_ou, pooled_outc_ou),
        n_matches=total_n,
        n_cold_start=total_cold,
    )
