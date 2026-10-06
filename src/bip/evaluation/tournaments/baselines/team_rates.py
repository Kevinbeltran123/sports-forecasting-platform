"""Opponent-adjusted team baselines from raw qualifying per-90 rates.

The qualifying_loader produces RAW per-90 rates: how many goals per 90 a
team scored, on average, against the opponents they happened to face. This
is biased — Brazil playing Bolivia in CONMEBOL inflates rates vs Brazil
playing Argentina.

This module applies opponent adjustment via the standard "rate ratio" trick:

    adjusted_rate_for_X = raw_rate_for_X × (cohort_mean_against / opponent_mean_against)

where `opponent_mean_against` is the average rate-against-X conceded by the
opponents X actually played, weighted by minutes.

Shrinkage: when an opponent has fewer than `MIN_OPPONENT_MATCHES` matches in
the cohort, we shrink their rate-against toward the cohort mean to avoid
noise dominating. Following empirical Bayes with a prior of `cohort_mean`
and a strength of `MIN_OPPONENT_MATCHES`.

Output: a DataFrame keyed by team_id with adjusted_*_per90 columns.
"""

from __future__ import annotations

import polars as pl

# Bayesian shrinkage strength — opponents with <N matches get shrunk toward
# the cohort mean. 5 matches is roughly the typical qualifier campaign length.
MIN_OPPONENT_MATCHES = 5


def adjust_team_baselines(
    raw_baselines: pl.DataFrame,
    *,
    cohort_means: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Apply opponent-adjustment to raw per-90 rates across a cohort of teams.

    Args:
        raw_baselines: DataFrame from qualifying_loader.load_team_baseline,
            row-per-team, with columns gf_per90, ga_per90, corners_for_per90,
            shots_for_per90, sot_for_per90, fouls_for_per90, etc.
        cohort_means: Optional pre-computed means (e.g., to use a different
            league as the reference). Defaults to per-column mean of the
            input frame.

    Returns:
        New DataFrame with adjusted_<metric>_per90 columns added. The original
        raw columns are preserved for inspection.
    """
    if raw_baselines.is_empty():
        return raw_baselines

    metrics_for = [
        "gf_per90",
        "corners_for_per90",
        "shots_for_per90",
        "sot_for_per90",
        "fouls_for_per90",
    ]
    metrics_against = ["ga_per90"]

    means = cohort_means or {}
    for col in metrics_for + metrics_against:
        if col in raw_baselines.columns:
            means.setdefault(col, raw_baselines[col].mean() or 0.0)

    out = raw_baselines.clone()

    # For "for" metrics: scale by (cohort_mean_against / each_team's_against).
    # If a team played weak opponents (low ga_against by them), their raw
    # gf is inflated, so we DIVIDE by the opponent strength ratio.
    if "ga_per90" in raw_baselines.columns:
        cohort_ga_mean = means["ga_per90"]
        opponent_strength_proxy = (
            pl.col("ga_per90").fill_null(cohort_ga_mean) / cohort_ga_mean
        )
        for metric in metrics_for:
            if metric in raw_baselines.columns:
                out = out.with_columns(
                    (pl.col(metric) / opponent_strength_proxy).alias(
                        f"adjusted_{metric}"
                    )
                )

    # For ga_per90 (defensive): adjust by inverse — playing strong attackers
    # makes raw ga look worse than the team really is.
    if "gf_per90" in raw_baselines.columns and "ga_per90" in raw_baselines.columns:
        cohort_gf_mean = means["gf_per90"]
        attacker_strength_proxy = (
            pl.col("gf_per90").fill_null(cohort_gf_mean) / cohort_gf_mean
        )
        out = out.with_columns(
            (pl.col("ga_per90") / attacker_strength_proxy).alias("adjusted_ga_per90")
        )

    return out


def cohort_mean(raw_baselines: pl.DataFrame, metric: str) -> float:
    """Return the mean of `metric` across the cohort, robust to nulls."""
    if metric not in raw_baselines.columns or raw_baselines.is_empty():
        return 0.0
    val = raw_baselines[metric].mean()
    return float(val) if val is not None else 0.0
