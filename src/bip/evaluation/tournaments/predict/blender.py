"""Blend opponent-adjusted team baselines with lineup-aggregated player rates.

Core formula::

    expected_per90 = α · team_baseline + (1 − α) · lineup_aggregate

Where:
- α (alpha) controls the relative weight. Lower α emphasises recent player
  form; higher α anchors to qualifier team identity. Default 0.35 — operator
  may tune in Phase 4 backtest.
- team_baseline is from baselines.team_rates.adjust_team_baselines.
- lineup_aggregate is the SUM of normalized player per-90 rates across the
  predicted XI (defended by `players.recent_form.normalize_player_rates`).

Markets handled:
- goals (gf_per90 vs goals_per90 sum)
- shots / shots_on_target / fouls

Markets NOT yet handled here (Phase 2.3 / 3):
- corners (team-level only — players don't take "their own" corner rate)
- cards (referee-dominated, model in Phase 3)
- BTTS / Over-Under (derived from goals λ via Poisson — Phase 2.3 models)

The output is a per-team dict of expected per-90 rates, ready to be fed
into Poisson / BivariatePoisson / ELO models in Phase 2.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DEFAULT_ALPHA = 0.35


@dataclass(frozen=True)
class BlendedRates:
    """Per-team expected rates per 90 minutes after blending."""

    team_id: int
    expected_goals_per90: float
    expected_shots_per90: float
    expected_sot_per90: float
    expected_fouls_committed_per90: float
    expected_corners_for_per90: float  # team-only (no player decomposition)

    # Trace fields (helpful for debugging predictions)
    alpha: float
    team_component_goals: float
    lineup_component_goals: float
    n_starters_with_form: int


def blend_team_rates(
    team_baseline_row: pl.DataFrame,
    lineup_form: pl.DataFrame,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> BlendedRates:
    """Blend a single team's baseline with its lineup's aggregated player form.

    Args:
        team_baseline_row: One-row DataFrame from adjust_team_baselines for
            this team. Must contain `team_id`, `adjusted_gf_per90`,
            `adjusted_shots_for_per90`, `adjusted_sot_for_per90`,
            `adjusted_fouls_for_per90`, `adjusted_corners_for_per90`.
        lineup_form: 11-row DataFrame, one row per starter, with normalized
            per-90 rates from normalize_player_rates. Columns:
            `normalized_goals_per90`, `normalized_shots_per90`,
            `normalized_sot_per90`, `fouls_committed_per90`, `reliable`.
        alpha: Weight on team-baseline component. Range [0, 1]. Default 0.35.

    Returns:
        BlendedRates dataclass with expected per-90 totals.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if team_baseline_row.height != 1:
        raise ValueError(f"team_baseline_row must be exactly 1 row, got {team_baseline_row.height}")

    team_id = int(team_baseline_row["team_id"][0])

    team_goals = _safe_get(team_baseline_row, "adjusted_gf_per90", "gf_per90")
    team_shots = _safe_get(team_baseline_row, "adjusted_shots_for_per90", "shots_for_per90")
    team_sot = _safe_get(team_baseline_row, "adjusted_sot_for_per90", "sot_for_per90")
    team_fouls = _safe_get(team_baseline_row, "adjusted_fouls_for_per90", "fouls_for_per90")
    team_corners = _safe_get(
        team_baseline_row, "adjusted_corners_for_per90", "corners_for_per90"
    )

    lineup_goals = _sum_with_fallback(lineup_form, "normalized_goals_per90", "goals_per90")
    lineup_shots = _sum_with_fallback(lineup_form, "normalized_shots_per90", "shots_per90")
    lineup_sot = _sum_with_fallback(lineup_form, "normalized_sot_per90", "sot_per90")
    lineup_fouls = _sum_with_fallback(lineup_form, "fouls_committed_per90", "fouls_committed_per90")

    n_with_form = _count_reliable(lineup_form)

    expected_goals = alpha * team_goals + (1 - alpha) * lineup_goals
    expected_shots = alpha * team_shots + (1 - alpha) * lineup_shots
    expected_sot = alpha * team_sot + (1 - alpha) * lineup_sot
    expected_fouls = alpha * team_fouls + (1 - alpha) * lineup_fouls

    return BlendedRates(
        team_id=team_id,
        expected_goals_per90=expected_goals,
        expected_shots_per90=expected_shots,
        expected_sot_per90=expected_sot,
        expected_fouls_committed_per90=expected_fouls,
        expected_corners_for_per90=team_corners,  # team-only
        alpha=alpha,
        team_component_goals=team_goals,
        lineup_component_goals=lineup_goals,
        n_starters_with_form=n_with_form,
    )


def _safe_get(df: pl.DataFrame, preferred: str, fallback: str) -> float:
    """Get the first row's value from `preferred` column, falling back to `fallback`."""
    col = preferred if preferred in df.columns else fallback
    if col not in df.columns:
        return 0.0
    val = df[col][0]
    return float(val) if val is not None else 0.0


def _sum_with_fallback(df: pl.DataFrame, preferred: str, fallback: str) -> float:
    """Sum a column across rows, preferring `preferred` then `fallback`."""
    if df.is_empty():
        return 0.0
    col = preferred if preferred in df.columns else fallback
    if col not in df.columns:
        return 0.0
    val = df[col].sum()
    return float(val) if val is not None else 0.0


def _count_reliable(df: pl.DataFrame) -> int:
    """Count rows with reliable=True. Returns 0 if column missing."""
    if df.is_empty() or "reliable" not in df.columns:
        return 0
    return int(df.filter(pl.col("reliable")).height)
