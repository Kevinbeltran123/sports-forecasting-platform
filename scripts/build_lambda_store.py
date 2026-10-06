"""Offline ML-lambda producer using penaltyblog Dixon-Coles.

Fits a Dixon-Coles goal model on available historical results and emits
a per-fixture lambda store: {fixture_id, lambda_home, lambda_away,
model_version} as a Polars parquet file.

The lambda store is consumed by two runtime components:
- derive_priors_from_fixture (dual_write.py): populates
  PreMatchPriors.lambda_home_prematch / away when a λ-store row exists,
  overriding the Sportmonks type_id-240 derivation.
- _choose_dominant_team_id (gsv_builder.py): uses the λ-gap as a
  principled dominant-team signal (ML-λ precedence tier between ELO and
  market).

# Usage

    uv run python scripts/build_lambda_store.py \\
        --history-parquet data/training/results.parquet \\
        --fixtures-parquet data/fixtures/upcoming.parquet \\
        --output data/cache/lambda_store.parquet

# Input schema

``--history-parquet`` must have columns:
    home_team  str   — home team name (consistent across history + fixtures)
    away_team  str   — away team name
    home_goals int   — final home goals
    away_goals int   — final away goals

``--fixtures-parquet`` must have columns:
    fixture_id int   — Sportmonks fixture ID
    home_team  str   — home team name (matching history vocabulary)
    away_team  str   — away team name (matching history vocabulary)

# Output schema

Parquet at ``--output``:
    fixture_id   int64   — Sportmonks fixture ID
    lambda_home  float64 — fitted E[home goals] from Dixon-Coles
    lambda_away  float64 — fitted E[away goals] from Dixon-Coles
    model_version str    — "dixon_coles_v1"

# Structure-first policy

This script ships with a synthetic corpus generator (``generate_synthetic_corpus``)
that is used in tests without any real data dependency. The producer is
correct for real data even if not yet run — per project convention.

# Graceful fallback

When a fixture's team names are not in the training vocabulary
(cold-start or newly promoted teams), the script logs a warning and
skips that fixture. The runtime fallback in derive_priors_from_fixture
handles absent rows.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import penaltyblog

log = logging.getLogger("v3.lambda_store")

MODEL_VERSION = "dixon_coles_v1"

# Minimum matches per team to include in training (fewer → unreliable estimates)
_MIN_MATCHES_PER_TEAM = 3

# Minimum gap between home and away λ to be considered decisive.
# Below this, the team with the higher λ barely outperforms the other —
# considered a coin-flip λ (falls back to market signal in dominant-team selection).
ML_LAMBDA_MIN_GAP = 0.15


@dataclass(frozen=True)
class LambdaRow:
    """Per-fixture lambda estimate."""
    fixture_id: int
    lambda_home: float
    lambda_away: float
    model_version: str = MODEL_VERSION


def generate_synthetic_corpus(n_teams: int = 6, n_fixtures: int = 30) -> dict:
    """Generate a minimal synthetic training corpus for testing.

    Returns a dict with keys ``results`` (list of dicts) and ``teams``
    (list of team names). Suitable for fit_dixon_coles and tests.
    """
    import random
    teams = [f"Team_{chr(65 + i)}" for i in range(n_teams)]
    results = []
    rng = random.Random(42)
    fixture_pairs = [
        (teams[i], teams[j])
        for i in range(n_teams)
        for j in range(n_teams)
        if i != j
    ]
    chosen = [fixture_pairs[i % len(fixture_pairs)] for i in range(n_fixtures)]
    for home, away in chosen:
        results.append({
            "home_team": home,
            "away_team": away,
            "home_goals": rng.randint(0, 3),
            "away_goals": rng.randint(0, 2),
        })
    return {"results": results, "teams": teams}


def fit_dixon_coles(
    results: Sequence[dict],
    *,
    min_matches: int = _MIN_MATCHES_PER_TEAM,
) -> "penaltyblog.models.DixonColesGoalModel | None":
    """Fit a Dixon-Coles model on ``results`` (list of dicts with
    home_team, away_team, home_goals, away_goals keys).

    Returns None if the corpus is too small to fit.
    """
    try:
        from penaltyblog.models import DixonColesGoalModel
    except ImportError:
        log.error("penaltyblog not installed — cannot fit Dixon-Coles model")
        return None

    if len(results) < 10:
        log.warning("Corpus too small (%d rows) — need at least 10 to fit", len(results))
        return None

    # Count matches per team and filter teams below the minimum
    from collections import Counter
    team_counts: Counter[str] = Counter()
    for r in results:
        team_counts[r["home_team"]] += 1
        team_counts[r["away_team"]] += 1
    valid_teams = {t for t, c in team_counts.items() if c >= min_matches}

    filtered = [
        r for r in results
        if r["home_team"] in valid_teams and r["away_team"] in valid_teams
    ]
    if len(filtered) < 10:
        log.warning("After filtering low-count teams: %d rows remain — too few to fit", len(filtered))
        return None

    goals_h = [int(r["home_goals"]) for r in filtered]
    goals_a = [int(r["away_goals"]) for r in filtered]
    teams_h = [str(r["home_team"]) for r in filtered]
    teams_a = [str(r["away_team"]) for r in filtered]

    try:
        model = DixonColesGoalModel(goals_h, goals_a, teams_h, teams_a)
        model.fit()
        log.info(
            "dixon_coles_fit_done n_matches=%d n_teams=%d aic=%.2f",
            len(filtered),
            model.n_teams,
            model.aic,
        )
        return model
    except Exception as exc:  # noqa: BLE001
        log.error("dixon_coles_fit_failed err=%s", exc)
        return None


def predict_lambdas(
    model: "penaltyblog.models.DixonColesGoalModel",
    fixtures: Sequence[dict],
) -> list[LambdaRow]:
    """Predict E[goals] for each fixture in ``fixtures``.

    ``fixtures`` is a list of dicts with keys ``fixture_id``,
    ``home_team``, ``away_team``.

    Teams not in the model vocabulary are skipped with a warning.
    """
    rows: list[LambdaRow] = []
    model_teams: frozenset[str] = frozenset(model.teams)

    for fx in fixtures:
        fid = int(fx["fixture_id"])
        home = str(fx["home_team"])
        away = str(fx["away_team"])

        if home not in model_teams:
            log.warning("lambda_store_skip fixture=%d home=%r not in vocab", fid, home)
            continue
        if away not in model_teams:
            log.warning("lambda_store_skip fixture=%d away=%r not in vocab", fid, away)
            continue

        try:
            grid = model.predict(home, away)
            rows.append(LambdaRow(
                fixture_id=fid,
                lambda_home=float(grid.home_goal_expectation),
                lambda_away=float(grid.away_goal_expectation),
                model_version=MODEL_VERSION,
            ))
        except Exception as exc:  # noqa: BLE001
            log.warning("lambda_predict_failed fixture=%d err=%s", fid, exc)

    log.info("lambda_store_predict_done n_fixtures=%d n_rows=%d", len(fixtures), len(rows))
    return rows


def write_lambda_store(rows: list[LambdaRow], output_path: Path) -> Path:
    """Write LambdaRow list to parquet.

    Schema: fixture_id (Int64), lambda_home (Float64), lambda_away
    (Float64), model_version (Utf8).
    """
    import polars as pl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "fixture_id": [r.fixture_id for r in rows],
        "lambda_home": [r.lambda_home for r in rows],
        "lambda_away": [r.lambda_away for r in rows],
        "model_version": [r.model_version for r in rows],
    }).with_columns([
        pl.col("fixture_id").cast(pl.Int64),
        pl.col("lambda_home").cast(pl.Float64),
        pl.col("lambda_away").cast(pl.Float64),
        pl.col("model_version").cast(pl.Utf8),
    ])
    df.write_parquet(output_path)
    log.info("lambda_store_written path=%s n=%d", output_path, len(rows))
    return output_path


def read_lambda_store(store_path: Path) -> "dict[int, LambdaRow]":
    """Read the lambda store parquet into a fixture_id → LambdaRow dict.

    Returns an empty dict if the file does not exist.
    """
    import polars as pl

    if not store_path.exists():
        return {}
    df = pl.read_parquet(store_path)
    result: dict[int, LambdaRow] = {}
    for row in df.iter_rows(named=True):
        fid = int(row["fixture_id"])
        result[fid] = LambdaRow(
            fixture_id=fid,
            lambda_home=float(row["lambda_home"]),
            lambda_away=float(row["lambda_away"]),
            model_version=str(row.get("model_version", MODEL_VERSION)),
        )
    return result


def build_lambda_store(
    history_rows: Sequence[dict],
    fixture_rows: Sequence[dict],
    output_path: Path,
    *,
    min_matches: int = _MIN_MATCHES_PER_TEAM,
) -> list[LambdaRow]:
    """End-to-end: fit model + predict + write. Returns list of LambdaRows."""
    model = fit_dixon_coles(history_rows, min_matches=min_matches)
    if model is None:
        log.error("build_lambda_store aborted: model fit failed")
        return []
    rows = predict_lambdas(model, fixture_rows)
    if rows:
        write_lambda_store(rows, output_path)
    return rows


# ──────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Build penaltyblog Dixon-Coles lambda store for v3 dominant-team priors",
    )
    parser.add_argument(
        "--history-parquet",
        required=True,
        help="Parquet with historical results (home_team, away_team, home_goals, away_goals)",
    )
    parser.add_argument(
        "--fixtures-parquet",
        required=True,
        help="Parquet with upcoming fixtures to predict (fixture_id, home_team, away_team)",
    )
    parser.add_argument(
        "--output",
        default="data/cache/lambda_store.parquet",
        help="Output lambda store parquet path (default: data/cache/lambda_store.parquet)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=_MIN_MATCHES_PER_TEAM,
        help=f"Minimum matches per team to include in training (default: {_MIN_MATCHES_PER_TEAM})",
    )
    args = parser.parse_args(argv)

    import polars as pl

    history_path = Path(args.history_parquet)
    if not history_path.exists():
        log.error("history parquet not found: %s", history_path)
        return 1

    fixtures_path = Path(args.fixtures_parquet)
    if not fixtures_path.exists():
        log.error("fixtures parquet not found: %s", fixtures_path)
        return 1

    history_df = pl.read_parquet(history_path)
    history_rows = history_df.select(
        ["home_team", "away_team", "home_goals", "away_goals"]
    ).to_dicts()

    fixtures_df = pl.read_parquet(fixtures_path)
    fixture_rows = fixtures_df.select(["fixture_id", "home_team", "away_team"]).to_dicts()

    output_path = Path(args.output)
    rows = build_lambda_store(
        history_rows, fixture_rows, output_path,
        min_matches=args.min_matches,
    )
    if not rows:
        return 1

    log.info("Done. %d λ rows written to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
