"""Typer CLI — `python -m bip.train fit|backtest|promote` (D-05)."""

from __future__ import annotations

from pathlib import Path

import typer

from bip.core.settings import Settings
from bip.train.pipeline import TrainingPipeline
from bip.train.registry import ModelRegistry

app = typer.Typer(help="bip.train — offline ML training pipeline for football.")


def _registry_path(settings: Settings) -> Path:
    return Path(settings.model_dir) / "football" / "registry.json"


@app.command()
def fit(
    league: str = typer.Argument(..., help="League slug, e.g. premier_league"),
    version: str = typer.Option(
        "auto", help="Version tag (e.g. v2). 'auto' = v{len(history)+1}."
    ),
) -> None:
    """Train ensemble for a league and save artifacts — ML-01."""
    settings = Settings()
    registry = ModelRegistry.load(_registry_path(settings))
    if version == "auto":
        history = (
            registry.data.get("leagues", {})
            .get(league, {})
            .get("history", [])
        )
        version = f"v{len(history) + 1}"
    pipeline = TrainingPipeline(settings=settings)
    meta = pipeline.run(league=league, version=version)
    clv_str = (
        f"{meta.walk_forward_mean_clv_pct:.2f}%"
        if meta.walk_forward_mean_clv_pct is not None
        else "N/A (no fold reached 20 picks)"
    )
    typer.echo(
        f"Trained {league}/{meta.version} — "
        f"CLV {clv_str} "
        f"over {meta.walk_forward_folds} folds"
    )


@app.command()
def backtest(
    league: str = typer.Argument(..., help="League slug"),
    version: str = typer.Argument(..., help="Version to re-backtest"),
) -> None:
    """Re-run walk-forward backtest for an existing version (updates metadata.json)."""
    settings = Settings()
    pipeline = TrainingPipeline(settings=settings)
    meta = pipeline.run(league=league, version=version)
    clv_str = (
        f"{meta.walk_forward_mean_clv_pct:.2f}%"
        if meta.walk_forward_mean_clv_pct is not None
        else "N/A (no fold reached 20 picks)"
    )
    typer.echo(
        f"Backtested {league}/{meta.version} — "
        f"CLV {clv_str}"
    )


@app.command()
def promote(
    league: str = typer.Argument(..., help="League slug"),
    version: str = typer.Argument(..., help="Version to promote to production"),
) -> None:
    """Mark version as production for this league — D-05 manual review."""
    settings = Settings()
    registry = ModelRegistry.load(_registry_path(settings))
    registry.promote(league=league, version=version)
    registry.save()
    typer.echo(f"Promoted {league}/{version} to production")
