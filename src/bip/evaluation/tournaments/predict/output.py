"""Match prediction container + JSON/Markdown emitters.

The MatchPrediction dataclass is the canonical handoff between the
orchestrator (match_runner) and the consumer (locked_predictions writer
or human review report).

JSON shape is committed to git as a contract — backward-compatible
changes preferred. Add fields, never rename or remove.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bip.evaluation.tournaments.predict.blender import BlendedRates
from bip.evaluation.tournaments.predict.player_projections import PlayerProjection
from bip.evaluation.tournaments.predictors.base import GoalsDistribution
from bip.evaluation.tournaments.predictors.corners_poisson import CornersDistribution

JSON_SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "spike-v1"


@dataclass(frozen=True)
class TeamSidePrediction:
    """All predicted quantities for one side of a match."""

    team_id: int
    team_name: str
    fifa_code: str
    lineup_confidence: str
    starters: tuple[PlayerProjection, ...]
    blended: BlendedRates


@dataclass(frozen=True)
class MatchPrediction:
    """Full per-match output. Locked predictions serialize this verbatim."""

    match_id: str
    tournament_slug: str
    kickoff: datetime
    predicted_at: datetime

    home: TeamSidePrediction
    away: TeamSidePrediction

    # Goal-distribution markets (1X2, BTTS, O/U) come from the chosen GoalsModel.
    goals: GoalsDistribution

    # Diagnostic fields
    alpha: float
    rho: float | None  # None for IndependentPoisson
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # Corners distribution — None when no corners model was run.
    corners: CornersDistribution | None = None


# ─────────────────────────────────────────────────────────────────
# JSON emitter
# ─────────────────────────────────────────────────────────────────


def to_json_dict(pred: MatchPrediction) -> dict[str, Any]:
    """Serialize MatchPrediction to a JSON-compatible dict."""
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "match_id": pred.match_id,
        "tournament_slug": pred.tournament_slug,
        "kickoff": pred.kickoff.isoformat(),
        "predicted_at": pred.predicted_at.isoformat(),
        "home": _team_side_to_json(pred.home),
        "away": _team_side_to_json(pred.away),
        "match_markets": {
            "p_home_win": pred.goals.p_home_win,
            "p_draw": pred.goals.p_draw,
            "p_away_win": pred.goals.p_away_win,
            "p_btts": pred.goals.p_btts,
            "p_over_2_5": pred.goals.p_over_2_5,
            "p_under_2_5": pred.goals.p_under_2_5,
            "expected_total_goals": pred.goals.expected_total_goals,
            "lambda_home": pred.goals.lambda_home,
            "lambda_away": pred.goals.lambda_away,
            "model": pred.goals.model_name,
        },
        "corners_markets": _corners_to_json(pred.corners),
        "diagnostics": {
            "alpha": pred.alpha,
            "rho": pred.rho,
            "n_starters_with_form_home": pred.home.blended.n_starters_with_form,
            "n_starters_with_form_away": pred.away.blended.n_starters_with_form,
            "warnings": list(pred.warnings),
        },
    }


def _corners_to_json(c: CornersDistribution | None) -> dict | None:
    if c is None:
        return None
    return {
        "lambda_home": c.lambda_home,
        "lambda_away": c.lambda_away,
        "lambda_total": c.lambda_total,
        "p_over_8_5": c.p_over_8_5,
        "p_over_9_5": c.p_over_9_5,
        "p_over_10_5": c.p_over_10_5,
        "p_over_11_5": c.p_over_11_5,
        "p_home_more": c.p_home_more,
        "p_away_more": c.p_away_more,
        "p_home_ahc_minus_1_5": c.p_home_ahc_minus_1_5,
        "p_away_ahc_minus_1_5": c.p_away_ahc_minus_1_5,
        "p_home_ahc_plus_1_5": c.p_home_ahc_plus_1_5,
        "p_total_fh_over_4_5": c.p_total_fh_over_4_5,
        "p_total_fh_over_5_5": c.p_total_fh_over_5_5,
    }


def write_json(pred: MatchPrediction, path: Path) -> None:
    """Persist a prediction to disk as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(to_json_dict(pred), f, indent=2, sort_keys=False)


def _team_side_to_json(side: TeamSidePrediction) -> dict[str, Any]:
    return {
        "team_id": side.team_id,
        "team_name": side.team_name,
        "fifa_code": side.fifa_code,
        "lineup": {
            "confidence": side.lineup_confidence,
            "starters": [_player_proj_to_json(p) for p in side.starters],
        },
        "expected": {
            "goals": side.blended.expected_goals_per90,
            "shots": side.blended.expected_shots_per90,
            "shots_on_target": side.blended.expected_sot_per90,
            "fouls_committed": side.blended.expected_fouls_committed_per90,
            "corners_for": side.blended.expected_corners_for_per90,
        },
    }


def _player_proj_to_json(p: PlayerProjection) -> dict[str, Any]:
    d = asdict(p)
    # Round projection values for readability without losing structure.
    for k in (
        "lambda_shots",
        "lambda_shots_on_target",
        "lambda_goals",
        "p_anytime_scorer",
        "lambda_fouls_committed",
    ):
        d[k] = round(d[k], 4)
    return d


# ─────────────────────────────────────────────────────────────────
# Markdown emitter
# ─────────────────────────────────────────────────────────────────


def to_markdown(pred: MatchPrediction) -> str:
    """Human-readable report for review before locking predictions."""
    lines: list[str] = []

    lines.append(f"# {pred.home.team_name} vs {pred.away.team_name}")
    lines.append("")
    lines.append(f"**Match:** `{pred.match_id}`  ")
    lines.append(f"**Kickoff:** {pred.kickoff.isoformat()}  ")
    lines.append(f"**Predicted at:** {pred.predicted_at.isoformat()}  ")
    lines.append(
        f"**Model:** {pred.goals.model_name}"
        + (f" (ρ={pred.rho})" if pred.rho is not None else "")
        + f"  α={pred.alpha}"
    )
    lines.append("")

    # Match markets
    lines.append("## Match markets")
    lines.append("")
    lines.append("| Market | Probability |")
    lines.append("|---|---|")
    lines.append(
        f"| {pred.home.fifa_code} win | {pred.goals.p_home_win:.3f} |"
    )
    lines.append(f"| Draw | {pred.goals.p_draw:.3f} |")
    lines.append(
        f"| {pred.away.fifa_code} win | {pred.goals.p_away_win:.3f} |"
    )
    lines.append(f"| BTTS | {pred.goals.p_btts:.3f} |")
    lines.append(f"| Over 2.5 | {pred.goals.p_over_2_5:.3f} |")
    lines.append(f"| Under 2.5 | {pred.goals.p_under_2_5:.3f} |")
    lines.append(f"| Expected total goals | {pred.goals.expected_total_goals:.2f} |")
    lines.append("")

    # Per-team expected stats
    lines.append("## Per-team expected (per 90)")
    lines.append("")
    lines.append("| Stat | Home | Away |")
    lines.append("|---|---:|---:|")
    rows = [
        ("Goals", pred.home.blended.expected_goals_per90, pred.away.blended.expected_goals_per90),
        ("Shots", pred.home.blended.expected_shots_per90, pred.away.blended.expected_shots_per90),
        ("Shots on target", pred.home.blended.expected_sot_per90, pred.away.blended.expected_sot_per90),
        ("Fouls", pred.home.blended.expected_fouls_committed_per90, pred.away.blended.expected_fouls_committed_per90),
        ("Corners for", pred.home.blended.expected_corners_for_per90, pred.away.blended.expected_corners_for_per90),
    ]
    for label, h, a in rows:
        lines.append(f"| {label} | {h:.2f} | {a:.2f} |")
    lines.append("")

    # Player projections per side
    for label, side in (("Home", pred.home), ("Away", pred.away)):
        lines.append(f"## {label} starters — {side.team_name} ({side.lineup_confidence})")
        lines.append("")
        lines.append("| Player | Pos | λ Shots | λ SoT | P(scorer) | λ Fouls | Reliable |")
        lines.append("|---|---|---:|---:|---:|---:|:---:|")
        for p in side.starters:
            tag = "✓" if p.reliable else "⚠"
            lines.append(
                f"| {p.name} | {p.position} | {p.lambda_shots:.2f} | "
                f"{p.lambda_shots_on_target:.2f} | {p.p_anytime_scorer:.3f} | "
                f"{p.lambda_fouls_committed:.2f} | {tag} |"
            )
        lines.append("")

    # Warnings
    if pred.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in pred.warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def write_markdown(pred: MatchPrediction, path: Path) -> None:
    """Persist a Markdown report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(to_markdown(pred))
