"""Ola 6 — ablation study.

Run the walk-forward backtest under five configurations:

- ``full``        : DIBP on, beta calibration on, match-importance on
- ``no_dibp``     : DIBP collapsed to plain BP (π=0, ρ=0)
- ``no_beta``     : raw markets (no calibration layer)
- ``no_importance``: K-factor flattened to friendly-only
- ``baseline``    : all three ablations together (= simplest model)

For each non-full configuration we compute the bootstrap delta-Brier vs
the full config. If a component's delta-Brier crosses zero (CI includes 0)
its effect is not statistically significant at the 5% level and should
be considered for removal per the Ola 6 gate in PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .backtest import BootstrapCI, V2BacktestResult, bootstrap_ci, run_walk_forward_backtest


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_dibp: bool
    use_beta_calibration: bool
    use_match_importance: bool


ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig("full", True, True, True),
    AblationConfig("no_dibp", False, True, True),
    AblationConfig("no_beta", True, False, True),
    AblationConfig("no_importance", True, True, False),
    AblationConfig("baseline", False, False, False),
)


@dataclass
class AblationCell:
    """One row of the ablation table."""

    config: str
    overall_brier_1x2: BootstrapCI
    overall_brier_btts: BootstrapCI
    overall_brier_ou_2_5: BootstrapCI
    overall_ece_1x2: float
    overall_ece_btts: float
    overall_ece_ou_2_5: float
    delta_brier_1x2: BootstrapCI | None  # vs full; None for the full row itself
    per_tournament_brier_1x2: dict[str, BootstrapCI]


def run_ablation_study(
    train: pl.DataFrame,
    calibration: pl.DataFrame,
    heldout: pl.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    configs: tuple[AblationConfig, ...] = ABLATION_CONFIGS,
) -> dict[str, AblationCell]:
    """Run all configurations, compute paired delta-Brier vs full."""

    results: dict[str, V2BacktestResult] = {}
    per_fixture_1x2: dict[str, np.ndarray] = {}
    for cfg in configs:
        r = run_walk_forward_backtest(
            train=train,
            calibration=calibration,
            heldout=heldout,
            n_bootstrap=n_bootstrap,
            seed=seed,
            use_dibp=cfg.use_dibp,
            use_beta_calibration=cfg.use_beta_calibration,
            use_match_importance=cfg.use_match_importance,
        )
        results[cfg.name] = r
        per_fixture_1x2[cfg.name] = np.concatenate(
            [m.per_fixture_brier_1x2 for m in r.per_tournament.values()]
        )

    cells: dict[str, AblationCell] = {}
    full_per_fixture = per_fixture_1x2["full"]
    for cfg in configs:
        r = results[cfg.name]
        if cfg.name == "full":
            delta = None
        else:
            # Paired delta: per-fixture (config - full), positive = config worse.
            assert per_fixture_1x2[cfg.name].shape == full_per_fixture.shape
            paired = per_fixture_1x2[cfg.name] - full_per_fixture
            delta = bootstrap_ci(paired, n_resamples=n_bootstrap, seed=seed + 200)

        cells[cfg.name] = AblationCell(
            config=cfg.name,
            overall_brier_1x2=r.overall_brier_1x2,
            overall_brier_btts=r.overall_brier_btts,
            overall_brier_ou_2_5=r.overall_brier_ou_2_5,
            overall_ece_1x2=r.overall_ece_1x2,
            overall_ece_btts=r.overall_ece_btts,
            overall_ece_ou_2_5=r.overall_ece_ou_2_5,
            delta_brier_1x2=delta,
            per_tournament_brier_1x2={slug: m.brier_1x2 for slug, m in r.per_tournament.items()},
        )
    return cells


def format_ablation_table(cells: dict[str, AblationCell]) -> str:
    """Render a markdown table of the ablation results."""

    lines: list[str] = []
    lines.append("## WC2026 v2 Ablation — Lock Convention 1X2 Brier")
    lines.append("")
    lines.append(
        "| Config | 1X2 Brier [CI] | ECE 1X2 | BTTS Brier | OU2.5 Brier | Δ Brier 1X2 vs full [CI] | Significant? |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, cell in cells.items():
        if cell.delta_brier_1x2 is None:
            delta_str = "—"
            sig = "—"
        else:
            d = cell.delta_brier_1x2
            delta_str = f"{d.point:+.4f} [{d.lower:+.4f}, {d.upper:+.4f}]"
            # Significant if the CI does NOT include zero.
            crosses_zero = d.lower <= 0.0 <= d.upper
            sig = "no" if crosses_zero else "**yes**"
        lines.append(
            f"| {name} "
            f"| {cell.overall_brier_1x2} "
            f"| {cell.overall_ece_1x2:.4f} "
            f"| {cell.overall_brier_btts.point:.4f} "
            f"| {cell.overall_brier_ou_2_5.point:.4f} "
            f"| {delta_str} | {sig} |"
        )

    lines.append("")
    lines.append("Per-tournament 1X2 Brier (gate-scale):")
    lines.append("")
    tournament_order = ("wc_2022", "afcon_2023", "copa_2024", "euro_2024")
    header = "| Config | " + " | ".join(tournament_order) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(tournament_order)))
    for name, cell in cells.items():
        row = [name]
        for slug in tournament_order:
            if slug in cell.per_tournament_brier_1x2:
                row.append(f"{cell.per_tournament_brier_1x2[slug].point:.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
