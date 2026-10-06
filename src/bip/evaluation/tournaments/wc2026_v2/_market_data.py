"""Transfermarkt squad-value covariate — loader + offset computer (Wave 1.A).

Implements the Peeters (2018) "wisdom of crowds" mechanism: log of the
squad's total Transfermarkt market value is folded into the team-strength
log-rate as an additive offset. The offset is computed once per tournament
(squad values are slowly-varying within a tournament window) and looked up
at predict time.

Rationale (paraphrased from internal research notes §2.A):

- Peeters 2018 reports market value beats FIFA ranking + Elo on intl
  qualifying. No published Brier delta for tournaments.
- Brown et al. 2024 (n=47k PL matches) finds market value subsumed by
  bookmaker odds for the Big Six; effect concentrated on high-value-spread
  regimes. International tournaments have ~30× squad-value spread (France
  €1.2B vs Saudi €40M), much higher than PL's ~5× → mechanism may transfer.
- Effect applied to attack only (simpler than symmetric attack+defense;
  ablation will reveal whether defense-side offset adds signal).
- Coefficient ``beta_mv`` is a hyperparameter, NOT fit jointly with the
  MLE — keeps the v2 MLE convergence guarantees intact. Default 0.10 is
  a heuristic compromise; the ablation in Ola 2.A sweeps {0.05, 0.10,
  0.15, 0.20} on the n=115 calibration corpus to pick the operating point
  (sweep on calibration, NOT on the n=199 hold-out → no leakage).

Public API:
- ``load_squad_values(path)`` — reads parquet/CSV → dict[team, value_eur]
- ``compute_offsets(values, *, beta_mv, reference)`` → dict[team, log_offset]
- ``DEFAULT_BETA_MV`` — module-level constant, 0.10
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import median

import polars as pl


DEFAULT_BETA_MV: float = 0.10
"""Default coefficient mapping log(value/reference) to log-rate offset.

The applied offset is ``beta_mv * log(value / reference)``. A team at
the reference value gets offset 0.0; a team at 2× reference gets
``beta_mv * log(2) ≈ 0.069``, which translates to a scoring-rate ratio
of ``exp(0.069) ≈ 1.072`` (7.2% more goals than against the same
opponent at reference).
"""

MIN_VALUE_EUR: float = 1_000_000.0
"""Lower clip on squad value (EUR) before log transform.

Squads with value below this floor are clamped to it. Prevents -inf
offsets from data errors (zero / missing / mis-parsed values) without
silently dropping the team. The floor is intentionally permissive
(€1M is well below any real intl squad in 2024-2026).
"""


def load_squad_values(path: str | Path) -> dict[str, float]:
    """Load Transfermarkt squad values into a {team_name: value_eur} dict.

    Accepts parquet or CSV. Required columns: ``team_name``,
    ``market_value_eur``. Optional ``tournament``, ``snapshot_date``
    columns are ignored here (callers slice by tournament before passing
    to this loader). Duplicate team_name rows raise ValueError — caller
    must pre-filter to one snapshot per team per tournament.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Squad values file not found: {p}")

    if p.suffix == ".parquet":
        df = pl.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pl.read_csv(p, separator="\t" if p.suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported squad values file format: {p.suffix}")

    required = {"team_name", "market_value_eur"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Squad values file missing required columns: {missing}")

    rows = df.select(["team_name", "market_value_eur"]).iter_rows(named=True)
    out: dict[str, float] = {}
    for row in rows:
        team = str(row["team_name"])
        value = float(row["market_value_eur"])
        if team in out:
            raise ValueError(
                f"Duplicate team_name in squad values file: {team!r}. "
                "Pre-filter to one snapshot per team before calling load_squad_values."
            )
        out[team] = value
    return out


def compute_offsets(
    values: dict[str, float],
    *,
    beta_mv: float = DEFAULT_BETA_MV,
    reference: float | None = None,
    min_value_eur: float = MIN_VALUE_EUR,
) -> dict[str, float]:
    """Map team squad values → log-rate offsets ``beta_mv * log(v / ref)``.

    Args:
      values: ``{team_name: market_value_eur}``. Empty dict returns
        empty offsets (no-op equivalent for the ablation toggle).
      beta_mv: coefficient on the log-ratio. Default 0.10.
      reference: anchor value. If ``None``, uses the median of ``values``
        (so the median-value team gets offset 0.0 by construction). The
        median is robust to extreme high-value outliers like France in a
        WC squad-list.
      min_value_eur: floor applied to values before the log to keep the
        transform finite when a team's value is unrealistically small.

    Returns:
      ``{team_name: log_offset}`` with the same keys as ``values``.
      Empty input → empty output.
    """

    if not values:
        return {}

    if reference is None:
        reference = median(values.values())
    reference = max(float(reference), float(min_value_eur))

    out: dict[str, float] = {}
    for team, raw_value in values.items():
        v = max(float(raw_value), float(min_value_eur))
        out[team] = float(beta_mv) * math.log(v / reference)
    return out


def offsets_summary(offsets: dict[str, float]) -> dict[str, float]:
    """Quick diagnostic stats for an offsets dict (for ablation logging).

    Returns dict with: ``count``, ``min``, ``max``, ``mean``, ``median``,
    ``abs_max``. Useful for ablation reports + sanity checks (e.g.,
    extreme offsets > 0.5 suggest beta_mv is too aggressive for the
    value spread).
    """

    if not offsets:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "abs_max": 0.0,
        }
    vals = list(offsets.values())
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "abs_max": max(abs(v) for v in vals),
    }
