"""Corpus loader + 3-way split for WC2026 lock_v2 spike.

Builds three disjoint match corpora from local data:

- **train** — martj42_international_results.csv minus the four hold-out
  tournament windows. Goals only, no xG. Used by Ola 1 weighted-MLE
  strength prior (Ley, Van de Wiele, Van Eetvelde 2019).
- **calibration** — StatsBomb open-data matches from WC2018 + Euro2020
  (n=115). Has xG. Used by Ola 2 beta calibrator (Kull et al. 2017) and
  Ola 3 DIBP π_diag fit (Karlis & Ntzoufras 2003).
- **heldout** — StatsBomb matches from WC2022 + AFCON2023 + Copa2024 +
  Euro2024 (n=199). Has xG. Walk-forward CV target (Ola 5). No signal
  from these matches touches any fit parameter.

Hold-out tournament windows (verified against both sources, 2026-05-23):
- WC2022:    2022-11-20 → 2022-12-18  (StatsBomb slug=wc_2022)
- AFCON2023: 2024-01-13 → 2024-02-11  (slug=afcon_2023)
- Euro2024:  2024-06-14 → 2024-07-14  (slug=euro_2024)
- Copa2024:  2024-06-21 → 2024-07-15  (slug=copa_2024)

License notes:
- martj42_international_results: CC0 (public domain)
- StatsBomb open-data: CC BY-NC 4.0 (non-commercial use — this is internal
  model validation, not redistribution)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[5]
MARTJ42_CSV = _REPO_ROOT / "data" / "cache" / "martj42_international_results.csv"
STATSBOMB_PARQUET = _REPO_ROOT / "data" / "cache" / "statsbomb" / "match_outcomes.parquet"

HELD_OUT_SLUGS: frozenset[str] = frozenset({"wc_2022", "afcon_2023", "euro_2024", "copa_2024"})
CALIBRATION_SLUGS: frozenset[str] = frozenset({"wc_2018", "euro_2020"})

# Hold-out windows as (start, end, martj42-tournament-name) so we can
# project the StatsBomb hold-outs onto martj42 rows for exclusion.
_HELD_OUT_WINDOWS: tuple[tuple[date, date, str], ...] = (
    (date(2022, 11, 20), date(2022, 12, 18), "FIFA World Cup"),
    (date(2024, 1, 13), date(2024, 2, 11), "African Cup of Nations"),
    (date(2024, 6, 14), date(2024, 7, 14), "UEFA Euro"),
    (date(2024, 6, 21), date(2024, 7, 15), "Copa América"),
)


@dataclass(frozen=True)
class CorpusSplit:
    """Three disjoint match corpora for the lock_v2 spike."""

    train: pl.DataFrame
    calibration: pl.DataFrame
    heldout: pl.DataFrame

    def summary(self) -> dict[str, dict[str, int | str]]:
        out: dict[str, dict[str, int | str]] = {}
        for name, df in (
            ("train", self.train),
            ("calibration", self.calibration),
            ("heldout", self.heldout),
        ):
            if df.height == 0:
                out[name] = {"n_matches": 0, "date_min": "", "date_max": ""}
                continue
            date_col = "match_date" if "match_date" in df.columns else "date"
            out[name] = {
                "n_matches": df.height,
                "date_min": str(df[date_col].min()),
                "date_max": str(df[date_col].max()),
            }
        return out


def load_martj42(path: Path | None = None) -> pl.DataFrame:
    """Load martj42_international_results.csv → typed Polars DataFrame.

    Schema: date | home_team | away_team | home_score | away_score |
    tournament | city | country | neutral. Rows with null scores
    (unplayed fixtures, NA in source) are dropped.
    """

    src = path or MARTJ42_CSV
    df = pl.read_csv(src, try_parse_dates=True, null_values=["NA"])
    return df.drop_nulls(["home_score", "away_score"])


def load_statsbomb(path: Path | None = None) -> pl.DataFrame:
    """Load StatsBomb match_outcomes.parquet → Polars DataFrame.

    Schema: match_id | tournament_slug | match_date | home_team |
    away_team | home_goals | away_goals | home_corners | away_corners |
    home_xg | away_xg | home_shots | away_shots.
    """

    return pl.read_parquet(path or STATSBOMB_PARQUET)


def _martj42_held_out_mask(df: pl.DataFrame) -> pl.Series:
    """Boolean mask: True iff a martj42 row falls inside a hold-out window."""

    expr = pl.lit(False)
    for start, end, tournament_name in _HELD_OUT_WINDOWS:
        window = (
            (pl.col("date") >= pl.lit(start))
            & (pl.col("date") <= pl.lit(end))
            & (pl.col("tournament") == tournament_name)
        )
        expr = expr | window
    return df.select(expr.alias("is_held_out")).to_series()


def build_split(
    martj42: pl.DataFrame | None = None,
    statsbomb: pl.DataFrame | None = None,
) -> CorpusSplit:
    """Build the three-way corpus split.

    - train: martj42 minus the four hold-out tournament windows.
    - calibration: StatsBomb slug ∈ {wc_2018, euro_2020}.
    - heldout: StatsBomb slug ∈ {wc_2022, afcon_2023, euro_2024,
      copa_2024}.
    """

    m = martj42 if martj42 is not None else load_martj42()
    s = statsbomb if statsbomb is not None else load_statsbomb()

    held_out_mask = _martj42_held_out_mask(m)
    train = m.filter(~held_out_mask)
    calibration = s.filter(pl.col("tournament_slug").is_in(list(CALIBRATION_SLUGS)))
    heldout = s.filter(pl.col("tournament_slug").is_in(list(HELD_OUT_SLUGS)))

    return CorpusSplit(train=train, calibration=calibration, heldout=heldout)
