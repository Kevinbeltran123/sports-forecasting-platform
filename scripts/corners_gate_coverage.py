"""CORNERS-01 Part B (D-17b): Polars coverage query over 02.1 results Parquet.

Per league x season, computes coverage of home_corners / away_corners (and any other
CORNER_TIMING_COLS). Filters on status='FT' first (specifics §199 -- PST/CANC/ABD must
NOT count as missing-coverage; they legitimately don't have corner data).

Gate: every league must have >=3 seasons with >=95% coverage. EITHER fail -> exit 1.

A5: home_corners / away_corners are NOT seeded by Phase 02.1 P09. This script will
LIKELY fail on first run -- that is the CORRECT outcome per D-17b. Do NOT pre-populate
data; the gate exists to surface the deficit before Phase 6 work begins.

PATTERNS.md drift risk #20: this is a one-shot manual trigger -- never a CI step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import structlog

from bip.core.settings import Settings
from bip.core.storage.parquet_store import ParquetStore

logger = structlog.get_logger(__name__)

# Required columns for the coverage check. As of Phase 02.1 results store these are
# NOT present (only home_goals / away_goals / status). A5 documents this expectation.
CORNER_TIMING_COLS: list[str] = ["home_corners", "away_corners"]

LEAGUES: list[str] = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
COVERAGE_THRESHOLD: float = 0.95
MIN_SEASONS: int = 3

OUTPUT_PATH: Path = Path("scripts/corners_gate_coverage.md")


def compute_coverage(store: ParquetStore) -> pl.DataFrame:
    """Per-league per-season coverage of CORNER_TIMING_COLS over FT-only rows.

    Returns a DataFrame with columns: league, season, ft_count, coverage_pct, missing_cols.
    coverage_pct = MIN over all CORNER_TIMING_COLS of (non-null fraction within FT rows).
    """
    rows: list[dict] = []
    for league in LEAGUES:
        try:
            df = store.read_results(sport="football", league=league)
        except Exception as exc:
            logger.warning("read_results_failed", league=league, error=str(exc))
            rows.append({"league": league, "season": "ALL", "ft_count": 0,
                          "coverage_pct": 0.0, "missing_cols": "read_failed"})
            continue

        if df.is_empty():
            rows.append({"league": league, "season": "ALL", "ft_count": 0,
                          "coverage_pct": 0.0, "missing_cols": "no_data"})
            continue

        # FT filter (specifics §199)
        ft = df.filter(pl.col("status") == "FT") if "status" in df.columns else df
        if ft.is_empty():
            rows.append({"league": league, "season": "ALL", "ft_count": 0,
                          "coverage_pct": 0.0, "missing_cols": "no_ft_rows"})
            continue

        # Identify missing columns up front -- A5: usually home_corners/away_corners absent
        missing = [c for c in CORNER_TIMING_COLS if c not in ft.columns]

        if missing == CORNER_TIMING_COLS:
            # Fast path -- none of the required columns exist; coverage is 0 across all seasons
            seasons = (ft.select("season").unique().to_series().to_list()
                       if "season" in ft.columns else ["ALL"])
            for s in seasons:
                cnt = ft.filter(pl.col("season") == s).height if "season" in ft.columns else ft.height
                rows.append({"league": league, "season": str(s), "ft_count": cnt,
                              "coverage_pct": 0.0, "missing_cols": ",".join(missing)})
            continue

        # Group by season, compute non-null fraction per present corner col
        present_cols = [c for c in CORNER_TIMING_COLS if c in ft.columns]
        agg_exprs = [pl.len().alias("ft_count")] + [
            pl.col(c).is_not_null().mean().alias(f"{c}_cov") for c in present_cols
        ]
        if "season" in ft.columns:
            agg = ft.group_by("season").agg(agg_exprs).sort("season")
            for r in agg.iter_rows(named=True):
                cov_cols = [v for k, v in r.items() if k.endswith("_cov")]
                min_cov = float(min(cov_cols)) if cov_cols else 0.0
                rows.append({
                    "league": league,
                    "season": str(r["season"]),
                    "ft_count": int(r["ft_count"]),
                    "coverage_pct": min_cov,
                    "missing_cols": ",".join(missing),
                })
        else:
            # No season column -> single bucket
            agg = ft.select(agg_exprs)
            r = agg.row(0, named=True) if agg.height else {}
            cov_cols = [v for k, v in r.items() if k.endswith("_cov")]
            min_cov = float(min(cov_cols)) if cov_cols else 0.0
            rows.append({"league": league, "season": "ALL",
                          "ft_count": int(r.get("ft_count", 0)),
                          "coverage_pct": min_cov,
                          "missing_cols": ",".join(missing)})

    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"league": pl.String, "season": pl.String, "ft_count": pl.Int64,
                 "coverage_pct": pl.Float64, "missing_cols": pl.String}
    )


def gate_decision(coverage: pl.DataFrame) -> tuple[bool, str]:
    """Returns (passed, markdown_summary). Pass requires every league to have >=MIN_SEASONS
    seasons with >=COVERAGE_THRESHOLD coverage."""
    md_lines = ["# CORNERS-01 Coverage Report (Part B / D-17b)", ""]
    md_lines.append(f"**Threshold:** {COVERAGE_THRESHOLD*100:.0f}% coverage on >={MIN_SEASONS} seasons per league")
    md_lines.append("")
    md_lines.append("| League | Seasons >=95% | Verdict |")
    md_lines.append("|--------|--------------|---------|")
    all_pass = True
    for league in LEAGUES:
        league_rows = coverage.filter(pl.col("league") == league)
        good = league_rows.filter(pl.col("coverage_pct") >= COVERAGE_THRESHOLD).height
        passed = good >= MIN_SEASONS
        all_pass &= passed
        md_lines.append(f"| {league} | {good} | {'PASS' if passed else 'FAIL'} |")
    md_lines.append("")
    md_lines.append("## Detail")
    md_lines.append("")
    if coverage.is_empty():
        md_lines.append("_(no data -- coverage table is empty; gate FAILS by definition)_")
    else:
        md_lines.append("| League | Season | FT count | Coverage % | Missing cols |")
        md_lines.append("|--------|--------|----------|-----------:|--------------|")
        for r in coverage.iter_rows(named=True):
            md_lines.append(
                f"| {r['league']} | {r['season']} | {r['ft_count']} | "
                f"{r['coverage_pct']*100:.1f}% | {r['missing_cols']} |"
            )
    md_lines.append("")
    md_lines.append(f"**Overall verdict:** {'PASS' if all_pass else 'FAIL'}")
    return all_pass, "\n".join(md_lines)


def _atomic_write_md(path: Path, content: str) -> None:
    """PATTERNS.md §7: write to .md.tmp then Path.replace for atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="CORNERS-01 D-17b: coverage gate over Phase 02.1 results Parquet"
    )
    parser.add_argument(
        "--parquet-base-path",
        default=None,
        help="Override parquet base path (default: read from Settings / PARQUET_BASE_PATH env var)",
    )
    args = parser.parse_args()

    # Resolve parquet_base_path: CLI arg > env var > Settings best-effort > hardcoded default.
    # Settings may fail with ValidationError when API keys are absent (this script only reads
    # Parquet, so DB / API credentials are not needed). Gracefully fall back so the script
    # remains runnable in CI and local environments without a full .env file.
    parquet_base_path_str: str
    if args.parquet_base_path:
        parquet_base_path_str = args.parquet_base_path
    elif "PARQUET_BASE_PATH" in os.environ:
        parquet_base_path_str = os.environ["PARQUET_BASE_PATH"]
    else:
        try:
            settings = Settings()
            parquet_base_path_str = settings.parquet_base_path
        except Exception:
            parquet_base_path_str = "data/cache"
            logger.warning(
                "settings_load_failed",
                note="Falling back to default parquet_base_path=data/cache; "
                     "set PARQUET_BASE_PATH env var or pass --parquet-base-path to override",
            )

    store = ParquetStore(base_path=Path(parquet_base_path_str))
    coverage = compute_coverage(store)
    passed, md = gate_decision(coverage)
    _atomic_write_md(OUTPUT_PATH, md)
    logger.info("corners_gate_coverage_written", path=str(OUTPUT_PATH), passed=passed)
    if passed:
        # Gate PASS -- write the corners_gate_pass.md marker (D-18 PASS path)
        _atomic_write_md(Path("scripts/corners_gate_pass.md"),
                          "# CORNERS-01 Gate -- PASS\n\n"
                          f"Generated by `scripts/corners_gate_coverage.py`. "
                          f"Coverage report: `scripts/corners_gate_coverage.md`.\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
