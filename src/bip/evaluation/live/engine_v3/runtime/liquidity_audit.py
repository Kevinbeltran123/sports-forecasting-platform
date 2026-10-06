"""Stake-cap (liquidity) measurement audit for v3 markets.

Risk #3 from the v3 engine design notes sec 8: "Market selector elige
mercados ilíquidos donde edge es teórico". Mitigation: liquidity_score
in MES is a HARD gate, but the scoring constants were set without
ground-truth stake-cap (market liquidity) data.

This module is the **measurement plumbing** for closing that gap.
It does NOT call any bookmaker API. Liquidity observations are
collected manually and written to a CSV.
This module:

1. Parses the operator-supplied CSV with stake-cap observations.
2. Joins observations with v3 shadow picks (which markets did v3
   actually want to fire on?).
3. Computes the distribution of v3 candidates by stake-cap bucket,
   broken down by market family.
4. Recommends a min_stake_cap threshold per family that retains 80%+
   of v3 candidates while excluding the "edge is theoretical" tail.

Output drives an explicit MES config update + a per-market blocklist
in market_selector.py.

# CSV schema (operator-supplied)

Column names exactly as below (case-sensitive, no extra columns):

  - market_family     (str): goals | btts | corners | cards | next_goal |
                              next_corner | props | result_1x2 | ...
  - market_id         (str): full market identifier as logged in shadow picks
                              (e.g. "match_goals_over_2.5")
  - bookmaker         (str): "betano" (others ignored for v3)
  - stake_cap_eur     (float): max single-bet cap observed, in EUR
  - sampled_at_utc    (str): ISO-8601 timestamp of the sample
  - notes             (str, optional): operator free text

One row per observation. Multiple observations per market_id allowed —
analysis takes the median.

# Why median not min

Stake caps fluctuate over time as the market absorbs volume. MIN of observations would over-discount
markets that had brief liquidity dips; MEDIAN reflects sustained
operating capacity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("v3.liquidity_audit")


# Bucket boundaries — natural break points for 1/4-Kelly sizing
# with EUR 50-200 notional unit stakes.
DEFAULT_STAKE_CAP_BUCKETS_EUR: tuple[float, ...] = (
    0.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0,
)

DEFAULT_TARGET_COVERAGE = 0.80  # keep 80% of v3 candidates


@dataclass(frozen=True)
class MarketLiquidityStats:
    """Aggregated liquidity for one market_id."""

    market_id: str
    family: str
    n_observations: int
    median_cap_eur: float
    min_cap_eur: float
    max_cap_eur: float


@dataclass(frozen=True)
class FamilyDistribution:
    """v3 candidate distribution by stake-cap bucket, for one family."""

    family: str
    n_candidates: int
    bucket_counts: dict[str, int]  # bucket label → count
    bucket_pct: dict[str, float]   # bucket label → % of candidates
    recommended_min_cap_eur: float  # keeps target_coverage of candidates
    n_markets_observed: int
    n_markets_unobserved: int


@dataclass(frozen=True)
class LiquidityAuditReport:
    """Top-level audit output."""

    by_family: list[FamilyDistribution]
    overall_observed_pct: float  # share of v3 candidates with a measured cap
    overall_n_candidates: int
    overall_n_with_cap: int
    by_market: list[MarketLiquidityStats] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────────────────────────────


REQUIRED_CSV_COLUMNS = (
    "market_family",
    "market_id",
    "bookmaker",
    "stake_cap_eur",
    "sampled_at_utc",
)


def load_stake_caps_csv(path: Path) -> Any:
    """Load the operator-supplied stake-cap observations.

    Validates the required columns are present. Filters to bookmaker=betano
    rows only (other books are ignored — v3 routes to Betano).
    """
    import polars as pl

    if not path.exists():
        return pl.DataFrame()

    df = pl.read_csv(path)
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"stake_caps CSV missing required columns: {missing}. "
            f"Found: {df.columns}"
        )

    df = df.filter(pl.col("bookmaker").str.to_lowercase() == "betano")
    return df


def aggregate_by_market(stake_caps: Any) -> list[MarketLiquidityStats]:
    """Median + min + max per market_id."""
    import polars as pl

    if stake_caps.is_empty():
        return []

    grouped = stake_caps.group_by("market_id").agg([
        pl.col("market_family").first().alias("family"),
        pl.col("stake_cap_eur").median().alias("median_cap"),
        pl.col("stake_cap_eur").min().alias("min_cap"),
        pl.col("stake_cap_eur").max().alias("max_cap"),
        pl.col("stake_cap_eur").count().alias("n"),
    ])

    out = []
    for row in grouped.iter_rows(named=True):
        out.append(MarketLiquidityStats(
            market_id=str(row["market_id"]),
            family=str(row["family"]),
            n_observations=int(row["n"]),
            median_cap_eur=float(row["median_cap"]),
            min_cap_eur=float(row["min_cap"]),
            max_cap_eur=float(row["max_cap"]),
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Distribution analysis
# ──────────────────────────────────────────────────────────────────────


def _bucket_label(cap_eur: float, buckets: tuple[float, ...]) -> str:
    """Map cap_eur to a bucket label like '50-100' or '5000+'."""
    for i in range(len(buckets) - 1):
        if buckets[i] <= cap_eur < buckets[i + 1]:
            return f"{int(buckets[i])}-{int(buckets[i+1])}"
    return f"{int(buckets[-1])}+"


def _all_bucket_labels(buckets: tuple[float, ...]) -> list[str]:
    labels = [
        f"{int(buckets[i])}-{int(buckets[i+1])}"
        for i in range(len(buckets) - 1)
    ]
    labels.append(f"{int(buckets[-1])}+")
    return labels


def compute_family_distribution(
    family: str,
    family_picks: Any,  # Polars DataFrame of v3 picks for this family
    market_caps: dict[str, MarketLiquidityStats],
    *,
    buckets: tuple[float, ...] = DEFAULT_STAKE_CAP_BUCKETS_EUR,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> FamilyDistribution:
    """For one market family, distribute v3 picks by stake-cap bucket
    and recommend a min cap that keeps target_coverage of candidates.
    """
    import polars as pl

    n_candidates = family_picks.height if not family_picks.is_empty() else 0

    bucket_counts: dict[str, int] = {b: 0 for b in _all_bucket_labels(buckets)}
    n_with_cap = 0

    for row in (family_picks.iter_rows(named=True)
                if not family_picks.is_empty() else []):
        market_id = str(row["market_id"])
        cap_stats = market_caps.get(market_id)
        if cap_stats is None:
            continue
        n_with_cap += 1
        label = _bucket_label(cap_stats.median_cap_eur, buckets)
        bucket_counts[label] = bucket_counts.get(label, 0) + 1

    bucket_pct = {
        b: (c / n_candidates if n_candidates > 0 else 0.0)
        for b, c in bucket_counts.items()
    }

    # Recommended min cap: walk buckets from highest to lowest;
    # find the lowest bucket boundary such that cumulative coverage
    # from that bucket UPWARD >= target_coverage.
    cumulative_high_to_low = 0.0
    recommended = float(buckets[-1])  # default: most restrictive
    labels_high_to_low = list(reversed(_all_bucket_labels(buckets)))
    boundaries_high_to_low = list(reversed(buckets))
    for label, boundary in zip(labels_high_to_low, boundaries_high_to_low):
        cumulative_high_to_low += bucket_pct.get(label, 0.0)
        if cumulative_high_to_low >= target_coverage:
            recommended = float(boundary)
            break

    n_markets_observed = sum(
        1
        for mkt in (family_picks["market_id"].unique().to_list()
                    if not family_picks.is_empty() else [])
        if str(mkt) in market_caps
    )
    n_markets_unobserved = (
        len(family_picks["market_id"].unique().to_list())
        - n_markets_observed
        if not family_picks.is_empty() else 0
    )

    return FamilyDistribution(
        family=family,
        n_candidates=n_candidates,
        bucket_counts=bucket_counts,
        bucket_pct=bucket_pct,
        recommended_min_cap_eur=recommended,
        n_markets_observed=n_markets_observed,
        n_markets_unobserved=n_markets_unobserved,
    )


def run_audit(
    *,
    shadow_picks: Any,  # Polars DataFrame of v3 shadow picks
    stake_caps_csv: Path,
    buckets: tuple[float, ...] = DEFAULT_STAKE_CAP_BUCKETS_EUR,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> LiquidityAuditReport:
    """One full audit pass.

    Loads the stake caps, joins with shadow picks, computes per-family
    distributions + recommendations.
    """
    import polars as pl

    stake_caps = load_stake_caps_csv(stake_caps_csv)
    by_market = aggregate_by_market(stake_caps)
    market_caps_index = {m.market_id: m for m in by_market}

    if shadow_picks.is_empty():
        return LiquidityAuditReport(
            by_family=[],
            overall_observed_pct=0.0,
            overall_n_candidates=0,
            overall_n_with_cap=0,
            by_market=by_market,
        )

    families = sorted(set(shadow_picks["family"].to_list()))
    distributions: list[FamilyDistribution] = []
    overall_n_candidates = 0
    overall_n_with_cap = 0

    for family in families:
        sub = shadow_picks.filter(pl.col("family") == family)
        dist = compute_family_distribution(
            family,
            sub,
            market_caps_index,
            buckets=buckets,
            target_coverage=target_coverage,
        )
        distributions.append(dist)
        overall_n_candidates += dist.n_candidates
        overall_n_with_cap += sum(dist.bucket_counts.values())

    overall_observed_pct = (
        overall_n_with_cap / overall_n_candidates
        if overall_n_candidates > 0 else 0.0
    )

    return LiquidityAuditReport(
        by_family=distributions,
        overall_observed_pct=overall_observed_pct,
        overall_n_candidates=overall_n_candidates,
        overall_n_with_cap=overall_n_with_cap,
        by_market=by_market,
    )


__all__ = [
    "DEFAULT_STAKE_CAP_BUCKETS_EUR",
    "DEFAULT_TARGET_COVERAGE",
    "FamilyDistribution",
    "LiquidityAuditReport",
    "MarketLiquidityStats",
    "REQUIRED_CSV_COLUMNS",
    "aggregate_by_market",
    "compute_family_distribution",
    "load_stake_caps_csv",
    "run_audit",
]
