"""Unit tests for _market_data loader + offset computer (Ola 1.A.1)."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from bip.evaluation.tournaments.wc2026_v2._market_data import (
    DEFAULT_BETA_MV,
    MIN_VALUE_EUR,
    compute_offsets,
    load_squad_values,
    offsets_summary,
)


# ─────────────────────────────────────────────────────────────────
# load_squad_values
# ─────────────────────────────────────────────────────────────────


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "squad_values.parquet"
    pl.DataFrame(rows).write_parquet(p)
    return p


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "squad_values.csv"
    pl.DataFrame(rows).write_csv(p)
    return p


def test_load_parquet_happy_path(tmp_path: Path) -> None:
    path = _write_parquet(
        tmp_path,
        [
            {"team_name": "France", "market_value_eur": 1_200_000_000.0},
            {"team_name": "Saudi Arabia", "market_value_eur": 40_000_000.0},
        ],
    )
    values = load_squad_values(path)
    assert values == {
        "France": 1_200_000_000.0,
        "Saudi Arabia": 40_000_000.0,
    }


def test_load_csv_happy_path(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [
            {"team_name": "Brazil", "market_value_eur": 900_000_000.0},
            {"team_name": "Ecuador", "market_value_eur": 50_000_000.0},
        ],
    )
    values = load_squad_values(path)
    assert values["Brazil"] == 900_000_000.0
    assert values["Ecuador"] == 50_000_000.0


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_squad_values(tmp_path / "does_not_exist.parquet")


def test_load_unsupported_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "values.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        load_squad_values(path)


def test_load_missing_required_columns_raises(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path, [{"team": "France", "value": 1.0}])
    with pytest.raises(ValueError, match="missing required columns"):
        load_squad_values(path)


def test_load_duplicate_team_raises(tmp_path: Path) -> None:
    path = _write_parquet(
        tmp_path,
        [
            {"team_name": "France", "market_value_eur": 1_000_000_000.0},
            {"team_name": "France", "market_value_eur": 1_300_000_000.0},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_squad_values(path)


# ─────────────────────────────────────────────────────────────────
# compute_offsets
# ─────────────────────────────────────────────────────────────────


def test_compute_offsets_empty_input() -> None:
    assert compute_offsets({}) == {}


def test_median_team_gets_zero_offset() -> None:
    values = {
        "Low": 100_000_000.0,
        "Mid": 500_000_000.0,
        "High": 1_000_000_000.0,
    }
    offsets = compute_offsets(values)
    assert offsets["Mid"] == pytest.approx(0.0)
    assert offsets["High"] > 0
    assert offsets["Low"] < 0


def test_offset_scales_with_beta_mv() -> None:
    values = {"A": 200_000_000.0, "B": 800_000_000.0}
    offsets_low = compute_offsets(values, beta_mv=0.05)
    offsets_high = compute_offsets(values, beta_mv=0.20)
    # Same direction, different magnitude.
    assert offsets_low["A"] < 0 and offsets_high["A"] < 0
    assert offsets_low["B"] > 0 and offsets_high["B"] > 0
    assert abs(offsets_high["A"]) == pytest.approx(4.0 * abs(offsets_low["A"]))


def test_explicit_reference_overrides_median() -> None:
    values = {"A": 100.0, "B": 200.0, "C": 300.0}
    # When reference = 200 (= B), B should get offset 0.
    offsets = compute_offsets(
        values, reference=200.0, min_value_eur=1.0, beta_mv=1.0
    )
    assert offsets["B"] == pytest.approx(0.0)
    assert offsets["A"] == pytest.approx(math.log(100.0 / 200.0))
    assert offsets["C"] == pytest.approx(math.log(300.0 / 200.0))


@pytest.mark.parametrize(
    "raw_value", [0.0, -1.0, 1.0, MIN_VALUE_EUR / 2, MIN_VALUE_EUR - 1]
)
def test_min_value_floor_prevents_neg_inf(raw_value: float) -> None:
    """Values at or below the floor must produce finite, bounded offsets."""
    values = {"weak": raw_value, "median": MIN_VALUE_EUR * 100}
    offsets = compute_offsets(values, min_value_eur=MIN_VALUE_EUR)
    assert math.isfinite(offsets["weak"])
    # The weak team's offset must be strictly negative (it's below the
    # reference, which is the geometric center of the clamped values)
    # unless the reference itself collapses to the floor.
    assert offsets["weak"] <= 0.0


def test_default_beta_mv_constant() -> None:
    assert DEFAULT_BETA_MV == 0.10


@pytest.mark.parametrize(
    "ratio,expected_offset",
    [
        (1.0, 0.0),
        (math.e, 1.0 * DEFAULT_BETA_MV),
        (1.0 / math.e, -1.0 * DEFAULT_BETA_MV),
        (math.e**2, 2.0 * DEFAULT_BETA_MV),
    ],
)
def test_offset_log_ratio_boundary_values(
    ratio: float, expected_offset: float
) -> None:
    """Sanity-check the log(value/reference) math at hand-pickable ratios."""
    reference = 200_000_000.0
    values = {"team": reference * ratio}
    offsets = compute_offsets(
        values, reference=reference, min_value_eur=1.0, beta_mv=DEFAULT_BETA_MV
    )
    assert offsets["team"] == pytest.approx(expected_offset, abs=1e-9)


# ─────────────────────────────────────────────────────────────────
# offsets_summary
# ─────────────────────────────────────────────────────────────────


def test_summary_empty_dict() -> None:
    s = offsets_summary({})
    assert s == {
        "count": 0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "median": 0.0,
        "abs_max": 0.0,
    }


def test_summary_basic_stats() -> None:
    offsets = {"a": -0.1, "b": 0.0, "c": 0.05, "d": 0.2}
    s = offsets_summary(offsets)
    assert s["count"] == 4
    assert s["min"] == pytest.approx(-0.1)
    assert s["max"] == pytest.approx(0.2)
    assert s["abs_max"] == pytest.approx(0.2)
    assert s["mean"] == pytest.approx((-0.1 + 0.0 + 0.05 + 0.2) / 4)
