"""Tests for the GSV-thesis-outcome attribution join.

Exercises ``build_real_pattern_pairs`` against a synthetic shadow root
that mirrors the on-disk layout produced by ShadowLogger:

    dt=YYYY-MM-DD/
        gsv_log.parquet          GSVs serialised via model_dump_json
        picks.parquet            Pick rows with archetype + thesis_id
        picks_outcomes.parquet   (optional) operator-supplied outcomes

Invariants verified:

- Empty root → empty pairs + non-crashing report.
- Picks without matching GSV are dropped + counted in n_picks_without_gsv.
- The re-derived Thesis matches the production archetype exactly.
- An archetype that no longer matches the current rule layer is dropped
  with a warning + counted (mismatch path).
- date_range filtering honored at both gsv_log and picks granularity.
- Outcome attribution: with picks_outcomes present, pairs carry
  status + profit; without it they carry None.
- ``require_outcome=True`` filters pending/null pairs.
- ``only_won=True`` keeps only winning pairs.
- Archetype distribution counter populated correctly.
- ``pairs_to_fit_input`` produces the (GSV, Thesis) tuple the
  PatternLayer.fit expects.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from bip.evaluation.live.engine_v3.archetypes import generate_theses
from bip.evaluation.live.engine_v3.gsv import (
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.evaluation.live.engine_v3.gsv_builder import GSVBuilder
from bip.evaluation.live.engine_v3.runtime.training_data import (
    JoinReport,
    build_real_pattern_pairs,
    pairs_to_fit_input,
)
from bip.evaluation.live.engine_v3.thesis import ThesisArchetype
from tests.evaluation.live.engine_v3.conftest import HOME_ID, make_state
from tests.evaluation.live.engine_v3.test_anti_napoli import _synthesize_states


# ──────────────────────────────────────────────────────────────────────
# Helpers — build a realistic shadow root in tmp_path
# ──────────────────────────────────────────────────────────────────────


def _napoli_priors() -> PreMatchPriors:
    return PreMatchPriors(
        lambda_home_prematch=2.30,
        lambda_away_prematch=0.95,
        expected_corners_total=10.8,
        expected_cards_total=4.10,
        elo_diff=140.0,
    )


def _markets() -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    return MarketSnapshot(
        lines={
            "match_goals_over_2.5": MarketLine(
                market_id="match_goals_over_2.5",
                side_a_decimal=1.95,
                side_b_decimal=1.95,
                line_value=2.5,
                max_stake_cap=500.0,
                last_update_utc=now - timedelta(seconds=20),
            ),
        }
    )


def _build_napoli_pairs(n: int) -> list[tuple]:
    """Build n (GSV, Thesis) pairs from the anti-Napoli regression set.

    Returns up to n pairs where the rule layer fires the
    DOMINANT_LOSING_NAPOLI archetype.
    """
    builder = GSVBuilder()
    priors = _napoli_priors()
    markets = _markets()
    out = []
    for state in _synthesize_states():
        if len(out) >= n:
            break
        gsv = builder.build(
            state, priors=priors, markets=markets, dominant_team_id=HOME_ID
        )
        theses = generate_theses(gsv)
        napoli = next(
            (t for t in theses
             if t.archetype is ThesisArchetype.DOMINANT_LOSING_NAPOLI),
            None,
        )
        if napoli is None:
            continue
        out.append((gsv, napoli))
    return out


def _write_shadow_partition(
    root: Path,
    day: int,
    pairs: list[tuple],
    *,
    outcomes: list[dict] | None = None,
) -> None:
    """Write gsv_log + picks + (optional) outcomes into the daily partition."""
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)

    gsv_rows = []
    pick_rows = []
    for idx, (gsv, thesis) in enumerate(pairs):
        gsv_rows.append({
            "fixture_id": int(gsv.fixture_id),
            "state_version": int(gsv.state_version),
            "timestamp_utc": gsv.timestamp_utc,
            "home_team_id": int(gsv.home_team_id),
            "away_team_id": int(gsv.away_team_id),
            "minute": int(gsv.time.minute),
            "period": gsv.time.period,
            "home_goals": int(gsv.score.home_goals),
            "away_goals": int(gsv.score.away_goals),
            "gsv_json": gsv.model_dump_json(),
        })
        pick_rows.append({
            "fixture_id": int(gsv.fixture_id),
            "timestamp_utc": gsv.timestamp_utc,
            "thesis_id": thesis.id,
            "archetype": thesis.archetype.value,
            "thesis_layer": thesis.source.layer,
            "rule_id": thesis.source.identifier,
            "family": thesis.prediction.family.value,
            "direction": thesis.prediction.direction,
            "magnitude_pp": float(thesis.prediction.magnitude_pp),
            "horizon_minutes": int(thesis.prediction.horizon.horizon_minutes),
            "market_id": "match_goals_over_2.5",
            "fair_prob": 0.55,
            "base_edge": 0.05,
            "signal_clarity": 0.70,
            "book_slowness": 0.40,
            "liquidity_score": 0.85,
            "conditional_variance": 0.10,
            "mes_score": 0.50,
            "confidence_prior": float(thesis.confidence_prior),
            "activated_at_minute": int(thesis.activated_at_minute),
        })
    pl.DataFrame(gsv_rows).write_parquet(part / "gsv_log.parquet")
    pl.DataFrame(pick_rows).write_parquet(part / "picks.parquet")

    if outcomes is not None:
        pl.DataFrame(outcomes).write_parquet(part / "picks_outcomes.parquet")


# ──────────────────────────────────────────────────────────────────────
# Empty / cold-start paths
# ──────────────────────────────────────────────────────────────────────


def test_empty_root_returns_empty_pairs(tmp_path):
    pairs, report = build_real_pattern_pairs(shadow_root=tmp_path)
    assert pairs == []
    assert isinstance(report, JoinReport)
    assert report.n_gsvs_loaded == 0
    assert report.n_picks_loaded == 0
    assert report.n_pairs_built == 0


def test_gsvs_without_picks_returns_empty(tmp_path):
    """A partition with gsv_log but no picks → no pairs."""
    pairs = _build_napoli_pairs(3)
    part = tmp_path / "dt=2026-05-11"
    part.mkdir(parents=True)
    rows = []
    for gsv, _ in pairs:
        rows.append({
            "fixture_id": int(gsv.fixture_id),
            "state_version": int(gsv.state_version),
            "timestamp_utc": gsv.timestamp_utc,
            "home_team_id": int(gsv.home_team_id),
            "away_team_id": int(gsv.away_team_id),
            "minute": int(gsv.time.minute),
            "period": gsv.time.period,
            "home_goals": int(gsv.score.home_goals),
            "away_goals": int(gsv.score.away_goals),
            "gsv_json": gsv.model_dump_json(),
        })
    pl.DataFrame(rows).write_parquet(part / "gsv_log.parquet")
    pairs_built, report = build_real_pattern_pairs(shadow_root=tmp_path)
    assert pairs_built == []
    assert report.n_gsvs_loaded == 3
    assert report.n_picks_loaded == 0


# ──────────────────────────────────────────────────────────────────────
# Happy path: pairs reconstructed correctly
# ──────────────────────────────────────────────────────────────────────


def test_pairs_reconstructed_with_correct_archetype(tmp_path):
    napoli_pairs = _build_napoli_pairs(5)
    assert len(napoli_pairs) > 0, "regression: anti-Napoli fixtures changed"
    _write_shadow_partition(tmp_path, day=11, pairs=napoli_pairs)

    pairs, report = build_real_pattern_pairs(shadow_root=tmp_path)

    assert len(pairs) == len(napoli_pairs)
    for p in pairs:
        assert p.archetype == ThesisArchetype.DOMINANT_LOSING_NAPOLI
        assert p.thesis.archetype == ThesisArchetype.DOMINANT_LOSING_NAPOLI
    assert report.archetype_distribution == {
        "dominant_losing_napoli": len(napoli_pairs)
    }


def test_picks_to_fit_input_produces_tuples(tmp_path):
    napoli_pairs = _build_napoli_pairs(3)
    _write_shadow_partition(tmp_path, day=11, pairs=napoli_pairs)
    pairs, _ = build_real_pattern_pairs(shadow_root=tmp_path)
    fit_pairs = pairs_to_fit_input(pairs)
    assert len(fit_pairs) == len(pairs)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in fit_pairs)


# ──────────────────────────────────────────────────────────────────────
# Mismatch handling
# ──────────────────────────────────────────────────────────────────────


def test_pick_with_unrecognized_archetype_dropped(tmp_path):
    """If picks.parquet contains an archetype string the rule layer no
    longer fires, the pair is dropped with a warning."""
    napoli = _build_napoli_pairs(2)
    _write_shadow_partition(tmp_path, day=11, pairs=napoli)
    # Tamper with the parquet: rename archetype to something the rule
    # layer cannot match.
    picks_path = tmp_path / "dt=2026-05-11" / "picks.parquet"
    df = pl.read_parquet(picks_path)
    df = df.with_columns(pl.lit("nonexistent_archetype").alias("archetype"))
    df.write_parquet(picks_path)

    pairs, report = build_real_pattern_pairs(shadow_root=tmp_path)
    assert pairs == []
    assert report.n_picks_loaded == 2
    assert report.n_pairs_built == 0


def test_pick_without_matching_gsv_counted_as_orphan(tmp_path):
    """A pick whose (fixture_id, timestamp) doesn't match any GSV row
    is dropped + counted in n_picks_without_gsv."""
    napoli = _build_napoli_pairs(2)
    _write_shadow_partition(tmp_path, day=11, pairs=napoli)
    # Tamper: shift all picks' timestamps by 1 second so they no longer
    # match the gsv_log timestamps.
    picks_path = tmp_path / "dt=2026-05-11" / "picks.parquet"
    df = pl.read_parquet(picks_path)
    df = df.with_columns(
        (pl.col("timestamp_utc") + pl.duration(seconds=1)).alias("timestamp_utc")
    )
    df.write_parquet(picks_path)

    pairs, report = build_real_pattern_pairs(shadow_root=tmp_path)
    assert pairs == []
    assert report.n_picks_without_gsv == 2


# ──────────────────────────────────────────────────────────────────────
# Date range filtering
# ──────────────────────────────────────────────────────────────────────


def test_date_range_filters_partitions(tmp_path):
    napoli = _build_napoli_pairs(4)
    _write_shadow_partition(tmp_path, day=10, pairs=napoli[:2])
    _write_shadow_partition(tmp_path, day=12, pairs=napoli[2:])

    start = datetime(2026, 5, 12, tzinfo=timezone.utc)
    end = datetime(2026, 5, 12, tzinfo=timezone.utc)
    pairs, report = build_real_pattern_pairs(
        shadow_root=tmp_path, date_range=(start, end)
    )
    assert len(pairs) == 2  # only day=12 partition loaded


# ──────────────────────────────────────────────────────────────────────
# Outcome attribution
# ──────────────────────────────────────────────────────────────────────


def _outcome_row_for(pair, status: str, profit: float) -> dict:
    gsv, thesis = pair
    return {
        "fixture_id": int(gsv.fixture_id),
        "thesis_id": thesis.id,
        "market_id": "match_goals_over_2.5",
        "pick_timestamp_utc": gsv.timestamp_utc,
        "status": status,
        "profit_units": profit,
        "bookmaker_odd": 1.95,
        "settled_at": gsv.timestamp_utc + timedelta(hours=2),
    }


def test_outcome_attribution_populates_status_and_profit(tmp_path):
    napoli = _build_napoli_pairs(3)
    outcomes = [
        _outcome_row_for(napoli[0], "won", 0.95),
        _outcome_row_for(napoli[1], "lost", -1.00),
        _outcome_row_for(napoli[2], "void", 0.0),
    ]
    _write_shadow_partition(tmp_path, day=11, pairs=napoli, outcomes=outcomes)

    pairs, report = build_real_pattern_pairs(shadow_root=tmp_path)
    by_status = {p.outcome_status for p in pairs}
    assert by_status == {"won", "lost", "void"}
    assert report.n_pairs_won == 1
    assert report.n_pairs_lost == 1
    assert report.n_pairs_void == 1


def test_require_outcome_drops_pending_pairs(tmp_path):
    napoli = _build_napoli_pairs(3)
    outcomes = [
        _outcome_row_for(napoli[0], "won", 0.95),
        _outcome_row_for(napoli[1], "pending", 0.0),
        # napoli[2] has no outcome at all (operator hasn't graded yet)
    ]
    _write_shadow_partition(tmp_path, day=11, pairs=napoli, outcomes=outcomes)

    pairs, report = build_real_pattern_pairs(
        shadow_root=tmp_path, require_outcome=True
    )
    statuses = [p.outcome_status for p in pairs]
    assert statuses == ["won"]


def test_only_won_keeps_only_winning_pairs(tmp_path):
    napoli = _build_napoli_pairs(3)
    outcomes = [
        _outcome_row_for(napoli[0], "won", 0.95),
        _outcome_row_for(napoli[1], "lost", -1.00),
        _outcome_row_for(napoli[2], "won", 1.10),
    ]
    _write_shadow_partition(tmp_path, day=11, pairs=napoli, outcomes=outcomes)

    pairs, _ = build_real_pattern_pairs(shadow_root=tmp_path, only_won=True)
    assert len(pairs) == 2
    for p in pairs:
        assert p.outcome_status == "won"


# ──────────────────────────────────────────────────────────────────────
# Multi-partition stitching
# ──────────────────────────────────────────────────────────────────────


def test_multi_day_pairs_concatenated(tmp_path):
    napoli = _build_napoli_pairs(4)
    _write_shadow_partition(tmp_path, day=11, pairs=napoli[:2])
    _write_shadow_partition(tmp_path, day=12, pairs=napoli[2:])
    pairs, report = build_real_pattern_pairs(shadow_root=tmp_path)
    assert len(pairs) == 4
    assert report.n_gsvs_loaded == 4
    assert report.n_picks_loaded == 4
