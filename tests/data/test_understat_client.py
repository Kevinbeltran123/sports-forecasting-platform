"""Tests for UnderstatClient — Phase 0.5 Layer 1 (structural).

Uses an in-test fake reader that returns hand-crafted DataFrames matching
soccerdata's expected raw shape. Validates:
  - Canonical schema enforcement (renames + dtype casts)
  - Missing-column failure mode
  - aggregate_xg_last_n correctness (sum, recency window, npxg vs xg)
  - Cache-dir creation on construction

Layer 2 tests (real-data backfill of top-5 EU 2020/21–2025/26) are queued
below as xfail with `# requires-real-data`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from bip.data.understat_client import (
    DEFAULT_DECAY,
    SHOT_EVENT_SCHEMA,
    TEAM_MATCH_STATS_SCHEMA,
    TOP_5_EU_LEAGUES,
    TeamFormFeatures,
    UnderstatClient,
    aggregate_xg_last_n,
    compute_team_form,
)

# ---------------------------------------------------------------------------
# Fakes — mirror what soccerdata.Understat returns, with column names that
# match the *raw* Understat conventions so the normalizer's renames are
# exercised.
# ---------------------------------------------------------------------------

def _fake_shot_events_df() -> pl.DataFrame:
    """Two shots in one Arsenal-Chelsea match — raw Understat columns."""
    return pl.DataFrame({
        "match_id": [1001, 1001],
        "id": [50001, 50002],
        "minute": [12, 67],
        "season": ["2425", "2425"],
        "league": ["ENG-Premier League", "ENG-Premier League"],
        "date": [datetime(2025, 9, 14), datetime(2025, 9, 14)],
        "h_team": ["Arsenal", "Arsenal"],
        "a_team": ["Chelsea", "Chelsea"],
        "h_a": ["h", "a"],
        "player": ["Saka", "Palmer"],
        "player_id": [1001, 2002],
        "X": [0.85, 0.78],
        "Y": [0.45, 0.52],
        "xG": [0.12, 0.34],
        "result": ["SavedShot", "Goal"],
        "situation": ["OpenPlay", "Penalty"],
        "shotType": ["RightFoot", "RightFoot"],
    })


def _fake_team_match_stats_df() -> pl.DataFrame:
    """Three Arsenal matches across two months — raw Understat columns."""
    return pl.DataFrame({
        "match_id": [1001, 1002, 1003, 1001, 1002, 1003],
        "season": ["2425"] * 6,
        "league": ["ENG-Premier League"] * 6,
        "date": [
            datetime(2025, 9, 14),
            datetime(2025, 9, 21),
            datetime(2025, 10, 5),
            datetime(2025, 9, 14),
            datetime(2025, 9, 21),
            datetime(2025, 10, 5),
        ],
        "team": ["Arsenal", "Arsenal", "Arsenal", "Chelsea", "Brighton", "Liverpool"],
        "opponent": ["Chelsea", "Brighton", "Liverpool", "Arsenal", "Arsenal", "Arsenal"],
        "venue": ["home", "away", "home", "away", "home", "away"],
        "goals": [2, 1, 3, 1, 1, 2],
        "xG": [1.85, 1.20, 2.40, 0.95, 1.10, 1.65],
        "npxG": [1.50, 1.20, 2.40, 0.95, 1.10, 1.65],
        "xGA": [0.95, 1.10, 1.65, 1.85, 1.20, 2.40],
        "deep": [12, 8, 15, 6, 9, 11],
        "PPDA": [9.5, 11.2, 8.7, 14.1, 12.8, 10.3],
    })


class _FakeReader:
    """Implements the _UnderstatReader Protocol for tests."""

    def __init__(self, **_: Any) -> None:
        # Accept all kwargs (leagues, seasons, data_dir) without inspecting them.
        pass

    def read_shot_events(self) -> pl.DataFrame:
        return _fake_shot_events_df()

    def read_team_match_stats(self) -> pl.DataFrame:
        return _fake_team_match_stats_df()


def _fake_factory(**kwargs: Any) -> _FakeReader:
    return _FakeReader(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_leagues_is_top_5_eu(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        assert client._leagues == TOP_5_EU_LEAGUES

    def test_cache_dir_created_on_init(self, tmp_path: Path) -> None:
        target = tmp_path / "deeper" / "understat"
        assert not target.exists()
        UnderstatClient(cache_dir=target, _reader_factory=_fake_factory)
        assert target.is_dir()

    def test_seasons_string_promoted_to_tuple(self, tmp_path: Path) -> None:
        client = UnderstatClient(
            seasons="2425", cache_dir=tmp_path, _reader_factory=_fake_factory
        )
        assert client._seasons == ("2425",)

    def test_reader_constructed_lazily(self, tmp_path: Path) -> None:
        """Reader factory not invoked until first read_* call."""
        calls: list[dict[str, Any]] = []

        def tracking_factory(**kwargs: Any) -> _FakeReader:
            calls.append(kwargs)
            return _FakeReader()

        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=tracking_factory)
        assert calls == []  # not yet
        client.read_shot_events()
        assert len(calls) == 1
        client.read_team_match_stats()
        assert len(calls) == 1  # reused, not reconstructed


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

class TestShotEventNormalization:
    def test_returns_canonical_columns_in_order(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_shot_events()
        assert df.columns == list(SHOT_EVENT_SCHEMA)

    def test_dtypes_match_canonical_schema(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_shot_events()
        for col, expected_dtype in SHOT_EVENT_SCHEMA.items():
            assert df.schema[col] == expected_dtype, (
                f"Column {col} expected {expected_dtype}, got {df.schema[col]}"
            )

    def test_is_penalty_derived_from_situation(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_shot_events()
        # Row 0: OpenPlay → False; Row 1: Penalty → True
        assert df["is_penalty"].to_list() == [False, True]

    def test_missing_canonical_column_raises(self, tmp_path: Path) -> None:
        class BadReader:
            def read_shot_events(self) -> pl.DataFrame:
                return pl.DataFrame({"match_id": [1], "id": [50001]})

            def read_team_match_stats(self) -> pl.DataFrame:
                return pl.DataFrame()

        client = UnderstatClient(
            cache_dir=tmp_path, _reader_factory=lambda **_: BadReader()
        )
        with pytest.raises(ValueError, match="missing canonical columns"):
            client.read_shot_events()


class TestTeamMatchStatsNormalization:
    def test_returns_canonical_columns(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        assert df.columns == list(TEAM_MATCH_STATS_SCHEMA)

    def test_dtypes_match_canonical_schema(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        for col, expected_dtype in TEAM_MATCH_STATS_SCHEMA.items():
            assert df.schema[col] == expected_dtype


# ---------------------------------------------------------------------------
# Aggregation helper — feeds xg_total_l5 covariate
# ---------------------------------------------------------------------------

class TestAggregateXgLastN:
    def test_default_uses_npxg_and_sums_all_in_window(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        # Arsenal has 3 matches; npxG values 1.50, 1.20, 2.40
        total = aggregate_xg_last_n(df, team="Arsenal", as_of=datetime(2025, 11, 1), n=5)
        assert total == pytest.approx(1.50 + 1.20 + 2.40, abs=1e-4)

    def test_n_truncates_window(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        # n=2 should pick the 2 most recent: 2.40 (Oct 5) + 1.20 (Sep 21) = 3.60
        total = aggregate_xg_last_n(df, team="Arsenal", as_of=datetime(2025, 11, 1), n=2)
        assert total == pytest.approx(2.40 + 1.20, abs=1e-4)

    def test_as_of_excludes_same_day_matches(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        # as_of = Oct 5 — Oct 5 match itself is excluded (strict <)
        total = aggregate_xg_last_n(df, team="Arsenal", as_of=datetime(2025, 10, 5), n=5)
        assert total == pytest.approx(1.50 + 1.20, abs=1e-4)

    def test_use_npxg_false_sums_xg_with_penalty(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        # xG (with penalty contribution) for Arsenal: 1.85 + 1.20 + 2.40
        total = aggregate_xg_last_n(
            df, team="Arsenal", as_of=datetime(2025, 11, 1), n=5, use_npxg=False
        )
        assert total == pytest.approx(1.85 + 1.20 + 2.40, abs=1e-4)

    def test_unknown_team_returns_zero(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        assert aggregate_xg_last_n(df, team="DoesNotExist", as_of=datetime.now()) == 0.0


# ---------------------------------------------------------------------------
# TeamFormFeatures — multi-dimensional form vector (the realistic feature)
# ---------------------------------------------------------------------------

@pytest.fixture
def arsenal_form(tmp_path: Path) -> TeamFormFeatures:
    """Arsenal form features at 2025-11-01 (after all 3 fixture matches)."""
    client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
    df = client.read_team_match_stats()
    return compute_team_form(df, team="Arsenal", as_of=datetime(2025, 11, 1))


class TestTeamFormVolume:
    def test_npxg_total_l5_sums_all_available(self, arsenal_form: TeamFormFeatures) -> None:
        # Arsenal has 3 matches; npxG: 1.50, 1.20, 2.40
        assert arsenal_form.npxg_total_l5 == pytest.approx(5.10, abs=1e-4)

    def test_npxg_total_l3_sums_top3_when_available(
        self, arsenal_form: TeamFormFeatures
    ) -> None:
        assert arsenal_form.npxg_total_l3 == pytest.approx(5.10, abs=1e-4)

    def test_xg_total_l5_includes_penalty_contribution(
        self, arsenal_form: TeamFormFeatures
    ) -> None:
        # xG includes penalties: 1.85 + 1.20 + 2.40
        assert arsenal_form.xg_total_l5 == pytest.approx(5.45, abs=1e-4)

    def test_xg_against_total_l5(self, arsenal_form: TeamFormFeatures) -> None:
        # xGA from fixtures: 0.95 + 1.10 + 1.65
        assert arsenal_form.xg_against_total_l5 == pytest.approx(3.70, abs=1e-4)

    def test_n_matches_used_caps_at_actual_history(
        self, arsenal_form: TeamFormFeatures
    ) -> None:
        assert arsenal_form.n_matches_used == 3


class TestTeamFormRecency:
    def test_decay_weights_recent_matches_higher(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        form = compute_team_form(df, team="Arsenal", as_of=datetime(2025, 11, 1))
        # Sorted descending by date: Oct 5 (npxG 2.40), Sep 21 (1.20), Sep 14 (1.50)
        # decay_l5 = 2.40*β^0 + 1.20*β^1 + 1.50*β^2
        beta = DEFAULT_DECAY
        expected = 2.40 * (beta**0) + 1.20 * (beta**1) + 1.50 * (beta**2)
        assert form.npxg_decay_l5 == pytest.approx(expected, abs=1e-4)

    def test_decay_beta_is_recorded(self, arsenal_form: TeamFormFeatures) -> None:
        assert arsenal_form.decay_beta == DEFAULT_DECAY

    def test_decay_with_beta_one_equals_simple_sum(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        form = compute_team_form(
            df, team="Arsenal", as_of=datetime(2025, 11, 1), decay_beta=1.0
        )
        assert form.npxg_decay_l5 == pytest.approx(form.npxg_total_l5, abs=1e-4)


class TestTeamFormVenue:
    def test_home_per_match_average(self, arsenal_form: TeamFormFeatures) -> None:
        # Arsenal home: Sep 14 (1.50) + Oct 5 (2.40) → mean = 1.95
        assert arsenal_form.npxg_per_match_home_l5 == pytest.approx(1.95, abs=1e-4)

    def test_away_per_match_average(self, arsenal_form: TeamFormFeatures) -> None:
        # Arsenal away: only Sep 21 (1.20)
        assert arsenal_form.npxg_per_match_away_l5 == pytest.approx(1.20, abs=1e-4)


class TestTeamFormVariance:
    def test_stdev_nonzero_with_spread(self, arsenal_form: TeamFormFeatures) -> None:
        # npxG = [2.40, 1.20, 1.50] (descending date order)
        # population stdev across these three values
        assert arsenal_form.npxg_stdev_l5 > 0.0

    def test_stdev_zero_with_single_match(self, tmp_path: Path) -> None:
        # Build a DataFrame already in canonical (post-normalization) schema
        single_match = pl.DataFrame(
            {
                "match_id": [1],
                "season": ["2425"],
                "league": ["ENG-Premier League"],
                "date": [datetime(2025, 8, 1)],
                "team": ["Solo"],
                "opponent": ["Other"],
                "venue": ["home"],
                "goals": [1],
                "xg": [1.5],
                "npxg": [1.5],
                "xg_against": [0.8],
                "deep": [10],
                "ppda": [10.0],
            },
            schema=TEAM_MATCH_STATS_SCHEMA,  # type: ignore[arg-type]
        )
        form = compute_team_form(single_match, team="Solo", as_of=datetime(2025, 12, 1))
        assert form.npxg_stdev_l5 == 0.0
        assert form.n_matches_used == 1


class TestTeamFormQuality:
    def test_xg_diff_l5_is_npxg_minus_xga(self, arsenal_form: TeamFormFeatures) -> None:
        # 5.10 npxG - 3.70 xGA = 1.40
        assert arsenal_form.xg_diff_l5 == pytest.approx(1.40, abs=1e-4)


class TestTeamFormEdgeCases:
    def test_unknown_team_returns_zeros(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        form = compute_team_form(df, team="DoesNotExist", as_of=datetime.now())
        assert form.n_matches_used == 0
        assert form.npxg_total_l5 == 0.0
        assert form.npxg_decay_l5 == 0.0
        assert form.npxg_stdev_l5 == 0.0
        assert form.xg_normalized_l5 is None  # Layer 2 placeholder

    def test_as_of_excludes_same_day_matches(self, tmp_path: Path) -> None:
        client = UnderstatClient(cache_dir=tmp_path, _reader_factory=_fake_factory)
        df = client.read_team_match_stats()
        form = compute_team_form(df, team="Arsenal", as_of=datetime(2025, 10, 5))
        # Oct 5 excluded; only Sep 14 + Sep 21 remain
        assert form.n_matches_used == 2
        assert form.npxg_total_l5 == pytest.approx(1.50 + 1.20, abs=1e-4)

    def test_as_dict_returns_flat_dict(self, arsenal_form: TeamFormFeatures) -> None:
        d = arsenal_form.as_dict()
        # All TeamFormFeatures fields appear as flat keys for direct DataFrame insertion
        assert "npxg_total_l5" in d
        assert "npxg_decay_l5" in d
        assert "xg_diff_l5" in d
        assert "n_matches_used" in d


# ---------------------------------------------------------------------------
# Layer 2 — queued real-data validations (require Understat backfill)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires-real-data: Understat backfill not loaded yet")
def test_layer2_real_backfill_top5_eu_5_seasons() -> None:
    """End-to-end: pull top-5 EU 2020/21–2025/26 and assert non-empty per league.

    Unblocks when:
      - `uv sync` has installed soccerdata
      - First run of `python -m bip.data.understat_client` (or a backfill
        script) populates `data/cache/understat/`
    """
    raise NotImplementedError("queued — see Phase 0.5 Layer 2")


@pytest.mark.xfail(reason="requires-real-data: corners model integration pending")
def test_layer2_xg_total_l5_feeds_corners_predictor() -> None:
    """Smoke: corners_poisson can consume aggregate_xg_last_n() output.

    Unblocks when corners_poisson signature is extended to accept the new
    covariate (this is v1.0 Phase 6 work, not part of the spike critical path).
    """
    raise NotImplementedError("queued — see SYNTHESIS.md Conclusion 2")
