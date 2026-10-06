"""Ola 0 — corpus loader integrity tests.

Verifies the three-way split (train/calibration/heldout) used by
``bip.evaluation.tournaments.wc2026_v2``:

- Disjoint by source identity (martj42 train cannot overlap StatsBomb
  calibration/heldout, since they live in separate tables; but the
  *content* must not overlap by date+team)
- Train excludes the four hold-out tournament windows
- Calibration = wc_2018 + euro_2020; Heldout = wc_2022 + afcon_2023 +
  euro_2024 + copa_2024
- Schema invariants per split
- Match counts match StatsBomb's known per-tournament counts
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bip.evaluation.tournaments.wc2026_v2.corpus import (
    CALIBRATION_SLUGS,
    HELD_OUT_SLUGS,
    CorpusSplit,
    _martj42_held_out_mask,
    build_split,
    load_martj42,
    load_statsbomb,
)

# ─────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────


@pytest.mark.requires_data
class TestLoaders:
    def test_martj42_loads(self) -> None:
        df = load_martj42()
        # Schema invariants the rest of the spike relies on.
        expected = {
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        }
        assert expected.issubset(set(df.columns))
        assert df.schema["date"] == pl.Date
        assert df.schema["home_score"] == pl.Int64
        assert df.schema["neutral"] == pl.Boolean
        # No null scores after drop_nulls.
        assert df["home_score"].null_count() == 0
        assert df["away_score"].null_count() == 0

    def test_statsbomb_loads(self) -> None:
        df = load_statsbomb()
        expected = {
            "match_id",
            "tournament_slug",
            "match_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
        }
        assert expected.issubset(set(df.columns))
        # 314 matches total across 6 tournaments per inventory.
        assert df.height == 314


# ─────────────────────────────────────────────────────────────────
# Hold-out mask (the load-bearing primitive)
# ─────────────────────────────────────────────────────────────────


class TestHeldOutMask:
    """Each parametrized case targets a boundary: window start, window end,
    one day outside (left and right), and a tournament-name confound."""

    @pytest.mark.parametrize(
        ("match_date", "tournament", "expected"),
        [
            # WC 2022 window [2022-11-20, 2022-12-18]
            (date(2022, 11, 20), "FIFA World Cup", True),
            (date(2022, 12, 18), "FIFA World Cup", True),
            (date(2022, 11, 19), "FIFA World Cup", False),
            (date(2022, 12, 19), "FIFA World Cup", False),
            (date(2022, 11, 25), "FIFA World Cup qualification", False),
            # AFCON 2023 (played Jan–Feb 2024) [2024-01-13, 2024-02-11]
            (date(2024, 1, 13), "African Cup of Nations", True),
            (date(2024, 2, 11), "African Cup of Nations", True),
            (date(2024, 1, 12), "African Cup of Nations", False),
            (date(2024, 2, 12), "African Cup of Nations", False),
            # Euro 2024 [2024-06-14, 2024-07-14]
            (date(2024, 6, 14), "UEFA Euro", True),
            (date(2024, 7, 14), "UEFA Euro", True),
            (date(2024, 7, 15), "UEFA Euro", False),
            # Copa 2024 [2024-06-21, 2024-07-15]
            (date(2024, 6, 21), "Copa América", True),
            (date(2024, 7, 15), "Copa América", True),
            (date(2024, 6, 20), "Copa América", False),
            # Different tournament inside someone else's window — must be False
            (date(2022, 11, 25), "Friendly", False),
            (date(2024, 1, 20), "AFC Asian Cup", False),
        ],
    )
    def test_window_membership(self, match_date: date, tournament: str, expected: bool) -> None:
        df = pl.DataFrame(
            {
                "date": [match_date],
                "tournament": [tournament],
            }
        )
        mask = _martj42_held_out_mask(df)
        assert bool(mask[0]) is expected

    def test_empty_frame_returns_empty_mask(self) -> None:
        df = pl.DataFrame(
            {"date": pl.Series([], dtype=pl.Date), "tournament": pl.Series([], dtype=pl.String)}
        )
        mask = _martj42_held_out_mask(df)
        assert mask.len() == 0


# ─────────────────────────────────────────────────────────────────
# build_split
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def split() -> CorpusSplit:
    return build_split()


@pytest.mark.requires_data
class TestSplit:
    def test_returns_corpus_split(self, split: CorpusSplit) -> None:
        assert isinstance(split, CorpusSplit)

    def test_train_excludes_held_out_windows(self, split: CorpusSplit) -> None:
        """Walking the four windows against `train` must find zero rows."""
        violations = _martj42_held_out_mask(split.train).sum()
        assert violations == 0

    def test_calibration_only_wc2018_and_euro2020(self, split: CorpusSplit) -> None:
        slugs = set(split.calibration["tournament_slug"].unique().to_list())
        assert slugs == set(CALIBRATION_SLUGS)

    def test_heldout_only_four_target_tournaments(self, split: CorpusSplit) -> None:
        slugs = set(split.heldout["tournament_slug"].unique().to_list())
        assert slugs == set(HELD_OUT_SLUGS)

    def test_no_statsbomb_tournament_in_both_calibration_and_heldout(
        self, split: CorpusSplit
    ) -> None:
        cal_slugs = set(split.calibration["tournament_slug"].unique().to_list())
        held_slugs = set(split.heldout["tournament_slug"].unique().to_list())
        assert cal_slugs.isdisjoint(held_slugs)

    def test_known_match_counts(self, split: CorpusSplit) -> None:
        """Per StatsBomb inventory: wc_2018=64, euro_2020=51 → calibration=115;
        wc_2022=64, afcon_2023=52, copa_2024=32, euro_2024=51 → heldout=199."""
        assert split.calibration.height == 115
        assert split.heldout.height == 199

    def test_train_size_within_expected_range(self, split: CorpusSplit) -> None:
        """martj42 has ~49,328 raw rows; 72 dropped for null scores; ~199
        matches excluded by hold-out windows (those windows also contain
        non-target friendlies but it's negligible). Expect train ≥ 48,800."""
        assert split.train.height >= 48_800
        assert split.train.height <= 49_400

    def test_summary_keys_present(self, split: CorpusSplit) -> None:
        s = split.summary()
        assert set(s.keys()) == {"train", "calibration", "heldout"}
        for v in s.values():
            assert "n_matches" in v
            assert "date_min" in v
            assert "date_max" in v


# ─────────────────────────────────────────────────────────────────
# Reproducibility — build_split must be deterministic
# ─────────────────────────────────────────────────────────────────


@pytest.mark.requires_data
class TestReproducibility:
    def test_two_builds_identical(self) -> None:
        a = build_split()
        b = build_split()
        assert a.train.height == b.train.height
        assert a.calibration.height == b.calibration.height
        assert a.heldout.height == b.heldout.height

    def test_caller_provided_frames_used(self) -> None:
        """Smoke test: passing custom frames bypasses default I/O."""
        martj42 = pl.DataFrame(
            {
                "date": [date(2010, 1, 1), date(2022, 12, 1)],
                "home_team": ["A", "B"],
                "away_team": ["C", "D"],
                "home_score": [1, 2],
                "away_score": [0, 1],
                "tournament": ["Friendly", "FIFA World Cup"],
                "neutral": [False, True],
            }
        )
        statsbomb = pl.DataFrame(
            {
                "match_id": [1, 2],
                "tournament_slug": ["wc_2018", "wc_2022"],
                "match_date": [date(2018, 6, 14), date(2022, 11, 20)],
                "home_team": ["X", "Y"],
                "away_team": ["Z", "W"],
                "home_goals": [1, 2],
                "away_goals": [0, 1],
                "home_corners": [5, 6],
                "away_corners": [3, 4],
                "home_xg": [1.2, 1.8],
                "away_xg": [0.9, 1.1],
                "home_shots": [10, 12],
                "away_shots": [8, 9],
            }
        )
        split = build_split(martj42, statsbomb)
        # WC2022 row should be excluded from train (held-out window).
        assert split.train.height == 1
        assert split.train["tournament"][0] == "Friendly"
        # StatsBomb wc_2018 → calibration; wc_2022 → heldout.
        assert split.calibration.height == 1
        assert split.heldout.height == 1
