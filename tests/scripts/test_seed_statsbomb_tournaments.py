"""Tests for scripts/seed_statsbomb_tournaments.py — Phase 5 Layer-2 ingest.

Layer-1 tests (always run): URL builders, event parser correctness on
synthetic mini-event-streams, match-outcome aggregation properties.

Layer-2 tests (skipif when cache missing): real-data ingest validation
once the operator has run ``scripts/seed_statsbomb_tournaments.py``.
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts.seed_statsbomb_tournaments import (
    DEFAULT_OUTCOMES_PARQUET,
    SUPPORTED_TOURNAMENTS,
    StatsBombMatchOutcome,
    StatsBombTournament,
    aggregate_match_to_outcome,
    count_corners_per_team,
)

# ── URL builder + tournament constants ───────────────────────────────────────


class TestStatsBombTournament:
    def test_matches_url_format(self):
        t = StatsBombTournament(43, 3, "wc_2018", "FIFA World Cup 2018")
        assert t.matches_url.endswith("/matches/43/3.json")

    def test_events_url_format(self):
        t = StatsBombTournament(43, 3, "wc_2018", "FIFA World Cup 2018")
        assert t.events_url(7563).endswith("/events/7563.json")

    def test_supported_tournaments_includes_all_six_modern_men_intl(self):
        slugs = {t.slug for t in SUPPORTED_TOURNAMENTS}
        assert {
            "wc_2018",
            "wc_2022",
            "euro_2020",
            "euro_2024",
            "copa_2024",
            "afcon_2023",
        } <= slugs

    def test_default_held_out_match_supported_tournaments(self):
        """Spike doc operator-approved held-out (Copa 2024, Euro 2024) must
        be present in the SUPPORTED_TOURNAMENTS list — Phase 5 backtest
        depends on this."""
        from scripts.backtest_int_tournaments import DEFAULT_HELD_OUT_TOURNAMENTS

        slugs = {t.slug for t in SUPPORTED_TOURNAMENTS}
        for held in DEFAULT_HELD_OUT_TOURNAMENTS:
            assert held in slugs, f"held-out '{held}' missing from SUPPORTED_TOURNAMENTS"


# ── Corner counting from synthetic event streams ─────────────────────────────


def _pass_corner(team: str) -> dict:
    return {
        "type": {"name": "Pass"},
        "pass": {"type": {"name": "Corner"}},
        "team": {"name": team},
    }


def _other_pass(team: str) -> dict:
    return {
        "type": {"name": "Pass"},
        "pass": {"type": {"name": "Free Kick"}},  # not a corner
        "team": {"name": team},
    }


def _shot(team: str) -> dict:
    return {"type": {"name": "Shot"}, "team": {"name": team}}


class TestCornerCounting:
    def test_counts_corners_per_team(self):
        events = [
            _pass_corner("Argentina"),
            _pass_corner("Argentina"),
            _pass_corner("Brazil"),
            _other_pass("Argentina"),  # ignored
            _shot("Brazil"),  # ignored
        ]
        counts = count_corners_per_team(events)
        assert counts == {"Argentina": 2, "Brazil": 1}

    def test_zero_corners_returns_empty_dict(self):
        events = [_other_pass("X"), _shot("Y")]
        assert count_corners_per_team(events) == {}

    def test_handles_missing_pass_type(self):
        """A Pass event without pass.type field should be silently ignored."""
        events = [
            {"type": {"name": "Pass"}, "team": {"name": "X"}},  # no pass dict
            _pass_corner("Y"),
        ]
        counts = count_corners_per_team(events)
        assert counts == {"Y": 1}

    def test_handles_missing_team(self):
        """Defensive: pass-corner event with no team field should not crash."""
        events = [
            {"type": {"name": "Pass"}, "pass": {"type": {"name": "Corner"}}},
            _pass_corner("Y"),
        ]
        counts = count_corners_per_team(events)
        assert counts == {"Y": 1}


# ── Match-outcome aggregation ────────────────────────────────────────────────


class TestMatchOutcomeAggregation:
    def _meta(self, home: str, away: str, hg: int, ag: int) -> dict:
        return {
            "match_id": 9999,
            "home_team": {"home_team_name": home},
            "away_team": {"away_team_name": away},
            "home_score": hg,
            "away_score": ag,
            "match_date": "2024-07-10",
        }

    def test_aggregates_match_with_corners(self):
        meta = self._meta("Argentina", "Brazil", 2, 1)
        events = [
            _pass_corner("Argentina"),
            _pass_corner("Argentina"),
            _pass_corner("Brazil"),
            _pass_corner("Brazil"),
            _pass_corner("Brazil"),
        ]
        outcome = aggregate_match_to_outcome(meta, events, "copa_2024")
        assert outcome.home_team == "Argentina"
        assert outcome.away_team == "Brazil"
        assert outcome.home_goals == 2
        assert outcome.away_goals == 1
        assert outcome.home_corners == 2
        assert outcome.away_corners == 3
        assert outcome.tournament_slug == "copa_2024"
        assert outcome.match_date == date(2024, 7, 10)
        assert outcome.match_id == 9999

    def test_zero_corners_gracefully_handled(self):
        outcome = aggregate_match_to_outcome(
            self._meta("X", "Y", 0, 0), events=[], tournament_slug="wc_2018"
        )
        assert outcome.home_corners == 0
        assert outcome.away_corners == 0

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            aggregate_match_to_outcome(
                {**self._meta("X", "Y", 1, 0), "match_date": "not-a-date"},
                events=[],
                tournament_slug="x",
            )


# ── StatsBombMatchOutcome derived properties ────────────────────────────────


class TestOutcomeProperties:
    def _make(self, hg: int, ag: int, hc: int = 5, ac: int = 5) -> StatsBombMatchOutcome:
        return StatsBombMatchOutcome(
            match_id=1,
            tournament_slug="wc_2018",
            match_date=date(2018, 6, 14),
            home_team="A",
            away_team="B",
            home_goals=hg,
            away_goals=ag,
            home_corners=hc,
            away_corners=ac,
        )

    def test_total_goals(self):
        assert self._make(2, 1).total_goals == 3

    def test_total_corners(self):
        assert self._make(0, 0, hc=4, ac=7).total_corners == 11

    def test_btts_yes(self):
        assert self._make(1, 1).btts == 1

    def test_btts_no_when_one_side_zero(self):
        assert self._make(2, 0).btts == 0
        assert self._make(0, 3).btts == 0

    def test_outcome_1x2(self):
        assert self._make(2, 1).outcome_1x2 == 0  # home win
        assert self._make(1, 1).outcome_1x2 == 1  # draw
        assert self._make(0, 2).outcome_1x2 == 2  # away win


# ── Layer-2: real-data ingest validation ─────────────────────────────────────


def _real_data_available() -> bool:
    return DEFAULT_OUTCOMES_PARQUET.exists()


@pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Run `uv run python scripts/seed_statsbomb_tournaments.py` to "
        "download the StatsBomb open-data and unblock this Layer-2 test."
    ),
)
class TestRealDataLayer2:
    def test_outcomes_parquet_loads_with_expected_volume(self):
        from scripts.seed_statsbomb_tournaments import load_outcomes_from_parquet

        outcomes = load_outcomes_from_parquet()
        # 6 tournaments × ~50 matches each ≈ 300 minimum
        assert len(outcomes) >= 250
        assert len(outcomes) <= 500

    def test_all_six_tournaments_present(self):
        from scripts.seed_statsbomb_tournaments import load_outcomes_from_parquet

        outcomes = load_outcomes_from_parquet()
        slugs = {o.tournament_slug for o in outcomes}
        assert {
            "wc_2018",
            "wc_2022",
            "euro_2020",
            "euro_2024",
            "copa_2024",
            "afcon_2023",
        } <= slugs

    def test_corner_counts_are_plausible(self):
        """Sanity: average corners per match should be in [4, 16] range
        (real football matches have ~4-12 total corners; <4 suggests
        an aggregation bug, >16 is unrealistic)."""
        from scripts.seed_statsbomb_tournaments import load_outcomes_from_parquet

        outcomes = load_outcomes_from_parquet()
        avg_corners = sum(o.total_corners for o in outcomes) / len(outcomes)
        assert 4.0 <= avg_corners <= 16.0, (
            f"Average corners per match = {avg_corners:.2f} is outside "
            f"the plausible [4, 16] range — likely an aggregation bug."
        )

    def test_goal_counts_match_expected_distribution(self):
        """Sanity: average goals per match in international tournaments
        is ~2.5 (range 2.0-3.5 across the 6 modern tournaments)."""
        from scripts.seed_statsbomb_tournaments import load_outcomes_from_parquet

        outcomes = load_outcomes_from_parquet()
        avg_goals = sum(o.total_goals for o in outcomes) / len(outcomes)
        assert 2.0 <= avg_goals <= 3.5, (
            f"Average goals per match = {avg_goals:.2f} is outside the "
            f"plausible [2.0, 3.5] range."
        )
