"""Tests for tsv_calculator — full TSV computation from synthetic data."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.api_football_parsers import (
    Event,
    EventTime,
    Fixture,
    FixtureCore,
    FixtureGoals,
    FixtureLeague,
    FixtureScore,
    FixtureStatus,
    FixtureTeam,
    FixtureTeams,
    StatItem,
    TeamStatsBlock,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_calculator import (
    MatchFeatures,
    _bootstrap_ci,
    _per15_from_minutes,
    compute_tsv,
    extract_match_features,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import CoachInfo


# ─── Helpers ────────────────────────────────────────────────────────────


def _fixture(fid: int, home_id: int, away_id: int, hg: int, ag: int) -> Fixture:
    return Fixture(
        fixture=FixtureCore(
            id=fid,
            date=datetime(2024, 6, 1),
            status=FixtureStatus(long="Match Finished", short="FT", elapsed=90),
        ),
        league=FixtureLeague(id=1, name="Test", season=2024),
        teams=FixtureTeams(
            home=FixtureTeam(id=home_id, name=f"Home{home_id}"),
            away=FixtureTeam(id=away_id, name=f"Away{away_id}"),
        ),
        goals=FixtureGoals(home=hg, away=ag),
        score=FixtureScore(
            halftime=FixtureGoals(home=0, away=0),
            fulltime=FixtureGoals(home=hg, away=ag),
        ),
    )


def _stats_block(team_id: int, **stat_kwargs) -> TeamStatsBlock:
    items = []
    for k, v in stat_kwargs.items():
        # Map our snake_case kwargs to API-Football's natural-case "type"
        natural = {
            "shots": "Total Shots",
            "shots_on_target": "Shots on Goal",
            "possession": "Ball Possession",
            "corners": "Corner Kicks",
            "fouls": "Fouls",
            "offsides": "Offsides",
        }.get(k, k)
        items.append(StatItem(type=natural, value=v))
    return TeamStatsBlock(
        team=FixtureTeam(id=team_id, name=f"T{team_id}"), statistics=items
    )


def _goal_event(team_id: int, minute: int) -> Event:
    return Event(
        time=EventTime(elapsed=minute, extra=None),
        team=FixtureTeam(id=team_id, name=f"T{team_id}"),
        type="Goal",
        detail="Normal Goal",
    )


def _conf_lookup(team_id: int) -> str:
    mapping = {1: "UEFA", 2: "CONMEBOL", 3: "CAF", 4: "AFC", 5: "CONCACAF"}
    return mapping.get(team_id, "UNK")


# ─── Bootstrap helper tests ─────────────────────────────────────────────


class TestBootstrapCi:
    def test_constant_array_has_zero_width_ci(self) -> None:
        d = _bootstrap_ci([2.0, 2.0, 2.0, 2.0, 2.0])
        assert d.mean == pytest.approx(2.0)
        assert d.ci_width == pytest.approx(0.0)
        assert d.n == 5

    def test_empty_input(self) -> None:
        d = _bootstrap_ci([])
        assert d.n == 0
        assert d.mean == 0.0

    def test_deterministic_with_seed(self) -> None:
        d1 = _bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        d2 = _bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert d1.ci_low == d2.ci_low
        assert d1.ci_high == d2.ci_high


# ─── Per-match extraction tests ─────────────────────────────────────────


class TestExtractMatchFeatures:
    def test_basic_home_win(self) -> None:
        fix = _fixture(fid=1, home_id=26, away_id=2, hg=3, ag=1)
        team_stats = _stats_block(26, shots=15, shots_on_target=6, possession="60%", corners=7, fouls=10, offsides=1)
        opp_stats = _stats_block(2, shots=8, shots_on_target=3, possession="40%", corners=3, fouls=15, offsides=4)
        events = [
            _goal_event(26, 12),
            _goal_event(26, 34),
            _goal_event(2, 50),  # opponent goal
            _goal_event(26, 78),
        ]
        m = extract_match_features(
            fix, team_id=26, team_stats=team_stats,
            opponent_stats=opp_stats, events=events,
            opponent_confederation_lookup=_conf_lookup,
        )
        assert m.goals_for == 3
        assert m.goals_against == 1
        assert m.goals_for_minutes == (12, 34, 78)
        assert m.goals_against_minutes == (50,)
        assert m.shots == 15
        assert m.shots_on_target == 6
        assert m.corners_for == 7
        # corners_against comes from opp_stats
        assert m.corners_against == 3
        assert m.possession == pytest.approx(60.0)
        # offsides_against = opponent's offsides
        assert m.offsides_against == 4
        assert m.btts is True
        assert m.over_25 is True
        assert m.over_35 is True
        assert m.clean_sheet is False

    def test_away_team_perspective(self) -> None:
        # Fixture: home=1 (UEFA), away=26 (CONMEBOL); we profile team 26 as away.
        fix = _fixture(fid=2, home_id=1, away_id=26, hg=1, ag=2)
        away_stats = _stats_block(26, shots=10)
        home_stats = _stats_block(1, shots=12)
        m = extract_match_features(
            fix, team_id=26,
            team_stats=away_stats, opponent_stats=home_stats,
            events=[_goal_event(26, 20), _goal_event(1, 40), _goal_event(26, 80)],
            opponent_confederation_lookup=_conf_lookup,
        )
        assert m.goals_for == 2
        assert m.goals_against == 1
        assert m.opponent_team_id == 1
        assert m.opponent_confederation == "UEFA"

    def test_team_not_in_fixture_raises(self) -> None:
        fix = _fixture(fid=99, home_id=1, away_id=2, hg=0, ag=0)
        with pytest.raises(ValueError):
            extract_match_features(
                fix, team_id=99, team_stats=None, opponent_stats=None,
                events=[], opponent_confederation_lookup=_conf_lookup,
            )

    def test_clean_sheet(self) -> None:
        fix = _fixture(fid=3, home_id=26, away_id=2, hg=2, ag=0)
        m = extract_match_features(
            fix, team_id=26, team_stats=None, opponent_stats=None,
            events=[_goal_event(26, 30), _goal_event(26, 60)],
            opponent_confederation_lookup=_conf_lookup,
        )
        assert m.clean_sheet is True
        assert m.btts is False


# ─── Per-15-min distribution tests ──────────────────────────────────────


class TestPer15FromMinutes:
    def test_goals_distributed(self) -> None:
        matches = [
            MatchFeatures(
                fixture_id=1, opponent_team_id=2, opponent_confederation="UEFA",
                goals_for=2, goals_against=0,
                goals_for_minutes=(10, 67),
                goals_against_minutes=(),
                shots=None, shots_on_target=None, corners_for=None,
                corners_against=None, possession=None, fouls=None,
                yellow_cards=0, red_cards=0, offsides_against=None,
            ),
            MatchFeatures(
                fixture_id=2, opponent_team_id=2, opponent_confederation="UEFA",
                goals_for=1, goals_against=0,
                goals_for_minutes=(89,),
                goals_against_minutes=(),
                shots=None, shots_on_target=None, corners_for=None,
                corners_against=None, possession=None, fouls=None,
                yellow_cards=0, red_cards=0, offsides_against=None,
            ),
        ]
        g15 = _per15_from_minutes(matches, "for")
        # Bucket 0-15: 1 goal (the 10') in match 1 — mean across 2 matches = 0.5
        assert g15.bucket_0_14.mean == pytest.approx(0.5)
        # Bucket 60-75: 1 goal (67') in match 1 — mean = 0.5
        assert g15.bucket_60_74.mean == pytest.approx(0.5)
        # Bucket 75-90: 1 goal (89') in match 2 — mean = 0.5
        assert g15.bucket_75_90.mean == pytest.approx(0.5)


# ─── Full TSV computation tests ─────────────────────────────────────────


class TestComputeTsv:
    def _build_matches(self, n: int = 12) -> list[MatchFeatures]:
        # Construct n matches; team_id=26 (Argentina-ish) vs UEFA opponents
        out = []
        for i in range(n):
            out.append(
                MatchFeatures(
                    fixture_id=1000 + i,
                    opponent_team_id=1,
                    opponent_confederation="UEFA",
                    goals_for=2 if i % 2 == 0 else 1,
                    goals_against=1 if i % 3 == 0 else 0,
                    goals_for_minutes=(20, 65) if i % 2 == 0 else (40,),
                    goals_against_minutes=(80,) if i % 3 == 0 else (),
                    shots=12 + i % 3, shots_on_target=5,
                    corners_for=6, corners_against=3,
                    possession=58.0,
                    fouls=11, yellow_cards=2, red_cards=0,
                    offsides_against=2,
                )
            )
        return out

    def test_green_flag_at_n10(self) -> None:
        tsv = compute_tsv(
            team_name="Argentina",
            api_football_team_id=26,
            confederation="CONMEBOL",
            coach=CoachInfo(coach_name="Scaloni", start_date=date(2018, 8, 23)),
            matches=self._build_matches(10),
        )
        assert tsv.flag == "green"
        assert tsv.n_matches == 10
        assert tsv.is_bettable is True

    def test_yellow_flag(self) -> None:
        tsv = compute_tsv(
            team_name="X", api_football_team_id=999, confederation="UEFA",
            coach=CoachInfo(coach_name="C", start_date=date(2025, 1, 1)),
            matches=self._build_matches(7),
        )
        assert tsv.flag == "yellow"
        assert tsv.is_bettable is False

    def test_red_flag(self) -> None:
        tsv = compute_tsv(
            team_name="X", api_football_team_id=999, confederation="UEFA",
            coach=CoachInfo(coach_name="C", start_date=date(2025, 6, 1)),
            matches=self._build_matches(3),
        )
        assert tsv.flag == "red"
        assert tsv.is_bettable is False

    def test_sub_profile_created_when_n_ge_3(self) -> None:
        tsv = compute_tsv(
            team_name="Argentina", api_football_team_id=26, confederation="CONMEBOL",
            coach=CoachInfo(coach_name="Scaloni", start_date=date(2018, 8, 23)),
            matches=self._build_matches(12),  # all opponent_conf=UEFA
        )
        assert "UEFA" in tsv.sub_profiles
        assert tsv.sub_profiles["UEFA"].n_matches_vs_conf == 12

    def test_sub_profile_skipped_when_n_lt_3(self) -> None:
        matches = self._build_matches(5)
        # Manually mutate to one UEFA + four CAF (so UEFA < 3)
        from dataclasses import replace
        matches[0] = replace(matches[0], opponent_confederation="UEFA")
        for i in range(1, 5):
            matches[i] = replace(matches[i], opponent_confederation="CAF")
        tsv = compute_tsv(
            team_name="X", api_football_team_id=999, confederation="UEFA",
            coach=CoachInfo(coach_name="C", start_date=date(2024, 1, 1)),
            matches=matches,
        )
        assert "UEFA" not in tsv.sub_profiles  # only 1 match
        assert "CAF" in tsv.sub_profiles  # 4 matches

    def test_mean_total_goals_reasonable(self) -> None:
        tsv = compute_tsv(
            team_name="Argentina", api_football_team_id=26, confederation="CONMEBOL",
            coach=CoachInfo(coach_name="Scaloni", start_date=date(2018, 8, 23)),
            matches=self._build_matches(15),
        )
        # 15 matches: half score 2 goals (alternating), avg goals_for ≈ 1.53
        # avg goals_against ≈ 0.33 (i % 3 == 0 contributes)
        # total ≈ 1.86
        assert 1.5 < tsv.mean_total_goals.mean < 2.5
