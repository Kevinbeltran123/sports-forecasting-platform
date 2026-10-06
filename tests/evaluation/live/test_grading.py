"""Tests for the unified grading module.

Each market we predict needs a grade_pick branch + at least one win/loss
test. These tests are the contract that prevents the "backtest reports
0/0 for unknown markets" regression that motivated the refactor.
"""

from __future__ import annotations

import pytest

from bip.evaluation.live.grading import FinalOutcome, grade_pick


def _outcome(
    *,
    home_goals: int = 1, away_goals: int = 1,
    home_1h: int = 0, away_1h: int = 0,
    total_cards: int = 4, total_corners: int = 9,
    goal_events: tuple[tuple[int, int], ...] = (),
    home_team_id: int = 10, away_team_id: int = 20,
) -> FinalOutcome:
    return FinalOutcome(
        fixture_id=1, league_id=8,
        home_goals=home_goals, away_goals=away_goals,
        home_goals_first_half=home_1h,
        away_goals_first_half=away_1h,
        total_cards=total_cards, total_corners=total_corners,
        is_finished=True, goal_events=goal_events,
        home_team_id=home_team_id, away_team_id=away_team_id,
    )


# ── 1X2 family ─────────────────────────────────────────────────────────────


class TestFulltimeResult:
    def test_home_win(self):
        o = _outcome(home_goals=2, away_goals=1)
        assert grade_pick("fulltime_result", "home", o) is True
        assert grade_pick("fulltime_result", "draw", o) is False
        assert grade_pick("fulltime_result", "away", o) is False

    def test_draw(self):
        o = _outcome(home_goals=1, away_goals=1)
        assert grade_pick("fulltime_result", "draw", o) is True

    def test_away_win(self):
        o = _outcome(home_goals=0, away_goals=2)
        assert grade_pick("fulltime_result", "away", o) is True


class TestDoubleChance:
    def test_1x_covers_home_and_draw(self):
        assert grade_pick("double_chance", "1x", _outcome(home_goals=2, away_goals=0)) is True
        assert grade_pick("double_chance", "1x", _outcome(home_goals=1, away_goals=1)) is True
        assert grade_pick("double_chance", "1x", _outcome(home_goals=0, away_goals=2)) is False

    def test_x2_covers_draw_and_away(self):
        assert grade_pick("double_chance", "x2", _outcome(home_goals=2, away_goals=0)) is False
        assert grade_pick("double_chance", "x2", _outcome(home_goals=1, away_goals=1)) is True
        assert grade_pick("double_chance", "x2", _outcome(home_goals=0, away_goals=2)) is True

    def test_12_covers_no_draw(self):
        assert grade_pick("double_chance", "12", _outcome(home_goals=2, away_goals=0)) is True
        assert grade_pick("double_chance", "12", _outcome(home_goals=1, away_goals=1)) is False
        assert grade_pick("double_chance", "12", _outcome(home_goals=0, away_goals=2)) is True


class TestDrawNoBet:
    def test_home_wins_home_pick_wins(self):
        assert grade_pick("draw_no_bet", "home", _outcome(home_goals=2, away_goals=0)) is True
        assert grade_pick("draw_no_bet", "away", _outcome(home_goals=2, away_goals=0)) is False

    def test_draw_returns_none_void(self):
        # Stake refunded on draw — None signals void
        o = _outcome(home_goals=1, away_goals=1)
        assert grade_pick("draw_no_bet", "home", o) is None
        assert grade_pick("draw_no_bet", "away", o) is None


class TestHtft:
    def test_home_home(self):
        # 1H result: home leads 1-0; FT: home wins 2-0
        o = _outcome(home_goals=2, away_goals=0, home_1h=1, away_1h=0)
        assert grade_pick("htft", "home_home", o) is True
        assert grade_pick("htft", "draw_home", o) is False

    def test_draw_away(self):
        # 1H: 0-0 draw; FT: away wins
        o = _outcome(home_goals=0, away_goals=2, home_1h=0, away_1h=0)
        assert grade_pick("htft", "draw_away", o) is True


# ── BTTS family ────────────────────────────────────────────────────────────


class TestBTTS:
    def test_btts_yes(self):
        o = _outcome(home_goals=1, away_goals=1)
        assert grade_pick("btts", "yes", o) is True
        assert grade_pick("btts", "no", o) is False

    def test_btts_no_when_one_team_scoreless(self):
        o = _outcome(home_goals=2, away_goals=0)
        assert grade_pick("btts", "no", o) is True
        assert grade_pick("btts", "yes", o) is False

    def test_btts_first_half(self):
        # Both teams score in 1H
        o = _outcome(home_goals=2, away_goals=1, home_1h=1, away_1h=1)
        assert grade_pick("btts_first_half", "yes", o) is True

    def test_btts_first_half_no_when_one_didnt_score_in_1h(self):
        # Both score full match, but one didn't score in 1H
        o = _outcome(home_goals=2, away_goals=1, home_1h=2, away_1h=0)
        assert grade_pick("btts_first_half", "yes", o) is False
        assert grade_pick("btts_first_half", "no", o) is True

    def test_btts_second_half(self):
        # Both score in 2H (home_1h=1, home_2h=2-1=1; away_1h=0, away_2h=1-0=1)
        o = _outcome(home_goals=2, away_goals=1, home_1h=1, away_1h=0)
        assert grade_pick("btts_second_half", "yes", o) is True

    def test_btts_second_half_no(self):
        # Both score in 1H but neither in 2H
        o = _outcome(home_goals=1, away_goals=1, home_1h=1, away_1h=1)
        assert grade_pick("btts_second_half", "yes", o) is False
        assert grade_pick("btts_second_half", "no", o) is True


# ── Clean sheets ───────────────────────────────────────────────────────────


class TestCleanSheet:
    def test_home_clean_sheet(self):
        # away_goals = 0 → home keeper kept clean
        o = _outcome(home_goals=2, away_goals=0)
        assert grade_pick("home_clean_sheet", "yes", o) is True
        assert grade_pick("home_clean_sheet", "no", o) is False

    def test_away_clean_sheet(self):
        o = _outcome(home_goals=0, away_goals=2)
        assert grade_pick("away_clean_sheet", "yes", o) is True
        assert grade_pick("away_clean_sheet", "no", o) is False


# ── Team to score first ────────────────────────────────────────────────────


class TestTeamToScoreFirst:
    def test_home_scores_first(self):
        o = _outcome(
            home_goals=1, away_goals=1,
            goal_events=((20, 10), (60, 20)),  # home(10) at 20, away(20) at 60
        )
        assert grade_pick("team_to_score_first", "home", o) is True
        assert grade_pick("team_to_score_first", "away", o) is False
        assert grade_pick("team_to_score_first", "none", o) is False

    def test_away_scores_first(self):
        o = _outcome(
            home_goals=1, away_goals=1,
            goal_events=((25, 20), (70, 10)),
        )
        assert grade_pick("team_to_score_first", "away", o) is True
        assert grade_pick("team_to_score_first", "home", o) is False

    def test_no_goals_match_zero_zero(self):
        o = _outcome(home_goals=0, away_goals=0, goal_events=())
        assert grade_pick("team_to_score_first", "none", o) is True
        assert grade_pick("team_to_score_first", "home", o) is False


# ── OU goals ───────────────────────────────────────────────────────────────


class TestOuGoals:
    @pytest.mark.parametrize("market,line,total,over_wins", [
        ("ou_0_5", 0.5, 0, False),
        ("ou_0_5", 0.5, 1, True),
        ("ou_1_5", 1.5, 1, False),
        ("ou_1_5", 1.5, 2, True),
        ("ou_2_5", 2.5, 2, False),
        ("ou_2_5", 2.5, 3, True),
        ("ou_3_5", 3.5, 3, False),
        ("ou_3_5", 3.5, 4, True),
    ])
    def test_total_goals(self, market, line, total, over_wins):
        # Distribute total between home + away
        o = _outcome(home_goals=total, away_goals=0)
        assert grade_pick(market, "over", o) is over_wins
        assert grade_pick(market, "under", o) is not over_wins

    def test_first_half_ou(self):
        # 1.5 in 1H. Match has 2 1H goals → over wins.
        o = _outcome(home_goals=3, away_goals=1, home_1h=2, away_1h=0)
        assert grade_pick("first_half_ou_1_5", "over", o) is True
        # 0.5 in 1H. Same match, 2 1H goals → over wins.
        assert grade_pick("first_half_ou_0_5", "over", o) is True
        # 0 1H goals → 0.5 over loses.
        o2 = _outcome(home_goals=2, away_goals=0, home_1h=0, away_1h=0)
        assert grade_pick("first_half_ou_0_5", "over", o2) is False


class TestPerTeamOu:
    def test_home_ou_1_5_over(self):
        assert grade_pick("home_ou_1_5", "over", _outcome(home_goals=2, away_goals=0)) is True
        assert grade_pick("home_ou_1_5", "over", _outcome(home_goals=1, away_goals=0)) is False
        assert grade_pick("home_ou_1_5", "under", _outcome(home_goals=1, away_goals=0)) is True

    def test_away_ou_1_5(self):
        assert grade_pick("away_ou_1_5", "over", _outcome(home_goals=0, away_goals=3)) is True
        assert grade_pick("away_ou_1_5", "under", _outcome(home_goals=0, away_goals=1)) is True


# ── Corners + Cards ────────────────────────────────────────────────────────


class TestCornersTotal:
    @pytest.mark.parametrize("market,line", [
        ("corners_total_8_5", 8.5),
        ("corners_total_9_5", 9.5),
        ("corners_total_10_5", 10.5),
        ("corners_total_11_5", 11.5),
    ])
    def test_corners_grading(self, market, line):
        # Total corners exactly at line+1 → over wins
        o_high = _outcome(total_corners=int(line) + 1)
        assert grade_pick(market, "over", o_high) is True
        # Below line → under wins
        o_low = _outcome(total_corners=int(line))
        assert grade_pick(market, "under", o_low) is True


class TestCardsTotal:
    @pytest.mark.parametrize("market,line", [
        ("cards_total_3_5", 3.5),
        ("cards_total_4_5", 4.5),
        ("cards_total_5_5", 5.5),
    ])
    def test_cards_grading(self, market, line):
        o_high = _outcome(total_cards=int(line) + 1)
        assert grade_pick(market, "over", o_high) is True
        o_low = _outcome(total_cards=int(line))
        assert grade_pick(market, "under", o_low) is True


# ── Unknown market ─────────────────────────────────────────────────────────


class TestUnknownMarket:
    def test_unknown_returns_none(self):
        o = _outcome()
        assert grade_pick("alien_market", "yes", o) is None
        assert grade_pick("ftr_typo", "home", o) is None
