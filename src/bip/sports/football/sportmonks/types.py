"""Sportmonks type_id constants.

Type IDs are stable across the API and represent statistic types,
prediction markets, in-play odds markets, etc. We hard-code the ones
we use into named constants rather than re-querying ``/core/types``
on every call. The full type catalog lives at
``data/cache/sportmonks/recon/30_all_types.json``.

The ``StatType`` enum applies to BOTH ``statistics`` (aggregate per
fixture × team) and ``trends`` (minute-by-minute per fixture × team).
The same type_id is reused.

Verified against /core/types pull on 2026-05-09. Sportmonks rarely
changes these IDs — they are part of their public contract.
"""

from __future__ import annotations

from enum import IntEnum


class StatType(IntEnum):
    """Statistics + Trends type_ids (verified 2026-05-09)."""

    CORNERS = 34
    SHOTS_OFF_TARGET = 41
    SHOTS_TOTAL = 42
    ATTACKS = 43
    DANGEROUS_ATTACKS = 44
    BALL_POSSESSION = 45
    BALL_SAFE = 46
    SHOTS_INSIDEBOX = 49
    SHOTS_OUTSIDEBOX = 50
    OFFSIDES = 51
    GOALS = 52
    GOAL_KICKS = 53
    GOAL_ATTEMPTS = 54
    FREE_KICKS = 55
    FOULS = 56
    SAVES = 57
    SHOTS_BLOCKED = 58
    SUBSTITUTIONS = 59
    THROWINS = 60
    LONG_PASSES = 62
    TACKLES = 78
    ASSISTS = 79
    PASSES = 80
    SUCCESSFUL_PASSES = 81
    SUCCESSFUL_PASSES_PERCENTAGE = 82
    YELLOW_CARDS = 84
    SHOTS_ON_TARGET = 86
    INJURIES = 87
    TOTAL_CROSSES = 98
    ACCURATE_CROSSES = 99
    INTERCEPTIONS = 100
    DUELS_WON = 106
    DRIBBLE_ATTEMPTS = 108
    SUCCESSFUL_DRIBBLES = 109
    KEY_PASSES = 117
    BIG_CHANCES_CREATED = 580
    BIG_CHANCES_MISSED = 581
    SUCCESSFUL_CROSSES_PERCENTAGE = 1533

    # IDs observed in trends but not yet mapped to documented names.
    # 1605, 27264, 27265 — placeholder; we ignore unknown ids on parse.


# Stats most useful as live signals — used to compute live xG proxy and
# momentum signal. Other stats are kept on-disk but not consumed by the
# predictor by default.
LIVE_PREDICTOR_STAT_IDS: frozenset[int] = frozenset({
    StatType.SHOTS_TOTAL,
    StatType.SHOTS_ON_TARGET,
    StatType.SHOTS_INSIDEBOX,
    StatType.SHOTS_OUTSIDEBOX,
    StatType.DANGEROUS_ATTACKS,
    StatType.ATTACKS,
    StatType.BIG_CHANCES_CREATED,
    StatType.BIG_CHANCES_MISSED,
    StatType.CORNERS,
    StatType.BALL_POSSESSION,
    StatType.GOALS,
    StatType.YELLOW_CARDS,
    StatType.SAVES,
    StatType.KEY_PASSES,
})


class PredictionType(IntEnum):
    """Sportmonks pre-built prediction type_ids (verified 2026-05-09)."""

    VALUEBET = 33
    BTTS_PROBABILITY = 231
    HTFT_PROBABILITY = 232
    FIRST_HALF_WINNER_PROBABILITY = 233
    OVER_UNDER_1_5_PROBABILITY = 234
    OVER_UNDER_2_5_PROBABILITY = 235
    OVER_UNDER_3_5_PROBABILITY = 236
    FULLTIME_RESULT_PROBABILITY = 237
    TEAM_TO_SCORE_FIRST_PROBABILITY = 238
    DOUBLE_CHANCE_PROBABILITY = 239
    CORRECT_SCORE_PROBABILITY = 240
    HOME_OVER_UNDER_3_5_PROBABILITY = 326
    AWAY_OVER_UNDER_3_5_PROBABILITY = 327
    AWAY_OVER_UNDER_2_5_PROBABILITY = 328
    HOME_OVER_UNDER_2_5_PROBABILITY = 330
    HOME_OVER_UNDER_1_5_PROBABILITY = 331
    AWAY_OVER_UNDER_1_5_PROBABILITY = 332
    AWAY_OVER_UNDER_0_5_PROBABILITY = 333
    HOME_OVER_UNDER_0_5_PROBABILITY = 334


class MarketID(IntEnum):
    """In-play odds market_ids (Sportmonks bookmaker market codes).

    Verified against bookmaker_id=2 on 2026-05-09. ``market_id`` is a
    Sportmonks-internal mapping; the corresponding ``market_description``
    string is canonical.
    """

    FULLTIME_RESULT = 1
    DOUBLE_CHANCE = 2
    FIRST_GOAL = 3
    MATCH_GOALS = 4
    ALTERNATIVE_MATCH_GOALS = 5
    ASIAN_HANDICAP_1_2 = 6
    GOAL_LINE_0_0 = 7
    FINAL_SCORE = 8
    THREE_WAY_HANDICAP = 9
    DRAW_NO_BET = 10
    LAST_TEAM_TO_SCORE = 11
    GOALS_ODD_EVEN = 12
    RESULT_BTTS = 13
    BOTH_TEAMS_TO_SCORE = 14
    BTTS_FIRST_HALF = 15
    BTTS_SECOND_HALF = 16
    TEAM_CLEAN_SHEET = 17
    HOME_EXACT_GOALS = 18  # team-specific; varies per fixture
    AWAY_EXACT_GOALS = 19
    HOME_GOALS = 20
    AWAY_GOALS = 21
    TO_WIN_2ND_HALF = 23
    TEAM_TO_SCORE_2ND_HALF = 25
    FIRST_HALF_ASIAN_HANDICAP = 26
    FIRST_HALF_GOAL_LINE = 27
    FIRST_HALF_GOALS = 28
    HALFTIME_FULLTIME = 29
    HALF_TIME_CORRECT_SCORE = 30
    HALF_TIME_RESULT = 31
    FIRST_HALF_HANDICAP = 32
    TWO_WAY_CORNERS = 60
    ASIAN_CORNERS = 61
    FIRST_HALF_ASIAN_CORNERS = 63
    MATCH_CORNERS = 68
    FIRST_HALF_CORNERS = 70
    CORNERS_RACE = 75
    GOALSCORERS = 90
    MULTI_SCORERS = 100
    TEAM_TO_SCORE_BOTH_HALVES = 248
    NUMBER_OF_CARDS = 255
    TO_QUALIFY = 256
    SECOND_HALF_CORNERS = 265


# Markets we actively chase — cherry-picked because Sportmonks emits a
# matching prediction for them, AND they appear in in-play odds, AND
# our value detector knows how to map them.
ACTIONABLE_MARKETS: frozenset[int] = frozenset({
    MarketID.FULLTIME_RESULT,
    MarketID.DOUBLE_CHANCE,
    MarketID.HALF_TIME_RESULT,
    MarketID.HALFTIME_FULLTIME,
    MarketID.MATCH_GOALS,
    MarketID.FIRST_HALF_GOALS,
    MarketID.BOTH_TEAMS_TO_SCORE,
    MarketID.BTTS_FIRST_HALF,
})


# State IDs we treat as "live" / "in-play" for the inplay endpoint.
# State 22 = HT break, 2 = 1H, 3 = 2H, 5 = FT (just finished, 15min window).
LIVE_STATE_IDS: frozenset[int] = frozenset({2, 3, 7, 8, 22})


# Period type_ids (from /core/types verification)
PERIOD_FIRST_HALF = 1
PERIOD_SECOND_HALF = 2
PERIOD_ET_1ST = 38
PERIOD_ET_2ND = 39
