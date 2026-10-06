"""Tests for team_form module — cache + form computation + predictor lift.

Covers:
- ``compute_team_form``: 1H/2H rate derivation from synthetic goal_events
- ``TeamFormCache.get/put``: round-trip + TTL expiry
- ``TeamFormCache.get_or_fetch``: cache-hit short-circuit + miss-fetch path
- Predictor BTTS-1H / BTTS-2H lift when team form is provided

Synthetic Sportmonks data — no live API required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.live.predictor import (
    BTTS_FIRST_HALF_FRACTION,
    BTTS_SECOND_HALF_FRACTION,
    MARKET_BTTS_FIRST_HALF,
    MARKET_BTTS_SECOND_HALF,
    LiveMatchPredictor,
)
from bip.evaluation.live.team_form import (
    LEAGUE_FIRST_HALF_GOAL_RATE_PRIOR,
    MIN_FIXTURES_FOR_FORM,
    TeamForm,
    TeamFormCache,
    compute_team_form,
)
from bip.sports.football.sportmonks.schemas import Event, Fixture, FixtureState, Participant


# ── Synthetic fixture builders ─────────────────────────────────────────────


def _participant(team_id: int, name: str = "T") -> Participant:
    return Participant.model_validate({"id": team_id, "name": f"{name}{team_id}"})


def _event(minute: int, team_id: int, type_id: int = 14) -> Event:
    """type_id 14 = GOAL by default."""
    return Event.model_validate({
        "id": id((minute, team_id)),
        "fixture_id": 1,
        "type_id": type_id,
        "minute": minute,
        "participant_id": team_id,
        "period_id": 1 if minute <= 45 else 2,
    })


def _fixture(
    *,
    fixture_id: int,
    home_id: int,
    away_id: int,
    starting_at: datetime,
    state_dev: str = "FT",
    goal_minutes_home: list[int] | None = None,
    goal_minutes_away: list[int] | None = None,
) -> Fixture:
    events = []
    for m in goal_minutes_home or []:
        events.append(_event(m, home_id))
    for m in goal_minutes_away or []:
        events.append(_event(m, away_id))
    return Fixture.model_validate({
        "id": fixture_id,
        "sport_id": 1,
        "league_id": 100,
        "season_id": 22,
        "starting_at": starting_at.isoformat(),
        "state": {
            "id": 5, "state": state_dev, "name": "Finished",
            "short_name": "FT", "developer_name": state_dev,
        },
        "participants": [
            _participant(home_id).model_dump(),
            _participant(away_id).model_dump(),
        ],
        "events": [e.model_dump() for e in events],
    })


# ── compute_team_form ───────────────────────────────────────────────────────


class TestComputeTeamForm:
    def test_returns_none_below_min_fixtures(self):
        # Only 2 completed fixtures — below MIN_FIXTURES_FOR_FORM (3)
        ref = datetime(2026, 5, 9, tzinfo=timezone.utc)
        fixtures = [
            _fixture(
                fixture_id=i, home_id=10, away_id=20,
                starting_at=ref - timedelta(days=i),
                goal_minutes_home=[30], goal_minutes_away=[60],
            )
            for i in range(2)
        ]
        assert compute_team_form(10, season_id=22, fixtures=fixtures) is None

    def test_first_half_goal_rate_pure(self):
        """Team scores in 1H in 4 of 5 matches → rate = 0.80."""
        ref = datetime(2026, 5, 9, tzinfo=timezone.utc)
        fixtures = [
            # Match 1: home scores at 20'
            _fixture(fixture_id=1, home_id=10, away_id=20,
                     starting_at=ref - timedelta(days=1),
                     goal_minutes_home=[20], goal_minutes_away=[60]),
            # Match 2: home scores at 35'
            _fixture(fixture_id=2, home_id=10, away_id=21,
                     starting_at=ref - timedelta(days=8),
                     goal_minutes_home=[35], goal_minutes_away=[]),
            # Match 3: home scores at 10' AND 70'
            _fixture(fixture_id=3, home_id=10, away_id=22,
                     starting_at=ref - timedelta(days=15),
                     goal_minutes_home=[10, 70], goal_minutes_away=[]),
            # Match 4: home doesn't score 1H (only 80')
            _fixture(fixture_id=4, home_id=10, away_id=23,
                     starting_at=ref - timedelta(days=22),
                     goal_minutes_home=[80], goal_minutes_away=[]),
            # Match 5: home scores at 44'
            _fixture(fixture_id=5, home_id=10, away_id=24,
                     starting_at=ref - timedelta(days=29),
                     goal_minutes_home=[44], goal_minutes_away=[10]),
        ]
        form = compute_team_form(10, season_id=22, fixtures=fixtures)
        assert form is not None
        assert form.n_matches == 5
        # Matches with ≥1 1H goal: 1, 2, 3, 5 → 4/5 = 0.80
        assert form.first_half_goal_rate == pytest.approx(0.80)

    def test_late_goals_rate(self):
        """Late = ≥75 min. Compute rate of matches with at least one late goal."""
        ref = datetime(2026, 5, 9, tzinfo=timezone.utc)
        fixtures = [
            _fixture(fixture_id=i, home_id=10, away_id=20 + i,
                     starting_at=ref - timedelta(days=i * 2),
                     goal_minutes_home=goal_mins,
                     goal_minutes_away=[])
            for i, goal_mins in enumerate([
                [80],       # late ✓
                [85, 30],   # late ✓
                [40],       # not late
                [10, 78],   # late ✓
                [20],       # not late
            ])
        ]
        form = compute_team_form(10, season_id=22, fixtures=fixtures)
        assert form is not None
        # 3 of 5 have a late goal
        assert form.late_goals_rate == pytest.approx(0.60)

    def test_excludes_unfinished_matches(self):
        """State != FT/AET/FT_PEN/FINISHED → excluded from aggregation."""
        ref = datetime(2026, 5, 9, tzinfo=timezone.utc)
        fixtures = [
            _fixture(fixture_id=1, home_id=10, away_id=20,
                     starting_at=ref - timedelta(days=1),
                     state_dev="INPLAY_2ND_HALF",  # in-play, exclude
                     goal_minutes_home=[20], goal_minutes_away=[]),
            _fixture(fixture_id=2, home_id=10, away_id=21,
                     starting_at=ref - timedelta(days=8),
                     state_dev="FT",
                     goal_minutes_home=[30], goal_minutes_away=[]),
            _fixture(fixture_id=3, home_id=10, away_id=22,
                     starting_at=ref - timedelta(days=15),
                     state_dev="FT",
                     goal_minutes_home=[35], goal_minutes_away=[]),
            _fixture(fixture_id=4, home_id=10, away_id=23,
                     starting_at=ref - timedelta(days=22),
                     state_dev="FT",
                     goal_minutes_home=[80], goal_minutes_away=[]),
        ]
        form = compute_team_form(10, season_id=22, fixtures=fixtures)
        assert form is not None
        # Only 3 finished matches counted (the in-play one excluded)
        assert form.n_matches == 3

    def test_clean_sheet_and_btts_rates(self):
        ref = datetime(2026, 5, 9, tzinfo=timezone.utc)
        fixtures = [
            # Clean sheet, both score? No — opp doesn't score
            _fixture(fixture_id=1, home_id=10, away_id=20,
                     starting_at=ref - timedelta(days=1),
                     goal_minutes_home=[30], goal_minutes_away=[]),
            # BTTS: home scores 30, away scores 60
            _fixture(fixture_id=2, home_id=10, away_id=21,
                     starting_at=ref - timedelta(days=8),
                     goal_minutes_home=[30], goal_minutes_away=[60]),
            # Clean sheet for team 10
            _fixture(fixture_id=3, home_id=10, away_id=22,
                     starting_at=ref - timedelta(days=15),
                     goal_minutes_home=[20, 70], goal_minutes_away=[]),
        ]
        form = compute_team_form(10, season_id=22, fixtures=fixtures)
        assert form is not None
        # Clean sheets: 1 and 3 → 2/3
        assert form.clean_sheet_rate == pytest.approx(2 / 3)
        # BTTS: only match 2 → 1/3
        assert form.btts_rate == pytest.approx(1 / 3)


# ── TeamFormCache ───────────────────────────────────────────────────────────


class TestTeamFormCache:
    def test_put_and_get_round_trip(self, tmp_path):
        db = tmp_path / "form.db"
        cache = TeamFormCache(db_path=db)
        form = TeamForm(
            team_id=10, season_id=22, n_matches=8,
            last_updated=datetime.now(timezone.utc),
            first_half_goal_rate=0.65, second_half_goal_rate=0.70,
            early_goals_rate=0.20, late_goals_rate=0.40,
            btts_rate=0.55, clean_sheet_rate=0.25,
            failed_to_score_rate=0.15,
            goals_scored_avg=1.6, goals_conceded_avg=1.2,
            wins_last_5=2, draws_last_5=2, losses_last_5=1,
        )
        cache.put(form)
        loaded = cache.get(10, 22)
        assert loaded is not None
        assert loaded.first_half_goal_rate == pytest.approx(0.65)
        assert loaded.wins_last_5 == 2

    def test_ttl_expiry(self, tmp_path):
        db = tmp_path / "form.db"
        cache = TeamFormCache(db_path=db, ttl_hours=1)
        old_form = TeamForm(
            team_id=10, season_id=22, n_matches=8,
            last_updated=datetime.now(timezone.utc) - timedelta(hours=2),
            first_half_goal_rate=0.65, second_half_goal_rate=0.70,
            early_goals_rate=0.20, late_goals_rate=0.40,
            btts_rate=0.55, clean_sheet_rate=0.25,
            failed_to_score_rate=0.15,
            goals_scored_avg=1.6, goals_conceded_avg=1.2,
            wins_last_5=2, draws_last_5=2, losses_last_5=1,
        )
        cache.put(old_form)
        # Older than ttl=1h → expired
        assert cache.get(10, 22) is None

    def test_get_or_fetch_uses_cache_when_fresh(self, tmp_path):
        db = tmp_path / "form.db"
        cache = TeamFormCache(db_path=db)
        fresh = TeamForm(
            team_id=10, season_id=22, n_matches=8,
            last_updated=datetime.now(timezone.utc),
            first_half_goal_rate=0.65, second_half_goal_rate=0.70,
            early_goals_rate=0.20, late_goals_rate=0.40,
            btts_rate=0.55, clean_sheet_rate=0.25,
            failed_to_score_rate=0.15,
            goals_scored_avg=1.6, goals_conceded_avg=1.2,
            wins_last_5=2, draws_last_5=2, losses_last_5=1,
        )
        cache.put(fresh)

        class _MockClient:
            calls = 0
            async def get_team_recent_fixtures(self, *_a, **_kw):
                _MockClient.calls += 1
                return []

        client = _MockClient()
        loaded = asyncio.run(cache.get_or_fetch(client, 10, 22))
        assert loaded is not None
        assert client.calls == 0  # cache hit — no API call

    def test_get_or_fetch_miss_calls_client(self, tmp_path):
        db = tmp_path / "form.db"
        cache = TeamFormCache(db_path=db)
        ref = datetime(2026, 5, 9, tzinfo=timezone.utc)
        api_fixtures = [
            _fixture(fixture_id=i, home_id=10, away_id=20 + i,
                     starting_at=ref - timedelta(days=i * 2),
                     goal_minutes_home=[20, 60], goal_minutes_away=[])
            for i in range(5)
        ]

        class _MockClient:
            async def get_team_recent_fixtures(self, *_a, **_kw):
                return api_fixtures

        loaded = asyncio.run(cache.get_or_fetch(_MockClient(), 10, 22))
        assert loaded is not None
        assert loaded.n_matches == 5
        # Was persisted to cache too
        again = cache.get(10, 22)
        assert again is not None

    def test_get_or_fetch_handles_client_error(self, tmp_path):
        db = tmp_path / "form.db"
        cache = TeamFormCache(db_path=db)

        class _BrokenClient:
            async def get_team_recent_fixtures(self, *_a, **_kw):
                raise RuntimeError("Sportmonks down")

        # Must NOT raise — degrade to None gracefully
        result = asyncio.run(cache.get_or_fetch(_BrokenClient(), 10, 22))
        assert result is None


# ── Predictor lift integration ──────────────────────────────────────────────


class TestPredictorTeamFormLift:
    """Verify that team-form signals materially shift BTTS-1H probability
    when both teams have a strong recent pattern that diverges from the
    league baseline."""

    def _make_form(self, *, fh_rate: float = 0.60, sh_rate: float = 0.65) -> TeamForm:
        return TeamForm(
            team_id=10, season_id=22, n_matches=10,
            last_updated=datetime(2026, 5, 9, tzinfo=timezone.utc),
            first_half_goal_rate=fh_rate, second_half_goal_rate=sh_rate,
            early_goals_rate=0.20, late_goals_rate=0.40,
            btts_rate=0.55, clean_sheet_rate=0.25,
            failed_to_score_rate=0.15,
            goals_scored_avg=1.6, goals_conceded_avg=1.2,
            wins_last_5=2, draws_last_5=2, losses_last_5=1,
        )

    def test_btts_1h_lifted_when_both_teams_score_1h_above_baseline(self):
        """Both teams score in 1H 60% of the time vs league 42% → joint
        ratio 0.36/0.18 = 2.0 → capped to 1.4× lift on BTTS-1H prob.
        """
        from bip.evaluation.live.match_state import LiveMatchState
        # Use balanced default Sportmonks state
        from tests.evaluation.live.test_audit_hardening import (
            _BALANCED_PRE_MATCH,
            _make_state,
        )
        baseline_state = _make_state(minute=20, home_goals=0)
        # Same state, but with high-1H form on both sides
        lifted_state = LiveMatchState(
            **{**baseline_state.__dict__,
               "home_team_form": self._make_form(fh_rate=0.60),
               "away_team_form": self._make_form(fh_rate=0.60)}
        )
        predictor = LiveMatchPredictor()
        baseline_btts1h = predictor.predict(baseline_state).by_market[
            MARKET_BTTS_FIRST_HALF
        ]
        lifted_btts1h = predictor.predict(lifted_state).by_market[
            MARKET_BTTS_FIRST_HALF
        ]
        # Lifted should have noticeably higher P(yes)
        assert lifted_btts1h["yes"] > baseline_btts1h["yes"] + 0.01

    def test_btts_1h_dampened_when_both_teams_score_1h_below_baseline(self):
        """Defensive teams (1H rate 0.25) → joint ratio 0.0625/0.18 = 0.35
        → capped to 0.7× damp on BTTS-1H prob.
        """
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        baseline_state = _make_state(minute=20, home_goals=0)
        damped_state = LiveMatchState(
            **{**baseline_state.__dict__,
               "home_team_form": self._make_form(fh_rate=0.25),
               "away_team_form": self._make_form(fh_rate=0.25)}
        )
        predictor = LiveMatchPredictor()
        baseline = predictor.predict(baseline_state).by_market[
            MARKET_BTTS_FIRST_HALF
        ]
        damped = predictor.predict(damped_state).by_market[
            MARKET_BTTS_FIRST_HALF
        ]
        assert damped["yes"] < baseline["yes"] - 0.01

    def test_no_lift_when_form_missing(self):
        """When either side's form is None, predictor must use unmodified
        Sportmonks baseline — graceful degradation."""
        from tests.evaluation.live.test_audit_hardening import _make_state
        state = _make_state(minute=20, home_goals=0)
        # State with home_team_form=None, away_team_form=None (default)
        predictor = LiveMatchPredictor()
        btts1h = predictor.predict(state).by_market[MARKET_BTTS_FIRST_HALF]
        # Baseline = sportmonks_yes × BTTS_FIRST_HALF_FRACTION = 0.6 × 0.45 = 0.27
        assert btts1h["yes"] == pytest.approx(
            0.60 * BTTS_FIRST_HALF_FRACTION, abs=0.01
        )
