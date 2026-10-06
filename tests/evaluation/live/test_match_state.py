"""Layer-1 tests for LiveMatchState.from_fixture()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bip.evaluation.live.match_state import LiveMatchState
from bip.sports.football.sportmonks.schemas import Fixture
from bip.sports.football.sportmonks.types import StatType

RECON_DIR = Path(__file__).resolve().parent.parent.parent.parent \
    / "data" / "cache" / "sportmonks" / "recon"


pytestmark = pytest.mark.skipif(
    not (RECON_DIR / "21_live_high_value.json").exists(),
    reason="Sportmonks recon fixture missing — run reconnaissance first.",
)


@pytest.fixture
def fixture() -> Fixture:
    raw = json.loads((RECON_DIR / "21_live_high_value.json").read_text())
    return Fixture.model_validate(raw["data"])


class TestStateConstruction:
    def test_basic_fields(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        assert state.fixture_id == fixture.id
        assert state.home_team_id != state.away_team_id
        assert state.home_team_name and state.away_team_name

    def test_score_extraction(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        assert state.home_goals >= 0
        assert state.away_goals >= 0

    def test_minute_in_valid_range(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        assert 0 <= state.minute <= 130  # incl. extra time

    def test_stats_indexed_by_type_id(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        # The recon fixture has 80 stats — both teams should have most types
        assert len(state.home_stats) > 10
        assert len(state.away_stats) > 10
        # Possession should be present
        assert StatType.BALL_POSSESSION in state.home_stats
        assert StatType.BALL_POSSESSION in state.away_stats

    def test_predictions_indexed(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        # The recon fixture has 29 predictions; verify indexing works
        assert len(state.sportmonks_predictions) > 5


class TestDerivedSignals:
    def test_xg_proxy_non_negative(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        assert state.home_shot_quality_xg_proxy >= 0
        assert state.away_shot_quality_xg_proxy >= 0

    def test_pressure_avg_in_range(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        # Pressure 0-100 → avg should also be in that range
        assert 0 <= state.home_pressure_avg <= 100
        assert 0 <= state.away_pressure_avg <= 100

    def test_remaining_minutes_consistent(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        if state.minute >= 90:
            assert state.remaining_minutes == 0
        else:
            assert state.remaining_minutes == 90 - state.minute

    def test_score_diff(self, fixture: Fixture):
        state = LiveMatchState.from_fixture(fixture)
        assert state.score_diff_home == state.home_goals - state.away_goals


class TestScoreExtractionOrderInvariance:
    """Regression for bug 2026-05-11: Sportmonks scores array can be
    non-chronological. Prior _extract_current_score walked all entries
    and overwrote based on iteration order, returning 1ST_HALF score
    even at FT. 66% of Day-1+2 fixtures affected (Randers vs Odense
    stored 0-1 vs real 2-2). Fix filters by type_id, preferring
    CURRENT (1525)."""

    @staticmethod
    def _make_fixture(scores_payload: list[dict]) -> Fixture:
        return Fixture.model_validate({
            "id": 999999, "sport_id": 1, "league_id": 271, "season_id": 1,
            "state": {"id": 5, "state": "FT", "name": "Full Time",
                      "short_name": "FT", "developer_name": "FT"},
            "participants": [
                {"id": 100, "name": "Home FC", "meta": {"location": "home"}},
                {"id": 200, "name": "Away SC", "meta": {"location": "away"}},
            ],
            "scores": scores_payload,
        })

    @staticmethod
    def _s(type_id: int, goals: int, side: str) -> dict:
        return {"id": 1, "fixture_id": 999999, "type_id": type_id,
                "score": {"goals": goals, "participant": side},
                "participant_id": 100 if side == "home" else 200}

    def test_randers_odense_non_chronological_at_ft(self):
        """Real-world Day-2 fixture: 1ST_HALF appears AFTER CURRENT
        in the scores array. Pre-fix returned 0-1 (half-time score).
        Post-fix must return 2-2 (final score, CURRENT type_id=1525)."""
        s = self._s
        fixture = self._make_fixture([
            s(2, 2, "home"),       # 2ND_HALF home
            s(48996, 2, "home"),   # 2ND_HALF_ONLY home
            s(1525, 2, "home"),    # CURRENT home
            s(1, 1, "away"),       # 1ST_HALF away
            s(2, 2, "away"),       # 2ND_HALF away
            s(1, 0, "home"),       # 1ST_HALF home  ← overwrote in old code
            s(1525, 2, "away"),    # CURRENT away
            s(48996, 1, "away"),   # 2ND_HALF_ONLY away ← overwrote in old code
        ])
        state = LiveMatchState.from_fixture(fixture)
        assert (state.home_goals, state.away_goals) == (2, 2)

    def test_current_wins_when_1h_appears_later(self):
        s = self._s
        fixture = self._make_fixture([
            s(1525, 4, "home"), s(1525, 2, "away"),
            s(1, 1, "home"), s(1, 0, "away"),
        ])
        state = LiveMatchState.from_fixture(fixture)
        assert (state.home_goals, state.away_goals) == (4, 2)

    def test_fallback_to_2nd_half_when_no_current(self):
        s = self._s
        fixture = self._make_fixture([
            s(1, 1, "home"), s(1, 0, "away"),
            s(2, 3, "home"), s(2, 2, "away"),
        ])
        state = LiveMatchState.from_fixture(fixture)
        assert (state.home_goals, state.away_goals) == (3, 2)

    def test_fallback_to_1st_half_at_half_time(self):
        s = self._s
        fixture = self._make_fixture([
            s(1, 1, "home"), s(1, 2, "away"),
        ])
        state = LiveMatchState.from_fixture(fixture)
        assert (state.home_goals, state.away_goals) == (1, 2)

    def test_empty_scores_returns_zero(self):
        fixture = self._make_fixture([])
        state = LiveMatchState.from_fixture(fixture)
        assert (state.home_goals, state.away_goals) == (0, 0)


class TestHomeAwayIdentification:
    """Regression tests for the home/away mis-identification bug.

    Day-4 (2026-05-13) discovery: ``_identify_home_away`` fell back to
    ``participants[0]=home, participants[1]=away`` ignoring Sportmonks'
    ``meta.location`` field. In 9 of 31 Day-4 fixtures the API returned
    participants away-first — most visibly Crystal Palace vs Man City,
    where the score "Palace 3-0 City" was recorded for what was really
    a 3-0 City win.
    """

    @staticmethod
    def _make_fixture(participants: list[dict], scores: list[dict] | None = None) -> Fixture:
        return Fixture.model_validate({
            "id": 12345, "sport_id": 1, "league_id": 8, "season_id": 1,
            "state": {"id": 5, "state": "FT", "name": "Full Time",
                      "short_name": "FT", "developer_name": "FT"},
            "participants": participants,
            "scores": scores or [],
        })

    def test_participants_away_first_uses_meta(self):
        """The Palace-City regression. ``participants[0]`` is the away
        team per ``meta.location`` — we must NOT trust array order."""
        # Order: City first in the array, but Sportmonks marks Palace as home
        # (analogous to the actual Palace-City case where order was inverted).
        fx = self._make_fixture(
            participants=[
                {"id": 9, "name": "Manchester City", "meta": {"location": "away"}},
                {"id": 51, "name": "Crystal Palace", "meta": {"location": "home"}},
            ],
            scores=[
                # Real home (Palace) scored 0; real away (City) scored 3
                {"id": 1, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 0, "participant": "home"},
                 "participant_id": 51},
                {"id": 2, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 3, "participant": "away"},
                 "participant_id": 9},
            ],
        )
        state = LiveMatchState.from_fixture(fx)
        assert state.home_team_id == 51  # Palace
        assert state.home_team_name == "Crystal Palace"
        assert state.away_team_id == 9  # City
        assert state.away_team_name == "Manchester City"
        # Score was tagged per Sportmonks home/away and binds to real
        # home/away — Palace (home) scored 0, City (away) scored 3.
        assert state.home_goals == 0
        assert state.away_goals == 3

    def test_participants_home_first_works_too(self):
        """Sanity: the common case where ``participants[0]`` IS home."""
        fx = self._make_fixture(
            participants=[
                {"id": 100, "name": "Home FC", "meta": {"location": "home"}},
                {"id": 200, "name": "Away SC", "meta": {"location": "away"}},
            ],
            scores=[
                {"id": 1, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 2, "participant": "home"},
                 "participant_id": 100},
                {"id": 2, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 1, "participant": "away"},
                 "participant_id": 200},
            ],
        )
        state = LiveMatchState.from_fixture(fx)
        assert state.home_team_id == 100
        assert state.away_team_id == 200
        assert (state.home_goals, state.away_goals) == (2, 1)

    def test_scores_layer_resolves_palace_city_real_payload(self):
        """The actual Day-4 Palace-City raw payload: no ``meta`` on
        participants, but scores cross-reference is unambiguous.

        Raw evidence (data/cache/sportmonks/snapshots/19427191/*.json):
            participant_id=9  score={participant: 'home'} → City is home
            participant_id=51 score={participant: 'away'} → Palace is away
        """
        fx = self._make_fixture(
            participants=[
                # API returned Palace first, but it's actually AWAY
                {"id": 51, "name": "Crystal Palace"},
                {"id": 9, "name": "Manchester City"},
            ],
            scores=[
                {"id": 1, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 3, "participant": "home"},
                 "participant_id": 9},
                {"id": 2, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 0, "participant": "away"},
                 "participant_id": 51},
            ],
        )
        state = LiveMatchState.from_fixture(fx)
        assert state.home_team_id == 9       # City
        assert state.home_team_name == "Manchester City"
        assert state.away_team_id == 51      # Palace
        assert state.away_team_name == "Crystal Palace"
        assert state.home_goals == 3
        assert state.away_goals == 0

    def test_statistics_layer_resolves_when_no_scores(self):
        """Pre-kickoff fixture: no scores yet but statistics emitted
        with ``location`` field."""
        fx_payload = {
            "id": 99, "sport_id": 1, "league_id": 8, "season_id": 1,
            "state": {"id": 1, "state": "NS", "name": "Not Started",
                      "short_name": "NS", "developer_name": "NS"},
            "participants": [
                {"id": 200, "name": "AwayTeam"},
                {"id": 100, "name": "HomeTeam"},
            ],
            "scores": [],
            "statistics": [
                {"id": 1, "fixture_id": 99, "type_id": 42,
                 "participant_id": 100, "data": {"value": 5},
                 "location": "home"},
                {"id": 2, "fixture_id": 99, "type_id": 42,
                 "participant_id": 200, "data": {"value": 3},
                 "location": "away"},
            ],
        }
        fx = Fixture.model_validate(fx_payload)
        state = LiveMatchState.from_fixture(fx)
        assert state.home_team_id == 100
        assert state.away_team_id == 200

    def test_array_fallback_only_when_no_layered_signal_available(self):
        """No meta, no scores, no statistics → last resort = array order.
        This is the pre-fix behaviour, kept only as a safety net for
        very early NS frames where literally no signal is yet present."""
        fx = self._make_fixture(
            participants=[
                {"id": 100, "name": "First (defaults to home)"},
                {"id": 200, "name": "Second (defaults to away)"},
            ],
            scores=[],
        )
        state = LiveMatchState.from_fixture(fx)
        assert state.home_team_id == 100
        assert state.away_team_id == 200

    def test_partial_meta_raises_rather_than_guess(self):
        """If exactly one participant has meta.location, refuse to
        silently mis-tag. Better to fail loudly than re-introduce the
        original bug class."""
        fx = self._make_fixture(
            participants=[
                {"id": 100, "name": "WithMeta", "meta": {"location": "home"}},
                {"id": 200, "name": "WithoutMeta"},
            ],
        )
        with pytest.raises(ValueError, match="incomplete participant meta"):
            LiveMatchState.from_fixture(fx)

    def test_fixture_convenience_accessors_use_scores_layer(self):
        """``Fixture.home_team()`` / ``away_team()`` must apply the same
        4-layer resolution as ``LiveMatchState.from_fixture``."""
        fx = self._make_fixture(
            participants=[
                {"id": 51, "name": "Crystal Palace"},
                {"id": 9, "name": "Manchester City"},
            ],
            scores=[
                {"id": 1, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 3, "participant": "home"},
                 "participant_id": 9},
                {"id": 2, "fixture_id": 12345, "type_id": 1525,
                 "score": {"goals": 0, "participant": "away"},
                 "participant_id": 51},
            ],
        )
        assert fx.home_team().id == 9       # City via scores layer
        assert fx.away_team().id == 51      # Palace via scores layer
