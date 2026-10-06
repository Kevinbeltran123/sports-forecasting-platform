"""Live betting pipeline for TSP.

Architecture (per operator decisions 2026-05-24):

  state_tracker
      poll API-Football for fixture + events
      build MatchState (current_minute, score, cards, corners)

  residual_predictor
      given pre-match TSP predictions + MatchState,
      compute updated FT probabilities by re-modeling
      remaining-minutes-lambda

  alert_state
      track per-fixture-per-market alerts already sent
      re-alert when edge grows >= 5pp

  runner (not unit-tested directly — integration only)
      asyncio loop with adaptive polling
      orchestrates state -> residual -> detector -> bot v2

Polling cadence (adaptive):
  - 60s default
  - 15s when within 10 min of HT (minute 35-45) or FT (minute 80-95)
"""

from bip.evaluation.tournaments.team_style_profiler.live.alert_state import (
    AlertHistory,
    should_alert,
)
from bip.evaluation.tournaments.team_style_profiler.live.match_state import (
    MatchState,
    build_match_state,
)
from bip.evaluation.tournaments.team_style_profiler.live.residual_predictor import (
    LiveMarketPredictions,
    predict_live_markets,
)

__all__ = [
    "AlertHistory",
    "LiveMarketPredictions",
    "MatchState",
    "build_match_state",
    "predict_live_markets",
    "should_alert",
]
