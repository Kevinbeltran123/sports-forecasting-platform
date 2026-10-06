"""Team Style Profiler (TSP) — WC2026 tactical-DNA system.

Independent of lock_v1. Builds a per-team style vector (TSV) filtered to
matches with the team's CURRENT coach since Jan 2023, then cross-team
predicts market-specific probabilities (BTTS, O/U goals, corners,
cards, AH, 1X2) and detects edge ≥+3% vs Pinnacle/Betano odds.

Design doc: internal notes

Architecture:
    tsv_schema       — Pydantic v2 models for TSV + sub-profiles
    coach_history    — current-coach mapping for WC2026 squads
    ingest           — API-Football pulls (fixtures + statistics + events)
    tsv_calculator   — compute the ~20 metrics from raw data
    tsv_validator    — flag red/green based on n_matches ≥ 10
    confederation_cohort — fallback when a team has no profile
    cross_team_predictor — pair two TSVs, output per-market probabilities
    value_detector   — compare to bookmaker odds, emit edge alerts
    live/            — live-state pulling + conditional re-prediction
"""

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    CoachInfo,
    DistributionStat,
    SubProfile,
    TeamStyleVector,
)

__all__ = [
    "CoachInfo",
    "DistributionStat",
    "SubProfile",
    "TeamStyleVector",
]
