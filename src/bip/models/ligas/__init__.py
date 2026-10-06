"""bip.models.ligas — LigasModel for 5 top European leagues.

Sprint 1 of the v4 refactor. See plan:
  internal design notes (not published)

PATTERN: adapter, NOT rewrite. LigasModel wraps the existing battle-tested
utilities in src/bip/train/ (StackedEnsemble, WalkForwardSplitter,
calibrator, simulate_pick) and exposes them through the v4 BaseModel
contract. The legacy train/ code keeps working — engine_v3 still uses it.

Phase 1 markets (per spec §4.8): 1x2 + ou_2.5. BTTS/Corners are Phase 2,
gated by live performance of Phase 1.
"""

# Phase 1 markets — the implementation spec §4.8
MARKETS: tuple[str, ...] = ("1x2", "ou_2.5")

# 5 top European leagues — the implementation spec §4.4
SUPPORTED_LEAGUES: frozenset[str] = frozenset(
    {"PL", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
)

# API-Football league IDs (spec §4.4)
LEAGUE_ID_MAP: dict[str, int] = {
    "PL": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

# LigasModel imported after constants so model.py can `from bip.models.ligas
# import SUPPORTED_LEAGUES, MARKETS` without circular issues.
from bip.models.ligas.gate import (  # noqa: E402
    DEFAULT_POLICY,
    GateDecision,
    GatePolicy,
    GateVerdict,
    bootstrap_brier_ci,
    bootstrap_roi_ci,
    evaluate_gate,
)
from bip.models.ligas.model import LigasModel  # noqa: E402

__all__ = [
    "DEFAULT_POLICY",
    "GateDecision",
    "GatePolicy",
    "GateVerdict",
    "LEAGUE_ID_MAP",
    "LigasModel",
    "MARKETS",
    "SUPPORTED_LEAGUES",
    "bootstrap_brier_ci",
    "bootstrap_roi_ci",
    "evaluate_gate",
]
