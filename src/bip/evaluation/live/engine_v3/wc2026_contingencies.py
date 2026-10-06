"""WC2026 contingency rules — D-series.

These rules require auxiliary inputs that are NOT wired to live feeds
yet. They're built as pure functions over Pydantic models so the caller
can supply data however it likes (manual entry, future webhook, replay).

Rules:
    D1  LINEUP_GATE       at T-1h, verify expected stars are in starting XI;
                          reduce or cancel sizing on mismatch
    D2  MD1_CONDITIONER   reweight MD2 picks based on actual MD1 outcomes
    D3  FRIENDLY_UPDATER  re-rate a team after a notable pre-WC friendly
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.live.engine_v3.wc2026_patterns import MatchContext

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


# ── Shared output ────────────────────────────────────────────────────────


class StakeAdjustment(str, Enum):
    HOLD = "hold"
    REDUCE_50 = "reduce_50"
    CANCEL = "cancel"


class ContingencyVerdict(BaseModel):
    model_config = _MODEL_CONFIG
    rule_id: str
    triggered: bool
    adjustment: StakeAdjustment = StakeAdjustment.HOLD
    affected_team: str | None = None
    reason: str


# ── D1 — Lineup confirmation gate ────────────────────────────────────────

# Players whose presence in the starting XI is binary-load-bearing for
# the lock's H/D/A and the pattern picks. Drawn from the dossier flags
# (Pulisic, Haaland, Salah, Mané, Bellingham for "reduce"; Yamal, Kudus,
# Koulibaly, Aouar already confirmed OUT so they're not on this list).
KEY_PLAYERS_REDUCE_ON_OMIT: dict[str, frozenset[str]] = {
    "United States": frozenset({"Christian Pulisic"}),
    "Norway": frozenset({"Erling Haaland"}),
    "Egypt": frozenset({"Mohamed Salah"}),
    "Senegal": frozenset({"Sadio Mané"}),
    "England": frozenset({"Jude Bellingham"}),
    "Portugal": frozenset({"Bernardo Silva"}),
    "France": frozenset({"Kylian Mbappé"}),
    "Argentina": frozenset({"Lionel Messi"}),
    "Brazil": frozenset({"Vinicius Junior"}),
    "Germany": frozenset({"Florian Wirtz"}),
}


class LineupSnapshot(BaseModel):
    """Starting XI for one team at T-1h (or whenever the operator pulls)."""

    model_config = _MODEL_CONFIG
    team: str
    starting_xi: list[str] = Field(default_factory=list)
    captured_at_utc: datetime


def gate_d1_lineup(
    ctx: MatchContext,
    home_lineup: LineupSnapshot | None,
    away_lineup: LineupSnapshot | None,
) -> ContingencyVerdict:
    """Verify each team's expected stars are in the starting XI.

    If a known key player is missing from the listed XI, return
    ``REDUCE_50``. If both teams have a key omission, return ``CANCEL``.
    """
    if home_lineup is None and away_lineup is None:
        return ContingencyVerdict(
            rule_id="D1", triggered=False,
            reason="no lineup snapshots provided",
        )
    missing: list[tuple[str, str]] = []  # (team, missing_player)
    for snap in (home_lineup, away_lineup):
        if snap is None:
            continue
        expected = KEY_PLAYERS_REDUCE_ON_OMIT.get(snap.team, frozenset())
        xi_set = set(snap.starting_xi)
        for player in expected:
            if player not in xi_set:
                missing.append((snap.team, player))
    if not missing:
        return ContingencyVerdict(
            rule_id="D1", triggered=False,
            reason="all expected stars present in starting XI",
        )
    teams_with_misses = {team for team, _ in missing}
    adjust = StakeAdjustment.CANCEL if len(teams_with_misses) >= 2 else StakeAdjustment.REDUCE_50
    bullets = [f"{team} missing {player}" for team, player in missing]
    return ContingencyVerdict(
        rule_id="D1",
        triggered=True,
        adjustment=adjust,
        affected_team=next(iter(teams_with_misses)),
        reason="; ".join(bullets),
    )


# ── D2 — MD1 result conditioner ──────────────────────────────────────────


class MD1Result(BaseModel):
    """One team's MD1 outcome from the operator's perspective."""

    model_config = _MODEL_CONFIG
    team: str
    won: bool
    drew: bool
    lost: bool
    goals_for: int
    goals_against: int


def condition_d2_md1(
    ctx: MatchContext,
    home_md1: MD1Result | None,
    away_md1: MD1Result | None,
) -> ContingencyVerdict:
    """Reweight MD2 picks based on MD1 group-stage state.

    Heuristics (operational, not statistical):
    - Both teams won MD1 → group already decided directionally; expect
      rotation + reduced intensity → trigger REDUCE_50 on tight picks.
    - One team lost MD1 → that team enters MD2 must-win → goal-scoring
      props shift; pre-match "under" theses need re-eval → REDUCE_50.
    - Both teams lost MD1 → desperate game, more open than lock priced
      → REDUCE_50 on under-direction picks.
    """
    if home_md1 is None or away_md1 is None:
        return ContingencyVerdict(
            rule_id="D2", triggered=False,
            reason="incomplete MD1 results",
        )
    flags: list[str] = []
    if home_md1.won and away_md1.won:
        flags.append("both teams won MD1; expect rotation + intensity drop")
    elif home_md1.lost and away_md1.lost:
        flags.append("both teams lost MD1; must-win pressure both ways → game more open")
    elif home_md1.lost:
        flags.append(f"{ctx.home_team} lost MD1 → must-win posture, attack-leaning")
    elif away_md1.lost:
        flags.append(f"{ctx.away_team} lost MD1 → must-win posture, attack-leaning")
    if not flags:
        return ContingencyVerdict(
            rule_id="D2", triggered=False,
            reason="MD1 outcomes do not alter MD2 priors",
        )
    return ContingencyVerdict(
        rule_id="D2",
        triggered=True,
        adjustment=StakeAdjustment.REDUCE_50,
        reason="; ".join(flags),
    )


# ── D3 — Pre-WC friendly upweighting ─────────────────────────────────────


class FriendlyResult(BaseModel):
    """One friendly match the operator wants to factor in."""

    model_config = _MODEL_CONFIG
    date_str: str  # ISO date "YYYY-MM-DD"
    team: str
    opponent: str
    goals_for: int
    goals_against: int
    opponent_tier: str  # "elite_uefa" | "elite_conmebol" | "mid" | "weak"


D3_NOTABLE_TIERS: frozenset[str] = frozenset({"elite_uefa", "elite_conmebol"})


def upweight_d3_friendly(
    ctx: MatchContext,
    home_friendly: FriendlyResult | None,
    away_friendly: FriendlyResult | None,
) -> ContingencyVerdict:
    """Flag teams that beat (or held) an elite UEFA/CONMEBOL opponent
    in a pre-WC friendly post-2026-05-09.

    A notable upset (e.g. NZL 4-1 England) signals the team is being
    underrated by the lock. We return REDUCE_50 on pre-match theses
    against this team (because the lock might be 5-10pp too low on them).
    """
    notable: list[str] = []
    for fr in (home_friendly, away_friendly):
        if fr is None:
            continue
        if fr.opponent_tier not in D3_NOTABLE_TIERS:
            continue
        # Win or draw against an elite is notable
        if fr.goals_for >= fr.goals_against:
            notable.append(
                f"{fr.team} {fr.goals_for}-{fr.goals_against} vs "
                f"{fr.opponent} ({fr.opponent_tier}) on {fr.date_str}"
            )
    if not notable:
        return ContingencyVerdict(
            rule_id="D3", triggered=False,
            reason="no notable friendly results to upweight",
        )
    return ContingencyVerdict(
        rule_id="D3",
        triggered=True,
        adjustment=StakeAdjustment.REDUCE_50,
        reason="; ".join(notable),
    )


# ── Orchestrator ─────────────────────────────────────────────────────────


def run_contingencies(
    ctx: MatchContext,
    *,
    home_lineup: LineupSnapshot | None = None,
    away_lineup: LineupSnapshot | None = None,
    home_md1: MD1Result | None = None,
    away_md1: MD1Result | None = None,
    home_friendly: FriendlyResult | None = None,
    away_friendly: FriendlyResult | None = None,
) -> list[ContingencyVerdict]:
    """Run all D-rules with whatever inputs are available."""
    return [
        gate_d1_lineup(ctx, home_lineup, away_lineup),
        condition_d2_md1(ctx, home_md1, away_md1),
        upweight_d3_friendly(ctx, home_friendly, away_friendly),
    ]


def worst_adjustment(verdicts: list[ContingencyVerdict]) -> StakeAdjustment:
    """Return the strongest adjustment across verdicts (CANCEL beats
    REDUCE_50 beats HOLD)."""
    rank = {
        StakeAdjustment.HOLD: 0,
        StakeAdjustment.REDUCE_50: 1,
        StakeAdjustment.CANCEL: 2,
    }
    worst = StakeAdjustment.HOLD
    for v in verdicts:
        if rank[v.adjustment] > rank[worst]:
            worst = v.adjustment
    return worst


__all__ = [
    "ContingencyVerdict",
    "D3_NOTABLE_TIERS",
    "FriendlyResult",
    "KEY_PLAYERS_REDUCE_ON_OMIT",
    "LineupSnapshot",
    "MD1Result",
    "StakeAdjustment",
    "condition_d2_md1",
    "gate_d1_lineup",
    "run_contingencies",
    "upweight_d3_friendly",
    "worst_adjustment",
]
