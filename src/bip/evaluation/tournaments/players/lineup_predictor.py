"""Predict the starting XI for a national team in a tournament match.

Three heuristic strategies, chained as fallbacks:

1. **Confirmed lineup** (most reliable): if API-Football returned a lineup
   for the fixture (typically published ~60 minutes pre-kickoff), use it
   directly. Tagged confidence='confirmed'.

2. **Most-frequent qualifier XI**: count how often each squad member
   started (was in the XI from kickoff) across the team's qualifying
   matches. The 11 with highest start-count, broken by minutes-played.
   Tagged confidence='heuristic_qualifier'.

3. **Position-balanced fallback**: when neither confirmed lineup nor
   qualifier history is available, build a 4-3-3 from the squad using
   the highest-minute player at each position (1 G, 4 D, 3 M, 3 F).
   Tagged confidence='fallback_squad'.

R-02 mitigation (SPIKE.md §5): the confidence tag travels with the
prediction so downstream consumers can flag low-confidence picks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

import structlog

from bip.evaluation.tournaments.models import Player, Position

logger = structlog.get_logger(__name__)


class LineupConfidence(str, Enum):
    CONFIRMED = "confirmed"
    HEURISTIC_QUALIFIER = "heuristic_qualifier"
    FALLBACK_SQUAD = "fallback_squad"


@dataclass(frozen=True)
class LineupPrediction:
    """11 players + the confidence basis."""

    players: tuple[Player, ...]
    confidence: LineupConfidence

    def __post_init__(self) -> None:
        if len(self.players) != 11:
            raise ValueError(f"LineupPrediction expects 11 players, got {len(self.players)}")


def predict_from_confirmed(confirmed_xi: list[Player]) -> LineupPrediction:
    """Wrap a confirmed XI list. Caller pulled it from /fixtures/lineups."""
    if len(confirmed_xi) != 11:
        raise ValueError(
            f"Confirmed XI must have 11 players, got {len(confirmed_xi)}"
        )
    return LineupPrediction(
        players=tuple(confirmed_xi),
        confidence=LineupConfidence.CONFIRMED,
    )


def predict_from_qualifier_history(
    squad: list[Player],
    starter_counts: dict[int, int],
    minutes_played: dict[int, int],
) -> LineupPrediction:
    """Pick the 11 squad members who started most frequently in qualifiers.

    Args:
        squad: Full roster (18-30 players).
        starter_counts: api_football_id -> # times in starting XI in qualifiers.
        minutes_played: api_football_id -> total qualifier minutes (tiebreaker).

    Returns:
        LineupPrediction with 11 players. If insufficient data (fewer than
        11 squad members have any starter_counts), falls through to
        fallback_squad strategy via predict_from_squad_only.
    """
    candidates = [p for p in squad if p.api_football_id in starter_counts]
    if len(candidates) < 11:
        logger.warning(
            "lineup_predictor_insufficient_qualifier_data",
            squad_size=len(squad),
            with_history=len(candidates),
        )
        return predict_from_squad_only(squad)

    # Sort by (starter_count desc, minutes desc, api_football_id asc — stable).
    ranked = sorted(
        candidates,
        key=lambda p: (
            -starter_counts.get(p.api_football_id, 0),
            -minutes_played.get(p.api_football_id, 0),
            p.api_football_id,
        ),
    )
    chosen = tuple(ranked[:11])
    return LineupPrediction(
        players=chosen,
        confidence=LineupConfidence.HEURISTIC_QUALIFIER,
    )


def predict_from_squad_only(squad: list[Player]) -> LineupPrediction:
    """Fallback: pick a 4-3-3 from squad using position-only heuristic.

    Required positions: 1 GK, 4 DEF, 3 MID, 3 FWD. If the squad lacks
    enough players at any position, raise ValueError.
    """
    by_position: dict[Position, list[Player]] = {pos: [] for pos in Position}
    for p in squad:
        by_position[p.position].append(p)

    requirements = {
        Position.GOALKEEPER: 1,
        Position.DEFENDER: 4,
        Position.MIDFIELDER: 3,
        Position.FORWARD: 3,
    }

    chosen: list[Player] = []
    for pos, count in requirements.items():
        bucket = by_position[pos]
        if len(bucket) < count:
            raise ValueError(
                f"Squad lacks {count} players at position {pos.value}; "
                f"only {len(bucket)} available"
            )
        # Stable tiebreak by api_football_id (lower id = older / more capped player).
        bucket_sorted = sorted(bucket, key=lambda p: p.api_football_id)
        chosen.extend(bucket_sorted[:count])

    return LineupPrediction(
        players=tuple(chosen),
        confidence=LineupConfidence.FALLBACK_SQUAD,
    )


def starter_count_from_lineups(
    lineups_responses: list[dict],
    team_id: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Aggregate starter counts and minutes from a list of /fixtures/lineups payloads.

    Args:
        lineups_responses: list of payloads from get_lineups(fixture_id) for
            the team's qualifier matches.
        team_id: API-Football team ID to filter on.

    Returns:
        (starter_counts, minutes_played) — both keyed by api_football_id.
        Note: minutes are approximated as 90 per starter appearance since
        per-fixture minute granularity requires /fixtures/players (heavier).
    """
    starts: Counter[int] = Counter()
    minutes: Counter[int] = Counter()
    for payload in lineups_responses:
        for team_block in payload.get("response", []):
            if team_block.get("team", {}).get("id") != team_id:
                continue
            for entry in team_block.get("startXI", []):
                pid = entry.get("player", {}).get("id")
                if pid is not None:
                    starts[pid] += 1
                    minutes[pid] += 90
    return dict(starts), dict(minutes)
