"""Host-nation filter — L5.6 finding (internal research notes).

EVIDENCE. n=199 hold-out walk-forward backtest. 15 of 199 matches feature
a host nation. 5 of 20 worst-Brier-decile matches (25%) feature a host —
3.32x over-representation, p=0.0065 (survives BH-FDR @ q=0.10). Drivers:
Qatar (WC22), Côte d'Ivoire (AFCON23), Germany (Euro24).

WHY. Lock_v1 has no home-advantage feature for tournament hosts. Weak
host teams (Qatar at WC22 with low Bradley-Terry strength) get
systematically mispriced because the predictor doesn't know about the
implicit gamma_home boost from hosting.

APPLICATION. When a fixture involves the host nation of its tournament,
downgrade confidence on lock_v1 picks by HOST_DOWNGRADE_FACTOR (default
1.2x — i.e. require 20% more edge to qualify). For WC2026 with three
co-hosts (USA / Mexico / Canada), this triggers on a meaningful slice of
the tournament.

NOT a hard exclude (that would lose too many fixtures); a confidence
downgrade is the principled response when n=15 sample size is moderate.
"""
from __future__ import annotations

from dataclasses import dataclass


HOST_DOWNGRADE_FACTOR: float = 1.2
"""Multiplier applied to edge thresholds when host participates. 1.2 means
the fixture needs 20% more edge to qualify than a non-host fixture."""


def _hosts_for(tournament_slug: str) -> frozenset[str]:
    # Local import to avoid circular dep with __init__.py at module load.
    from bip.evaluation.tournaments.patterns_v2 import HOST_NATIONS

    return HOST_NATIONS.get(tournament_slug, frozenset())


def host_nation_in_fixture(
    home_team: str, away_team: str, tournament_slug: str
) -> bool:
    """True iff either team is the host nation for ``tournament_slug``.

    >>> host_nation_in_fixture("USA", "Argentina", "world_cup_2026")
    True
    >>> host_nation_in_fixture("Brazil", "Argentina", "world_cup_2026")
    False
    >>> host_nation_in_fixture("Qatar", "Ecuador", "wc_2022")
    True
    """
    hosts = _hosts_for(tournament_slug)
    return home_team in hosts or away_team in hosts


@dataclass(frozen=True)
class HostNationVerdict:
    """Outcome of host-nation filter for a single fixture."""

    home_team: str
    away_team: str
    tournament_slug: str
    host_in_fixture: bool
    downgrade_factor: float
    """1.0 if no host; HOST_DOWNGRADE_FACTOR otherwise."""
    rationale: str


def host_nation_verdict(
    home_team: str, away_team: str, tournament_slug: str
) -> HostNationVerdict:
    """Returns a HostNationVerdict for the fixture.

    Use this to gate the pick pipeline: multiply the configured edge
    threshold by ``verdict.downgrade_factor`` before deciding whether the
    fixture qualifies for auto-pick.
    """
    in_fixture = host_nation_in_fixture(home_team, away_team, tournament_slug)
    if not in_fixture:
        return HostNationVerdict(
            home_team=home_team,
            away_team=away_team,
            tournament_slug=tournament_slug,
            host_in_fixture=False,
            downgrade_factor=1.0,
            rationale="non-host fixture",
        )
    hosts = _hosts_for(tournament_slug)
    host_side = "home" if home_team in hosts else "away"
    return HostNationVerdict(
        home_team=home_team,
        away_team=away_team,
        tournament_slug=tournament_slug,
        host_in_fixture=True,
        downgrade_factor=HOST_DOWNGRADE_FACTOR,
        rationale=(
            f"host_nation ({host_side}); lock_v1 lacks host feature "
            f"(L5.6 evidence: n_host=15/199 hold-out, 3.32x over-rep in "
            f"worst-Brier-decile, p=0.0065)"
        ),
    )
