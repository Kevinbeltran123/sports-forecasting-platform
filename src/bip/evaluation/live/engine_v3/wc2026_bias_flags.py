"""WC2026 bias flaggers — corrections for observed lock_v1 systematic
errors.

These rules do NOT produce picks. They emit ``BiasFlag`` annotations
that the operator (and downstream alerting) can use to decide:
- B1: "lock under-prices a strong favorite vs WC-debutant — play
       derivatives, not ML"
- B2: "lock tilts against the TM-elite at 6-16x; consider fade-direction
       on ML"
- B3: "team has a material post-lock event the model never saw"

Evidence: observational, drawn from internal research notes.
NOT FDR-validated like A-series. Treat as **hypothesis flags** for the
operator, not auto-actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.live.engine_v3.wc2026_patterns import MatchContext

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

# ── B1: known WC-debutants / historically-weak underdogs ────────────────

WC_DEBUTANT_OR_WEAK: frozenset[str] = frozenset({
    # First-ever WC participation in 2026
    "Curaçao", "Cape Verde", "Uzbekistan", "Haiti",
    # Long droughts + perennial group exits
    "Saudi Arabia", "Qatar", "Iraq", "Jordan",
    "Bosnia and Herzegovina",  # 2nd ever (2014, 2026)
    "New Zealand",  # OFC slot, first WC since 2010
    "Scotland",     # First WC since 1998
    "Norway",       # First WC since 1998 (but Haaland-anchored; weak flag only)
})

B1_TM_RATIO_THRESHOLD = 10.0


class LockPrediction(BaseModel):
    """Lock_v1 H/D/A probabilities for a fixture (slice of lock.json)."""

    model_config = _MODEL_CONFIG
    p_home_win: float = Field(ge=0.0, le=1.0)
    p_draw: float = Field(ge=0.0, le=1.0)
    p_away_win: float = Field(ge=0.0, le=1.0)


class BiasFlag(BaseModel):
    """One observational lock-bias annotation."""

    model_config = _MODEL_CONFIG
    flag_id: str
    triggered: bool
    severity: str  # "high" | "medium" | "low"
    affected_team: str | None = None
    market_guidance: str  # what TO do / what to AVOID
    reason: str


def flag_b1_strong_favorite_vs_debutant(
    ctx: MatchContext, lock: LockPrediction
) -> BiasFlag:
    """Detect the "lock under-prices favorite vs WC-debutant" pattern.

    Triggers when TM ratio ≥ 10x AND the underdog is in the known
    debutant/weak list. Recommends derivatives over moneyline.
    """
    if ctx.home_market_value_eur is None or ctx.away_market_value_eur is None:
        return BiasFlag(
            flag_id="B1", triggered=False, severity="low",
            market_guidance="—",
            reason="no market-value data",
        )
    if ctx.home_market_value_eur <= 0 or ctx.away_market_value_eur <= 0:
        return BiasFlag(
            flag_id="B1", triggered=False, severity="low",
            market_guidance="—",
            reason="zero market value",
        )
    h, a = ctx.home_market_value_eur, ctx.away_market_value_eur
    if h > a:
        ratio = h / a
        favorite, underdog = ctx.home_team, ctx.away_team
        fav_lock_prob = lock.p_home_win
    else:
        ratio = a / h
        favorite, underdog = ctx.away_team, ctx.home_team
        fav_lock_prob = lock.p_away_win

    if ratio < B1_TM_RATIO_THRESHOLD:
        return BiasFlag(
            flag_id="B1", triggered=False, severity="low",
            market_guidance="—",
            reason=f"TM ratio {ratio:.2f} < {B1_TM_RATIO_THRESHOLD}",
        )
    if underdog not in WC_DEBUTANT_OR_WEAK:
        return BiasFlag(
            flag_id="B1", triggered=False, severity="low",
            market_guidance="—",
            reason=f"underdog {underdog!r} not in debutant/weak list",
        )

    return BiasFlag(
        flag_id="B1",
        triggered=True,
        severity="high",
        affected_team=favorite,
        market_guidance=(
            f"Lock favors {favorite} at {fav_lock_prob:.2f}; market likely closes "
            f"+0.10 to +0.20 higher. Prefer derivatives (team-total Over, "
            "Asian-corner handicap, clean sheet) instead of moneyline."
        ),
        reason=(
            f"TM ratio {ratio:.2f}x ≥ {B1_TM_RATIO_THRESHOLD} and underdog "
            f"{underdog!r} is a known WC-debutant/historically-weak side"
        ),
    )


# ── B2: lock tilts against TM-elite at 6-16x band ────────────────────────

B2_TM_LOW, B2_TM_HIGH = 6.0, 16.0
B2_FAVORITE_CONFEDS: frozenset[str] = frozenset({"UEFA", "CONMEBOL"})
B2_UNDERDOG_CONFEDS: frozenset[str] = frozenset({"AFC", "CAF", "CONCACAF", "OFC"})

# Confederation lookup. Maps the 48 WC2026 team names to their
# confederation slug. Mid-tournament adds (knockout-only) get extended
# in the lock-ingest hook.
TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA (16)
    "France": "UEFA", "England": "UEFA", "Spain": "UEFA", "Germany": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Turkey": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Norway": "UEFA", "Sweden": "UEFA",
    "Czech Republic": "UEFA", "Austria": "UEFA", "Scotland": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    # CONMEBOL (6)
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    # CAF (9)
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Tunisia": "CAF",
    "Algeria": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF", "Cape Verde": "CAF",
    "South Africa": "CAF", "DR Congo": "CAF",
    # AFC (8)
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Uzbekistan": "AFC",
    "Jordan": "AFC", "Iraq": "AFC",
    # CONCACAF (6)
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curaçao": "CONCACAF",
    # OFC (1)
    "New Zealand": "OFC",
}


def flag_b2_lock_tilts_against_elite(
    ctx: MatchContext, lock: LockPrediction
) -> BiasFlag:
    """Detect the AUS-TUR / AUT-JOR / TUR-PAR systematic-bias pattern.

    Triggers when:
    - TM ratio is in [6, 16]
    - Favorite is UEFA or CONMEBOL
    - Underdog is AFC, CAF, CONCACAF, or OFC
    - Lock does NOT favor the TM-elite (lock_favorite_prob < 0.45)
    """
    if ctx.home_market_value_eur is None or ctx.away_market_value_eur is None:
        return BiasFlag(
            flag_id="B2", triggered=False, severity="low",
            market_guidance="—",
            reason="no market-value data",
        )
    if ctx.home_market_value_eur <= 0 or ctx.away_market_value_eur <= 0:
        return BiasFlag(
            flag_id="B2", triggered=False, severity="low",
            market_guidance="—",
            reason="zero market value",
        )

    h, a = ctx.home_market_value_eur, ctx.away_market_value_eur
    if h > a:
        ratio = h / a
        favorite, underdog = ctx.home_team, ctx.away_team
        fav_lock_prob = lock.p_home_win
    else:
        ratio = a / h
        favorite, underdog = ctx.away_team, ctx.home_team
        fav_lock_prob = lock.p_away_win

    if not (B2_TM_LOW <= ratio <= B2_TM_HIGH):
        return BiasFlag(
            flag_id="B2", triggered=False, severity="low",
            market_guidance="—",
            reason=f"TM ratio {ratio:.2f} outside [{B2_TM_LOW}, {B2_TM_HIGH}] band",
        )

    fav_conf = TEAM_CONFEDERATION.get(favorite, "UNKNOWN")
    dog_conf = TEAM_CONFEDERATION.get(underdog, "UNKNOWN")
    if fav_conf not in B2_FAVORITE_CONFEDS:
        return BiasFlag(
            flag_id="B2", triggered=False, severity="low",
            market_guidance="—",
            reason=f"favorite confederation {fav_conf!r} not UEFA/CONMEBOL",
        )
    if dog_conf not in B2_UNDERDOG_CONFEDS:
        return BiasFlag(
            flag_id="B2", triggered=False, severity="low",
            market_guidance="—",
            reason=f"underdog confederation {dog_conf!r} not AFC/CAF/CONCACAF/OFC",
        )

    if fav_lock_prob >= 0.45:
        return BiasFlag(
            flag_id="B2", triggered=False, severity="low",
            market_guidance="—",
            reason=(
                f"lock favors elite ({favorite} prob={fav_lock_prob:.2f} ≥ 0.45); "
                "no fade signal"
            ),
        )

    return BiasFlag(
        flag_id="B2",
        triggered=True,
        severity="medium",
        affected_team=favorite,
        market_guidance=(
            f"Lock has {favorite} at {fav_lock_prob:.2f} despite TM {ratio:.1f}x. "
            f"Consider fade-direction: {favorite} ML at implied ≤0.45-0.50, "
            "Kelly 1/8."
        ),
        reason=(
            f"TM ratio {ratio:.2f} in [6, 16] band; {favorite} ({fav_conf}) vs "
            f"{underdog} ({dog_conf}); lock fav-prob {fav_lock_prob:.2f} < 0.45"
        ),
    )


# ── B3: post-lock material events registry ───────────────────────────────

LOCK_EMIT_DATE = date(2026, 5, 9)


@dataclass(frozen=True)
class PostLockEvent:
    team: str
    event_type: str  # "injury_out" | "coach_change" | "squad_omission" | "fitness_doubt"
    description: str
    event_date: date
    severity: str  # "high" | "medium" | "low"


# Source: post-lock news compiled from dossiers 01-44.
POST_LOCK_EVENTS: tuple[PostLockEvent, ...] = (
    PostLockEvent(
        team="Spain", event_type="injury_out",
        description="Yamal hamstring; OUT MD1 + MD2, target return MD3 vs Uruguay 2026-06-27",
        event_date=date(2026, 4, 22), severity="high",
    ),
    PostLockEvent(
        team="Japan", event_type="injury_out",
        description="Mitoma hamstring; ruled out tournament, omitted from squad 2026-05-15",
        event_date=date(2026, 5, 9), severity="high",
    ),
    PostLockEvent(
        team="Netherlands", event_type="injury_out",
        description="Xavi Simons serious knee; OUT tournament",
        event_date=date(2026, 5, 1), severity="high",
    ),
    PostLockEvent(
        team="Netherlands", event_type="fitness_doubt",
        description="Depay shin + grade-2 thigh; 'could miss WC entirely' per Tribuna 2026-05-20",
        event_date=date(2026, 5, 20), severity="medium",
    ),
    PostLockEvent(
        team="Ghana", event_type="fitness_doubt",
        description="Kudus thigh; Spurs: 'for this season has gone' — HIGH RISK",
        event_date=date(2026, 1, 8), severity="high",
    ),
    PostLockEvent(
        team="Senegal", event_type="injury_out",
        description="Koulibaly thigh hematoma; season-ending per Pulse Sports 2026-05-07",
        event_date=date(2026, 4, 8), severity="high",
    ),
    PostLockEvent(
        team="Algeria", event_type="fitness_doubt",
        description="Aouar hospitalized per Al Ittihad announcement May 2026",
        event_date=date(2026, 5, 10), severity="medium",
    ),
    PostLockEvent(
        team="Saudi Arabia", event_type="coach_change",
        description="Donis appointed 2026-04-24 after Renard sacked 2026-04-17",
        event_date=date(2026, 4, 24), severity="high",
    ),
    PostLockEvent(
        team="Ghana", event_type="coach_change",
        description="Queiroz appointed 2026-04-14 after Addo dismissed",
        event_date=date(2026, 4, 14), severity="high",
    ),
    PostLockEvent(
        team="Uzbekistan", event_type="coach_change",
        description="Cannavaro appointed 2025-10-06 after Kapadze interim period",
        event_date=date(2025, 10, 6), severity="medium",
    ),
    PostLockEvent(
        team="Canada", event_type="fitness_doubt",
        description="Davies third muscle injury since Feb; MD1 doubtful, MD2 more realistic re-entry",
        event_date=date(2026, 3, 1), severity="medium",
    ),
    PostLockEvent(
        team="Bosnia and Herzegovina", event_type="squad_omission",
        description="Pjanić retired Dec 2025; midfield reshape (Tahirović + Gigović)",
        event_date=date(2025, 12, 1), severity="medium",
    ),
    PostLockEvent(
        team="Jordan", event_type="injury_out",
        description="Al-Naimat ACL ruptured Dec 2025; Jordan's #9 replaced by Olwan",
        event_date=date(2025, 12, 12), severity="high",
    ),
    PostLockEvent(
        team="Iran", event_type="squad_omission",
        description="Azmoun politically omitted (Dubai-ruler Instagram photo controversy)",
        event_date=date(2026, 3, 1), severity="medium",
    ),
    PostLockEvent(
        team="Belgium", event_type="fitness_doubt",
        description="Courtois (Mar thigh) + Lukaku (Aug 2025 quad, only 1G in 6 apps) + KDB (hamstring Oct→Mar)",
        event_date=date(2026, 3, 11), severity="medium",
    ),
    PostLockEvent(
        team="Portugal", event_type="fitness_doubt",
        description="Rúben Dias hamstring since mid-March 2026",
        event_date=date(2026, 3, 15), severity="medium",
    ),
    PostLockEvent(
        team="United States", event_type="fitness_doubt",
        description="Pulisic gluteal + Richards ankle ligaments (UECL final) + Weah undisclosed",
        event_date=date(2026, 5, 1), severity="medium",
    ),
    PostLockEvent(
        team="Ivory Coast", event_type="fitness_doubt",
        description="Ndicka (Roma season-ending) + Kossounou (grade-1 tear 2026-05-10) — CB depth crisis",
        event_date=date(2026, 5, 10), severity="medium",
    ),
)


def post_lock_events_for(team: str) -> tuple[PostLockEvent, ...]:
    """Return all known post-lock events for a team."""
    return tuple(ev for ev in POST_LOCK_EVENTS if ev.team == team)


def flag_b3_post_lock_events(ctx: MatchContext) -> BiasFlag:
    """Flag any material post-lock event affecting either side."""
    home_events = post_lock_events_for(ctx.home_team)
    away_events = post_lock_events_for(ctx.away_team)
    if not home_events and not away_events:
        return BiasFlag(
            flag_id="B3", triggered=False, severity="low",
            market_guidance="—",
            reason="no known post-lock events for either team",
        )
    all_events = home_events + away_events
    # Severity rank: high=2 > medium=1 > low=0.
    _SEVERITY_RANK = {"high": 2, "medium": 1, "low": 0}
    highest = max(all_events, key=lambda e: _SEVERITY_RANK.get(e.severity, 0))
    bullets = []
    for ev in all_events:
        bullets.append(f"[{ev.severity.upper()}] {ev.team}: {ev.description}")
    return BiasFlag(
        flag_id="B3",
        triggered=True,
        severity=highest.severity,
        affected_team=highest.team,
        market_guidance=(
            "Lock_v1 emitted 2026-05-09 does NOT see these events. "
            "Re-evaluate fixture manually before staking."
        ),
        reason=" | ".join(bullets),
    )


__all__ = [
    "B1_TM_RATIO_THRESHOLD",
    "B2_FAVORITE_CONFEDS",
    "B2_TM_HIGH",
    "B2_TM_LOW",
    "B2_UNDERDOG_CONFEDS",
    "BiasFlag",
    "LOCK_EMIT_DATE",
    "LockPrediction",
    "POST_LOCK_EVENTS",
    "PostLockEvent",
    "TEAM_CONFEDERATION",
    "WC_DEBUTANT_OR_WEAK",
    "flag_b1_strong_favorite_vs_debutant",
    "flag_b2_lock_tilts_against_elite",
    "flag_b3_post_lock_events",
    "post_lock_events_for",
]
