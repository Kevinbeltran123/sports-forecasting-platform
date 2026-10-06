"""Tests for tactical identity (planteamiento) derivation + matchup reads."""
from __future__ import annotations

from datetime import datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.statsbomb_advanced import (
    TeamStatsBombProfile,
)
from bip.evaluation.tournaments.team_style_profiler.tactical_identity import (
    PHILOSOPHY_ANCHORS,
    SCOUTING_IDENTITIES,
    MarketLean,
    TacticalIdentity,
    _finishing_profile,
    _press_intensity,
    _set_piece_reliance,
    derive_tactical_identity,
    finishing_regression,
    read_matchup,
    synthesize_archetype,
    tactical_identity_for,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat


def _ds(mean: float, n: int = 12) -> DistributionStat:
    return DistributionStat(mean=mean, ci_low=mean, ci_high=mean, n=n)


def _identity(
    name: str = "T",
    *,
    press="balanced_press",
    set_piece="mixed",
    finishing="neutral",
    ppda=12.0,
    sp_share=0.15,
    conv=1.0,
    n=12,
    anchor=None,
) -> TacticalIdentity:
    return TacticalIdentity(
        team_name=name,
        n_matches=n,
        confidence="green" if n >= 10 else "yellow" if n >= 5 else "red",
        press_intensity=press,
        set_piece_reliance=set_piece,
        finishing_profile=finishing,
        ppda=ppda,
        set_piece_xg_share=sp_share,
        conversion_rate=conv,
        archetype=synthesize_archetype(press, set_piece, finishing),
        anchor=anchor,
    )


# ── Dimension thresholds: boundary parametrization  ──


@pytest.mark.parametrize(
    "ppda,expected",
    [
        (6.1, "high_press"),     # Spain (real)
        (11.0, "high_press"),    # boundary: <= 11.0 inclusive
        (11.01, "balanced_press"),
        (13.49, "balanced_press"),
        (13.5, "low_block"),     # boundary: >= 13.5 inclusive
        (32.0, "low_block"),     # Qatar (real)
    ],
)
def test_press_intensity_boundaries(ppda, expected):
    assert _press_intensity(ppda) == expected


@pytest.mark.parametrize(
    "share,expected",
    [
        (0.0, "open_play"),
        (0.09, "open_play"),     # boundary: <= 0.09 inclusive
        (0.091, "mixed"),
        (0.209, "mixed"),
        (0.21, "set_piece_reliant"),  # boundary: >= 0.21 inclusive
        (0.50, "set_piece_reliant"),
    ],
)
def test_set_piece_reliance_boundaries(share, expected):
    assert _set_piece_reliance(share) == expected


@pytest.mark.parametrize(
    "conv,expected",
    [
        (0.35, "wasteful"),
        (0.90, "wasteful"),      # boundary: <= 0.90 inclusive
        (0.901, "neutral"),
        (1.099, "neutral"),
        (1.10, "clinical"),      # boundary: >= 1.10 inclusive
        (1.81, "clinical"),
    ],
)
def test_finishing_profile_boundaries(conv, expected):
    assert _finishing_profile(conv) == expected


# ── Archetype synthesis ──


@pytest.mark.parametrize(
    "press,sp,fin,expected",
    [
        ("high_press", "mixed", "clinical", "posesión-presión letal"),
        ("high_press", "mixed", "wasteful", "dominador estéril"),
        ("high_press", "set_piece_reliant", "clinical", "presión + pizarra"),
        ("low_block", "set_piece_reliant", "neutral", "especialista a balón parado"),
        ("low_block", "mixed", "clinical", "contragolpe letal"),
        ("low_block", "mixed", "wasteful", "muro reactivo"),
        ("balanced_press", "mixed", "neutral", "pragmático flexible"),
        ("balanced_press", "set_piece_reliant", "neutral", "equilibrado con balón parado"),
    ],
)
def test_synthesize_archetype(press, sp, fin, expected):
    assert synthesize_archetype(press, sp, fin) == expected


def test_wasteful_high_press_beats_set_piece_branch():
    # dominador estéril must win over "presión + pizarra" when wasteful.
    assert synthesize_archetype("high_press", "set_piece_reliant", "wasteful") == "dominador estéril"


# ── Matchup reads ──


def _has(leans, substr: str) -> bool:
    return any(substr in l.text for l in leans)


def _lean(leans, substr: str):
    return next(l for l in leans if substr in l.market)


def test_two_low_blocks_lean_under():
    home = _identity("Qatar", press="low_block", finishing="wasteful")
    away = _identity("KSA", press="low_block", set_piece="set_piece_reliant", sp_share=0.23)
    r = read_matchup(home, away)
    assert r.game_shape == "cerrado"
    assert _has(r.market_leans, "Under 2.5")
    assert _has(r.market_leans, "BTTS No")
    # structured leans, all BACK on a cagey game
    assert _lean(r.market_leans, "Under 2.5").direction == "BACK"


def test_two_high_press_lean_over():
    home = _identity("ARG", press="high_press", finishing="clinical")
    away = _identity("MEX", press="high_press", finishing="clinical")
    r = read_matchup(home, away)
    assert r.game_shape == "abierto"
    assert _has(r.market_leans, "Over 2.5")
    assert _has(r.market_leans, "BTTS Sí")


def test_wasteful_presser_vs_block_warns_under_and_ah():
    presser = _identity("BRA", press="high_press", finishing="wasteful", ppda=9.9)
    blocker = _identity("IRN", press="low_block", finishing="wasteful", ppda=19.9)
    r = read_matchup(presser, blocker)
    assert r.game_shape == "territorial"
    assert _has(r.market_leans, "domina pero no concreta")
    ah = _lean(r.market_leans, "AH -1.5 BRA")
    assert ah.direction == "FADE"
    assert _has(r.market_leans, "Córners BRA Over")


def test_clinical_presser_vs_block_backs_ah():
    presser = _identity("ESP", press="high_press", finishing="clinical", ppda=6.1)
    blocker = _identity("MAR", press="low_block", finishing="neutral", ppda=16.4)
    r = read_matchup(presser, blocker)
    ah = _lean(r.market_leans, "AH -0.5/-1 ESP")
    assert ah.direction == "BACK"


def test_red_flag_produces_caveat():
    home = _identity("Qatar", press="low_block", n=3)
    away = _identity("KSA", press="low_block", n=12)
    r = read_matchup(home, away)
    assert any("muestra muy pequeña" in c and "Qatar" in c for c in r.caveats)


def test_anchor_note_surfaced_as_caveat():
    morocco_anchor = PHILOSOPHY_ANCHORS["Morocco"]
    assert morocco_anchor.note  # has a coach-change note
    home = _identity("Spain", press="high_press", finishing="clinical")
    away = _identity("Morocco", press="low_block", anchor=morocco_anchor)
    r = read_matchup(home, away)
    assert any("Morocco" in c and "Ouahbi" in c for c in r.caveats)


# ── Full derivation wiring ──


def test_derive_tactical_identity_end_to_end():
    profile = TeamStatsBombProfile(
        team_name="Spain",
        confederation="UEFA",
        n_matches=21,
        competitions_covered=["FIFA World Cup 2022"],
        last_updated=datetime(2026, 5, 30),
        xg_for_per_match=_ds(2.45),
        xg_against_per_match=_ds(1.0),
        xg_open_play_per_match=_ds(2.0),
        xg_set_piece_per_match=_ds(0.4),
        xg_penalty_per_match=_ds(0.05),
        big_chances_per_match=_ds(3.0),
        big_chances_against_per_match=_ds(1.0),
        shots_per_match=_ds(15.0),
        shots_against_per_match=_ds(8.0),
        pressures_per_match=_ds(146.0),
        ppda=_ds(6.1),
        conversion_rate=_ds(1.17),
        set_piece_xg_share=_ds(0.19),
    )
    tid = derive_tactical_identity(profile)
    assert tid.team_name == "Spain"
    assert tid.press_intensity == "high_press"
    assert tid.finishing_profile == "clinical"
    assert tid.archetype == "posesión-presión letal"
    assert tid.confidence == "green"
    assert tid.anchor is not None  # Spain seeded in PHILOSOPHY_ANCHORS
    assert tid.is_measured


# ── Scouting identities (8 teams without StatsBomb) ──


def test_eight_scouting_teams_present():
    expected = {
        "Norway", "New Zealand", "Uzbekistan", "Jordan", "Iraq",
        "Bosnia and Herzegovina", "Haiti", "Curaçao",
    }
    assert set(SCOUTING_IDENTITIES) == expected


@pytest.mark.parametrize("name", list(SCOUTING_IDENTITIES))
def test_scouting_identity_shape(name):
    tid = SCOUTING_IDENTITIES[name]
    assert tid.confidence == "scouting"
    assert not tid.is_measured
    assert tid.ppda is None and tid.set_piece_xg_share is None and tid.conversion_rate is None
    assert tid.archetype  # archetype still synthesized from scouting labels
    assert tid.anchor is not None


def test_tactical_identity_for_resolver():
    # no profile + scouting team -> scouting identity
    assert tactical_identity_for("Norway").confidence == "scouting"
    # unknown team, no profile -> None
    assert tactical_identity_for("Atlantis") is None


def test_scouting_vs_measured_matchup_flags_caveat():
    measured = _identity("Brazil", press="high_press", finishing="wasteful", ppda=9.9)
    scouting = SCOUTING_IDENTITIES["Norway"]
    r = read_matchup(measured, scouting)
    # read still works; scouting caveat surfaced; no crash on None numerics
    assert any("SCOUTING" in c and "Norway" in c for c in r.caveats)
    assert r.market_leans  # leans still produced from labels


def test_set_piece_lean_handles_none_share():
    # scouting team flagged set_piece_reliant has None share -> no crash, no %.
    nz = SCOUTING_IDENTITIES["New Zealand"]  # set_piece_reliant, share None
    opp = _identity("Spain", press="high_press", finishing="clinical")
    r = read_matchup(opp, nz)
    sp = next((l for l in r.market_leans if "balón parado New Zealand" in l.market), None)
    assert sp is not None
    assert "%" not in sp.rationale  # no share rendered when None


# ── Cross-confirmation: game-plan leans ↔ TSV picks ──


def _matchup_with_leans(leans: list[MarketLean]):
    from bip.evaluation.tournaments.team_style_profiler.tactical_identity import MatchupRead
    h = _identity("H", press="low_block")
    a = _identity("A", press="low_block")
    return MatchupRead(home=h, away=a, tempo="t", game_shape="cerrado",
                       market_leans=leans, caveats=[])


def test_cross_confirmation_boosts_aligned_pick():
    from bip.evaluation.tournaments.team_style_profiler.match_dossier import (
        Pick, _apply_tactical_confirmation,
    )
    pick = Pick(market="Under 2.5 goals", direction="BACK", category="MODERATE",
                score=0.50, rationale=["base"])
    read = _matchup_with_leans([MarketLean("Under 2.5", "BACK", "dos bloques bajos")])
    out = _apply_tactical_confirmation([pick], read)
    assert out[0].score > 0.50
    assert any("✓ Planteamiento confirma" in r for r in out[0].rationale)


def test_cross_confirmation_flags_conflicting_pick():
    from bip.evaluation.tournaments.team_style_profiler.match_dossier import (
        Pick, _apply_tactical_confirmation,
    )
    pick = Pick(market="Under 2.5 goals", direction="BACK", category="MODERATE",
                score=0.50, rationale=["base"])
    read = _matchup_with_leans([MarketLean("Over 2.5", "BACK", "ida y vuelta")])
    out = _apply_tactical_confirmation([pick], read)
    assert out[0].score == 0.50  # not boosted
    assert any("sugiere lo contrario" in f for f in out[0].risk_flags)


def test_cross_confirmation_caution_adds_flag_no_boost():
    from bip.evaluation.tournaments.team_style_profiler.match_dossier import (
        Pick, _apply_tactical_confirmation,
    )
    pick = Pick(market="BTTS No", direction="BACK", category="MODERATE",
                score=0.50, rationale=["base"])
    read = _matchup_with_leans([MarketLean("BTTS No", "CAUTION", "rival vivo al contragolpe")])
    out = _apply_tactical_confirmation([pick], read)
    assert out[0].score == 0.50
    assert any("Planteamiento" in f for f in out[0].risk_flags)


# ── Finishing mean-reversion (regression signal) ──


def test_finishing_regression_fades_overperformer():
    note = finishing_regression(_identity("Spain", conv=1.30, n=12))
    assert note is not None and "sobre-rinde" in note and "FADE" in note


def test_finishing_regression_backs_underperformer():
    note = finishing_regression(_identity("France", conv=0.70, n=12))
    assert note is not None and "infra-rinde" in note and "BACK" in note


def test_finishing_regression_none_when_neutral():
    assert finishing_regression(_identity("X", conv=1.0, n=12)) is None


def test_finishing_regression_gated_by_low_confidence():
    # overperforming but only 3 matches (red) → too noisy, no signal
    assert finishing_regression(_identity("Y", conv=1.40, n=3)) is None


def test_finishing_regression_none_without_conversion():
    assert finishing_regression(_identity("Z", conv=None, n=12)) is None
