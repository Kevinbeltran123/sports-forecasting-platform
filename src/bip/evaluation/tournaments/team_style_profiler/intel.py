"""Match Intel assembler — fuses every collected layer into one fixture view
WITHOUT losing anything relevant.

Design (operator directive 2026-05-30: integrate all info, lose NOTHING):
  - Additive & optional: each layer is a section; a missing layer degrades to a
    note, never a fabricated value.
  - Provenance travels: every section keeps its source + confidence.
  - Cross-source links ANNOTATE, never delete: an injured player's prop is
    flagged ⛔, not removed; the full injury list is always shown.
  - Conservative name matching: token-subset or surname+initial; ambiguous →
    abstain and record it (a false link loses more fidelity than no link).

Wraps the existing match_dossier (picks + planteamiento + patterns) untouched
and adds: availability, referee+injury-annotated prop boards, weak-links,
creators, advanced team shape.
"""
from __future__ import annotations

import dataclasses
import unicodedata
from dataclasses import dataclass, field

from bip.evaluation.tournaments.team_style_profiler.advanced_metrics import (
    AdvancedTeamProfile,
)
from bip.evaluation.tournaments.team_style_profiler.match_dossier import MatchDossier
from bip.evaluation.tournaments.team_style_profiler.player_advanced import (
    PlayerAdvancedProfile,
    weak_links,
)
from bip.evaluation.tournaments.team_style_profiler.player_props import (
    PlayerPropProfile,
    PropCandidate,
    prop_board,
)
from bip.evaluation.tournaments.team_style_profiler.press_resistance import (
    PressResistance,
    press_resistance_note,
)
from bip.evaluation.tournaments.team_style_profiler.referee_tendencies import (
    RefereeTendency,
    apply_referee_to_board,
    referee_match_note,
)
from bip.evaluation.tournaments.team_style_profiler.set_piece_intel import (
    SetPieceIntel,
    set_piece_for,
)
from bip.evaluation.tournaments.team_style_profiler.tactical_identity import (
    TacticalIdentity,
    finishing_regression,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import TeamStyleVector


# ── Conservative cross-source name matching (the "lose nothing" crux) ──


def normalize_name(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return [t for t in s.lower().replace("-", " ").replace(".", " ").split() if t]


def match_name(target: str, candidates: list[str]) -> str | None:
    """Return the single candidate matching `target`, else None.

    Matches on token-subset (either direction) or surname+first-initial.
    If MORE THAN ONE candidate matches, returns None (abstain — never guess).
    """
    tt = normalize_name(target)
    if not tt:
        return None
    tset = set(tt)
    hits = []
    for c in candidates:
        ct = normalize_name(c)
        if not ct:
            continue
        cset = set(ct)
        subset = tset <= cset or cset <= tset
        surname_initial = tt[-1] == ct[-1] and tt[0][:1] == ct[0][:1]
        if subset or surname_initial:
            hits.append(c)
    return hits[0] if len(hits) == 1 else None


# ── Sections ──


@dataclass(frozen=True)
class Injury:
    name: str
    injury: str
    until: str | None
    ongoing: bool


@dataclass(frozen=True)
class WeakLinkNote:
    team: str
    player: str
    position: str
    reason: str
    kind: str


@dataclass(frozen=True)
class Creator:
    team: str
    player: str
    position: str
    sca_per90: float
    prog_passes_per90: float


@dataclass(frozen=True)
class DuelMatchup:
    """An attacking threat crossed against the opponent's matching weak-link.

    Evidence, not verdict: the analyst confirms the two players actually meet in
    the same zone before betting it (positions are coarse, sides not enforced)."""

    attacker_team: str
    attacker: str
    defender_team: str
    defender: str
    defender_position: str
    kind: str            # "pace/1v1" | "aerial"
    market: str          # "faltas/tarjetas/penal" | "cabeza/córner"
    read: str


@dataclass(frozen=True)
class AdvancedShape:
    team: str
    field_tilt: float
    line_height: float
    directness: float
    gk_goals_prevented: float
    game_state: str          # front-runner / chaser / game-manager / balanced
    read: str                # one-line analyst read


@dataclass(frozen=True)
class GameScript:
    """Projected pre-match game-state trajectory from the strength gap.

    The empirically-grounded corners signal: the team that ends up CHASING wins
    corners. Pre-match we only project who is *likely* to chase; the live engine
    confirms or flips it (an early goal by the underdog inverts the script)."""

    favorite: str | None       # None when the gap is too small to call
    gap: float                 # λ_home − λ_away (home perspective, from TSV goals)
    read: str
    leans: list[str]


@dataclass(frozen=True)
class MatchIntel:
    dossier: MatchDossier                       # existing picks + planteamiento + patterns
    home_injuries: list[Injury]
    away_injuries: list[Injury]
    home_prop_board: list[PropCandidate]        # referee + injury annotated (depth)
    away_prop_board: list[PropCandidate]
    weak_links: list[WeakLinkNote]
    creators: list[Creator]
    home_shape: AdvancedShape | None
    away_shape: AdvancedShape | None
    provenance_notes: list[str] = field(default_factory=list)  # unmatched / missing layers
    blind_spots: list[str] = field(default_factory=list)       # §0 "investigate yourself"
    home_set_piece: SetPieceIntel | None = None
    away_set_piece: SetPieceIntel | None = None
    home_recent_board: list[PropCandidate] = field(default_factory=list)  # ESPN current form
    away_recent_board: list[PropCandidate] = field(default_factory=list)
    home_lineup: list[str] = field(default_factory=list)  # confirmed XI (API-Football)
    away_lineup: list[str] = field(default_factory=list)
    duels: list[DuelMatchup] = field(default_factory=list)  # threat × weak-link cross
    game_script: GameScript | None = None                   # projected game-state trajectory
    regression_notes: list[str] = field(default_factory=list)  # finishing mean-reversion
    press_notes: list[str] = field(default_factory=list)       # press × build-up resistance


def _game_state_label(p: AdvancedTeamProfile) -> str:
    if p.xg_share_trailing >= 0.45:
        return "chaser (xG inflado yendo por detrás)"
    if p.xg_share_leading >= 0.38:
        return "front-runner (sigue creando con ventaja)"
    if p.xg_share_level >= 0.45:
        return "game-manager (vive en el empate)"
    return "equilibrado"


def _shape(p: AdvancedTeamProfile | None) -> AdvancedShape | None:
    if p is None:
        return None
    gs = _game_state_label(p)
    line = "línea alta" if p.line_height.mean >= 42 else "bloque profundo" if p.line_height.mean <= 33 else "línea media"
    direct = "directo" if p.directness.mean >= 0.25 else "posesión corta" if p.directness.mean <= 0.17 else "mixto"
    read = f"{line}, {direct}; {gs}"
    return AdvancedShape(
        team=p.team, field_tilt=p.field_tilt.mean, line_height=p.line_height.mean,
        directness=p.directness.mean, gk_goals_prevented=p.gk_goals_prevented_per_match.mean,
        game_state=gs, read=read)


def _annotated_board(
    props: list[PlayerPropProfile], team: str,
    referee: RefereeTendency | None, injuries: list[Injury],
    lineup: list[str] | None = None,
) -> tuple[list[PropCandidate], list[str]]:
    """Build a team's prop board; annotate (never drop) referee + injuries +
    confirmed-XI status. Injury flag takes priority over starter status."""
    board = [dataclasses.replace(c, team=team) for c in prop_board(props)]
    board = apply_referee_to_board(board, referee)   # preserves team/flag via replace
    inj_names = [i.name for i in injuries]
    out: list[PropCandidate] = []
    matched: set[str] = set()
    for c in board:
        m = match_name(c.player_name, inj_names)
        if m is not None:
            inj = next(i for i in injuries if i.name == m)
            until = f"hasta {inj.until}" if inj.until else "sin fecha de vuelta"
            out.append(dataclasses.replace(
                c, flag=f"⛔ LESIONADO: {inj.injury} ({until}) — NO apostar"))
            matched.add(m)
        elif lineup:
            in_xi = match_name(c.player_name, lineup) is not None
            out.append(dataclasses.replace(
                c, flag="✓ titular (XI confirmado)" if in_xi else "⚠ NO en el XI confirmado — ¿banquillo?"))
        else:
            out.append(c)
    notes = [f"{team}: lesionado {i.name} ({i.injury}) no está en el prop board "
             f"(sin perfil de stats) — revisar aparte"
             for i in injuries if i.name not in matched]
    return out, notes


def _blind_spots(
    ctx, home_props, away_props, wl: list[WeakLinkNote],
    home_sp: SetPieceIntel | None, away_sp: SetPieceIntel | None,
    home_lineup: list[str] | None = None, away_lineup: list[str] | None = None,
) -> list[str]:
    """What the AUTOMATION cannot capture — the human dig-list per fixture.

    Derived from the actual fixture state, not generic. This is the guard against
    automation hiding important detail: it makes the system surface its own blind
    spots rather than present a clean (false-confidence) answer.
    """
    spots: list[str] = []
    # 1 — lineups: the biggest unknown. If API-Football has the confirmed XI, the
    # blind spot is RESOLVED by data; otherwise it stays a human task.
    if home_lineup and away_lineup:
        spots.append("✓ XI CONFIRMADO (API-Football) para ambos — proyecciones sobre "
                     "titulares reales; verificar solo cambios de último minuto.")
    else:
        missing = [t for t, lu in ((ctx.home_team, home_lineup), (ctx.away_team, away_lineup)) if not lu]
        spots.append(f"CONFIRMAR XI a ~60' del inicio ({', '.join(missing)} sin XI confirmado aún) "
                     "— toda proyección asume alineación; una baja de última hora cambia el partido.")
    # 2 — new coach / regime: legacy data not predictive.
    for tid in (ctx.home_tid, ctx.away_tid):
        if tid is None:
            continue
        if tid.anchor and tid.anchor.note:
            spots.append(f"{tid.team_name}: {tid.anchor.note}")
        if tid.confidence == "scouting":
            spots.append(f"{tid.team_name}: identidad de SCOUTING (sin StatsBomb) — "
                         f"el planteamiento es juicio cualitativo, verificar en vídeo.")
    # 3 — set-piece routines: rich detail the stats don't encode.
    for name, sp in ((ctx.home_team, home_sp), (ctx.away_team, away_sp)):
        if sp is not None:
            spots.append(f"{name}: revisar rutinas de balón parado (penal: {sp.penalty_taker}; "
                         f"portero: {sp.keeper_shootout}) — no está en los números.")
    # 4 — WHY each weak-link is weak (the data says 'beaten', not why).
    for w in wl[:3]:
        spots.append(f"{w.team} — {w.player}: confirmar POR QUÉ ({w.reason}): "
                     f"¿lentitud, posición, contra quién? Ver vídeo antes de explotarlo.")
    # 5 — provenance: ESPN props have no xG → scorer less reliable.
    for name, props in ((ctx.home_team, home_props), (ctx.away_team, away_props)):
        if props and any(getattr(p, "source", "") == "espn" for p in props):
            spots.append(f"{name}: props por tasa de goles (ESPN, sin xG) — "
                         f"anytime-scorer menos fiable; usar como guía, no como cierre.")
    return spots


_SCRIPT_GAP = 0.35   # λ gap (≈ a third of a goal) to call a clear favorite


def game_script(
    home_team: str, away_team: str,
    home_tsv: TeamStyleVector | None, away_tsv: TeamStyleVector | None,
    home_tid: TacticalIdentity | None, away_tid: TacticalIdentity | None,
) -> GameScript | None:
    """Project the expected game-state trajectory from the TSV strength gap.

    λ via the same attack×defense mix as the predictor, computed inline from
    the TSV goal rates (no BettableProfile plumbing). Honest: missing/unusable
    TSV → None (no fabricated favorite).
    """
    if home_tsv is None or away_tsv is None:
        return None
    if not (home_tsv.goals_for_per_match.is_usable and away_tsv.goals_for_per_match.is_usable):
        return None
    lam_h = (home_tsv.goals_for_per_match.mean + away_tsv.goals_against_per_match.mean) / 2.0
    lam_a = (away_tsv.goals_for_per_match.mean + home_tsv.goals_against_per_match.mean) / 2.0
    gap = lam_h - lam_a
    if abs(gap) < _SCRIPT_GAP:
        return GameScript(
            favorite=None, gap=gap,
            read=("Sin favorito claro (λ ≈ parejo): el guion lo define el primer gol. "
                  "EN VIVO el primer gol invierte el favorito de córners — el que quede "
                  "por detrás dominará territorio y córners."),
            leans=["Córners: esperar al primer gol (el que persigue gana córners)",
                   "Sin lean de territorio pre-partido"])
    fav, dog = (home_team, away_team) if gap > 0 else (away_team, home_team)
    dog_tid = away_tid if gap > 0 else home_tid
    dog_low = dog_tid is not None and dog_tid.press_intensity == "low_block"
    leans = [f"Córners {fav} temprano (domina territorio)",
             f"Córners {dog} si va por detrás (el que persigue gana córners)"]
    if dog_low:
        leans.append(f"{dog} en bloque bajo → Under si {fav} no concreta + córners {fav}")
    return GameScript(
        favorite=fav, gap=gap,
        read=(f"{fav} favorito (λ {abs(gap):+.2f}). Guion probable: {fav} domina territorio, "
              f"{dog} persigue → córners de {dog} tarde si va por detrás. "
              f"Si {dog} marca primero (upset), {fav} vuelca → BTTS/Over."),
        leans=leans)


def duel_matchups(
    weak: list[WeakLinkNote], home_team: str, away_team: str,
    home_props: list[PlayerPropProfile], away_props: list[PlayerPropProfile],
    home_advanced: list[PlayerAdvancedProfile] | None,
    away_advanced: list[PlayerAdvancedProfile] | None,
) -> list[DuelMatchup]:
    """Cross each weak-link with the OPPOSING team's top threat of matching kind.

      pace/1v1 weak-link → opponent's top foul-drawer  → faltas/tarjetas/penal
      aerial   weak-link → opponent's top aerial winner → cabeza/córner

    Honest: no suitable threat (green/yellow, rate > 0) → no duel (never fabricated).
    """
    # A weak-link owned by the home team is exploited by the away attackers.
    pools = {
        home_team: (away_props, away_advanced or []),
        away_team: (home_props, home_advanced or []),
    }
    out: list[DuelMatchup] = []
    for w in weak:
        if w.team not in pools:
            continue
        att_props, att_adv = pools[w.team]
        att_team = away_team if w.team == home_team else home_team
        if w.kind == "pace/1v1":
            cands = [p for p in att_props
                     if p.confidence in ("green", "yellow") and p.fouls_drawn_per90.mean > 0]
            if not cands:
                continue
            a = max(cands, key=lambda p: p.fouls_drawn_per90.mean)
            out.append(DuelMatchup(
                attacker_team=att_team, attacker=a.player_name,
                defender_team=w.team, defender=w.player, defender_position=w.position,
                kind=w.kind, market="faltas/tarjetas/penal",
                read=(f"{a.player_name} provoca {a.fouls_drawn_per90.mean:.1f} faltas/90 "
                      f"vs {w.player} ({w.reason}) → faltas/tarjetas sobre {w.player}, "
                      f"penal si es en área. Confirmar que coinciden en zona.")))
        elif w.kind == "aerial":
            cands = [p for p in att_adv
                     if p.confidence in ("green", "yellow") and p.aerial_won_per90.mean > 0]
            if not cands:
                continue
            a = max(cands, key=lambda p: p.aerial_won_per90.mean)
            out.append(DuelMatchup(
                attacker_team=att_team, attacker=a.player_name,
                defender_team=w.team, defender=w.player, defender_position=w.position,
                kind=w.kind, market="cabeza/córner",
                read=(f"{a.player_name} gana {a.aerial_won_per90.mean:.1f} aéreos/90 "
                      f"vs {w.player} ({w.reason}) → remate de cabeza / córner.")))
    return out


def assemble_intel(
    dossier: MatchDossier,
    *,
    home_props: list[PlayerPropProfile] | None = None,
    away_props: list[PlayerPropProfile] | None = None,
    home_advanced: list[PlayerAdvancedProfile] | None = None,
    away_advanced: list[PlayerAdvancedProfile] | None = None,
    home_team_adv: AdvancedTeamProfile | None = None,
    away_team_adv: AdvancedTeamProfile | None = None,
    home_injuries: list[Injury] | None = None,
    away_injuries: list[Injury] | None = None,
    referee: RefereeTendency | None = None,
    home_props_recent: list[PlayerPropProfile] | None = None,
    away_props_recent: list[PlayerPropProfile] | None = None,
    home_lineup: list[str] | None = None,
    away_lineup: list[str] | None = None,
    home_press_resistance: PressResistance | None = None,
    away_press_resistance: PressResistance | None = None,
) -> MatchIntel:
    ctx = dossier.context
    home_props = home_props or []
    away_props = away_props or []
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []
    notes: list[str] = []

    hb, hn = _annotated_board(home_props, ctx.home_team, referee, home_injuries, home_lineup)
    ab, an = _annotated_board(away_props, ctx.away_team, referee, away_injuries, away_lineup)
    notes.extend(hn)
    notes.extend(an)
    hrb, _ = _annotated_board(home_props_recent or [], ctx.home_team, referee, home_injuries, home_lineup)
    arb, _ = _annotated_board(away_props_recent or [], ctx.away_team, referee, away_injuries, away_lineup)

    wl: list[WeakLinkNote] = []
    creators: list[Creator] = []
    for team, adv in ((ctx.home_team, home_advanced), (ctx.away_team, away_advanced)):
        if not adv:
            notes.append(f"{team}: sin métricas avanzadas de jugador (no en StatsBomb)")
            continue
        for w in weak_links(adv):
            wl.append(WeakLinkNote(team=team, player=w["player"], position=w["position"],
                                   reason=w["reason"], kind=w["kind"]))
        for p in sorted([x for x in adv if x.confidence in ("green", "yellow")],
                        key=lambda x: -x.sca_per90.mean)[:3]:
            creators.append(Creator(team=team, player=p.player_name, position=p.position,
                                    sca_per90=p.sca_per90.mean,
                                    prog_passes_per90=p.prog_passes_per90.mean))

    if referee is None:
        notes.append("Árbitro no asignado (FIFA designa por ronda) — board sin tilt de tarjetas")
    else:
        notes.append(referee_match_note(referee))

    duels = duel_matchups(wl, ctx.home_team, ctx.away_team,
                          home_props, away_props, home_advanced, away_advanced)
    gs = game_script(ctx.home_team, ctx.away_team, ctx.home_tsv, ctx.away_tsv,
                     ctx.home_tid, ctx.away_tid)
    regression = [n for tid in (ctx.home_tid, ctx.away_tid) if tid is not None
                  for n in (finishing_regression(tid),) if n is not None]
    # Press × build-up: home presses → away builds, and vice-versa.
    press_notes: list[str] = []
    if ctx.home_tid is not None:
        n = press_resistance_note(ctx.home_tid.press_intensity, away_press_resistance)
        if n is not None:
            press_notes.append(n)
    if ctx.away_tid is not None:
        n = press_resistance_note(ctx.away_tid.press_intensity, home_press_resistance)
        if n is not None:
            press_notes.append(n)

    home_sp = set_piece_for(ctx.home_team)
    away_sp = set_piece_for(ctx.away_team)
    blind = _blind_spots(ctx, home_props, away_props, wl, home_sp, away_sp,
                         home_lineup, away_lineup)

    return MatchIntel(
        dossier=dossier, home_injuries=home_injuries, away_injuries=away_injuries,
        home_prop_board=hb, away_prop_board=ab, weak_links=wl, creators=creators,
        home_shape=_shape(home_team_adv), away_shape=_shape(away_team_adv),
        provenance_notes=notes, blind_spots=blind,
        home_set_piece=home_sp, away_set_piece=away_sp,
        home_recent_board=hrb, away_recent_board=arb,
        home_lineup=home_lineup or [], away_lineup=away_lineup or [],
        duels=duels, game_script=gs, regression_notes=regression,
        press_notes=press_notes,
    )


# ── Render: existing dossier + the fused intel sections ──


def _shape_lines(s: AdvancedShape | None, name: str) -> list[str]:
    if s is None:
        return [f"- **{name}:** *sin forma avanzada (no en StatsBomb)*"]
    return [
        f"- **{name}:** {s.read}",
        f"  - field tilt {s.field_tilt:.0%} · línea x={s.line_height:.0f} · "
        f"{s.directness:.0%} pases largos · GK {s.gk_goals_prevented:+.2f}/p",
    ]


def render_intel_markdown(intel: MatchIntel) -> str:
    from bip.evaluation.tournaments.team_style_profiler.dossier_renderer import (
        render_markdown,
    )

    ctx = intel.dossier.context
    # §0 — the human dig-list goes FIRST: the briefing frames the evidence.
    head = [
        f"# Match Intel — {ctx.home_team} vs {ctx.away_team}", "",
        "## §0. Investigar tú (lo que el dato NO captura)", "",
        "*La automatización organiza la evidencia; estas son las preguntas que "
        "decide el analista. NO son conclusiones.*", "",
    ]
    for s in intel.blind_spots:
        head.append(f"- [ ] {s}")
    head += ["", "---", ""]

    out = head + [render_markdown(intel.dossier), "", "---", ""]
    if intel.game_script is not None or intel.regression_notes or intel.press_notes:
        out += ["## §3b. Guion de partido + regresión + presión (leans de goles)", ""]
        if intel.game_script is not None:
            gs = intel.game_script
            out.append(f"- {gs.read}")
            for ln in gs.leans:
                out.append(f"  - {ln}")
        for n in intel.regression_notes:
            out.append(f"- {n}")
        for n in intel.press_notes:
            out.append(f"- {n}")
        out += ["", "---", ""]
    out += ["## §4. Disponibilidad (Transfermarkt)", ""]
    for name, inj in ((ctx.home_team, intel.home_injuries), (ctx.away_team, intel.away_injuries)):
        if inj:
            out.append(f"**{name} — {len(inj)} baja(s):**")
            for i in inj:
                until = f"hasta {i.until}" if i.until else "sin fecha de vuelta"
                out.append(f"- ⛔ {i.name}: {i.injury} ({until})")
        else:
            out.append(f"**{name}:** sin bajas reportadas / no recolectado")
        out.append("")
    for name, xi in ((ctx.home_team, intel.home_lineup), (ctx.away_team, intel.away_lineup)):
        if xi:
            out.append(f"**XI confirmado {name} (API-Football):** {', '.join(xi)}")
            out.append("")

    out += ["---", "", "## §5. Forma avanzada de equipo (StatsBomb)", ""]
    out += _shape_lines(intel.home_shape, ctx.home_team)
    out += _shape_lines(intel.away_shape, ctx.away_team)

    out += ["", "---", "", "## §6. Inteligencia de jugador", "", "### Prop board (softest-first)"]
    for name, board in ((ctx.home_team, intel.home_prop_board), (ctx.away_team, intel.away_prop_board)):
        out.append(f"\n**{name}:**")
        if not board:
            out.append("- *(sin perfil de props)*")
        for c in board:
            soft = {1: "SOFT", 2: "med", 3: "hard"}.get(c.softness, "?")
            flag = f"  {c.flag}" if c.flag else ""
            out.append(f"- [{soft}] {c.market} — {c.player_name}: {c.stat}{flag}")

    if intel.home_recent_board or intel.away_recent_board:
        out += ["", "### Forma actual (ESPN, últimos partidos — recencia, sin xG)"]
        for name, board in ((ctx.home_team, intel.home_recent_board),
                            (ctx.away_team, intel.away_recent_board)):
            if not board:
                continue
            out.append(f"\n**{name}:**")
            for c in board[:6]:
                soft = {1: "SOFT", 2: "med", 3: "hard"}.get(c.softness, "?")
                flag = f"  {c.flag}" if c.flag else ""
                out.append(f"- [{soft}] {c.market} — {c.player_name}: {c.stat}{flag}")

    if intel.weak_links:
        out += ["", "### Eslabones débiles (a atacar)"]
        for w in intel.weak_links:
            out.append(f"- {w.team}: **{w.player}** ({w.position}) — {w.reason}")
    if intel.creators:
        out += ["", "### Creadores (SCA/90)"]
        for c in intel.creators:
            out.append(f"- {c.team}: **{c.player}** {c.sca_per90:.1f} SCA, {c.prog_passes_per90:.0f} prog-pass")
    if intel.duels:
        out += ["", "### Duelos clave (amenaza × eslabón débil — confirmar zona)"]
        for d in intel.duels:
            out.append(f"- [{d.market}] **{d.attacker}** ({d.attacker_team}) → "
                       f"{d.defender} ({d.defender_team}, {d.defender_position}): {d.read}")

    if intel.home_set_piece or intel.away_set_piece:
        out += ["", "---", "", "## §6b. Balón parado (catálogo curado)", ""]
        for name, sp in ((ctx.home_team, intel.home_set_piece), (ctx.away_team, intel.away_set_piece)):
            if sp is None:
                out.append(f"**{name}:** *no en el catálogo*")
                continue
            out.append(f"**{name}** — amenaza tier {sp.sp_threat_tier}")
            out.append(f"- penal #1: {sp.penalty_taker}")
            out.append(f"- portero (tanda): {sp.keeper_shootout}")
            out.append(f"- rutina: {sp.signature}")
            out.append(f"- explotar: {sp.exploitable}")
            if sp.note:
                out.append(f"- ⚠ {sp.note}")
        out.append("")

    if intel.provenance_notes:
        out += ["", "---", "", "## §7. Procedencia / cobertura", ""]
        for n in intel.provenance_notes:
            out.append(f"- {n}")

    return "\n".join(out)
