"""I/O layer for the Match Intel assembler — loads every cached layer, pulls
live injuries, looks up the referee, and orchestrates `build_match_intel`.

Keeps intel.py pure (no I/O). This is the single entry point behind the
one-command flow (an offline research script (not published)).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date
from pathlib import Path

from bip.evaluation.tournaments.team_style_profiler.advanced_metrics import AdvancedTeamProfile
from bip.evaluation.tournaments.team_style_profiler.intel import (
    Injury, MatchIntel, assemble_intel,
)
from bip.evaluation.tournaments.team_style_profiler.match_dossier import (
    DossierContext, generate_dossier,
)
from bip.evaluation.tournaments.team_style_profiler.player_advanced import PlayerAdvancedProfile
from bip.evaluation.tournaments.team_style_profiler.player_props import PlayerPropProfile
from bip.evaluation.tournaments.team_style_profiler.press_resistance import PressResistance
from bip.evaluation.tournaments.team_style_profiler.referee_tendencies import RefereeTendency
from bip.evaluation.tournaments.team_style_profiler.statsbomb_advanced import TeamStatsBombProfile
from bip.evaluation.tournaments.team_style_profiler.tactical_identity import tactical_identity_for
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat, TeamStyleVector

PROJECT = Path(__file__).resolve().parents[5]
CACHE = PROJECT / "data" / "cache" / "tsp"


def _api_football_key() -> str | None:
    env = PROJECT / ".env"
    if not env.exists():
        return None
    m = re.search(r"(?im)^\s*API_FOOTBALL_KEY\s*=\s*(.+)$", env.read_text())
    return m.group(1).strip().strip('"').strip("'") if m else None


def _slug(n: str) -> str:
    return n.replace(" ", "_").replace("'", "").replace("ô", "o").lower()


def _ds(m) -> DistributionStat:
    m = float(m or 0.0)
    return DistributionStat(mean=m, ci_low=m, ci_high=m, n=8)


def _load(sub: str, slug: str) -> dict | None:
    f = CACHE / sub / f"{slug}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _props_from(d: dict | None) -> list[PlayerPropProfile]:
    if not d:
        return []
    raw = d.get("source", "statsbomb")
    src = ("espn" if "espn" in raw
           else "api_football" if "api_football" in raw
           else "statsbomb")
    out = []
    for i, p in enumerate(d["players"]):
        out.append(PlayerPropProfile(
            player_id=i, player_name=p["name"], team=d["team"], position=p.get("position", "?"),
            n_matches=p["n_matches"], minutes_total=p["n_matches"] * 90.0, confidence=p["confidence"],
            shots_per90=_ds(p.get("shots_per90")), shots_on_target_per90=_ds(p.get("sot_per90")),
            xg_per90=_ds(p.get("xg_per90")), goals_per90=_ds(p.get("goals_per90")),
            fouls_committed_per90=_ds(p.get("fouls_per90")),
            fouls_drawn_per90=_ds(p.get("fouls_drawn_per90")),
            yellow_cards_per90=_ds(p.get("yellows_per90")),
            key_passes_per90=_ds(0), assists_per90=_ds(p.get("assists_per90")),
            penalties_taken=p.get("penalties_taken", 0), source=src))
    return out


def load_props(slug: str) -> list[PlayerPropProfile]:
    # Prefiere props de API-Football (selección actual, #2) sobre StatsBomb-torneo.
    return _props_from(_load("player_props_af", slug) or _load("player_props", slug))


def load_recent_props(slug: str) -> list[PlayerPropProfile]:
    return _props_from(_load("player_props_espn", slug))


def load_advanced(slug: str) -> list[PlayerAdvancedProfile] | None:
    d = _load("player_advanced", slug)
    if not d:
        return None
    out = []
    for i, p in enumerate(d["players"]):
        awr = max(p.get("aerial_win_rate", 0.5), 0.01)
        aw = p.get("aerial_won_per90", 0)
        out.append(PlayerAdvancedProfile(
            player_id=i, player_name=p["name"], team=d["team"], position=p.get("position", "?"),
            n_matches=p["n_matches"], confidence=p["confidence"],
            prog_passes_per90=_ds(p.get("prog_passes_per90")), prog_carries_per90=_ds(p.get("prog_carries_per90")),
            sca_per90=_ds(p.get("sca_per90")), gca_per90=_ds(p.get("gca_per90")),
            tackles_per90=_ds(p.get("tackles_per90")), interceptions_per90=_ds(p.get("interceptions_per90")),
            clearances_per90=_ds(0), recoveries_per90=_ds(0),
            dribbled_past_per90=_ds(p.get("dribbled_past_per90")),
            aerial_won_per90=_ds(aw), aerial_lost_per90=_ds(max(0.0, aw / awr - aw))))
    return out


def load_team_advanced(slug: str, team: str) -> AdvancedTeamProfile | None:
    d = _load("advanced_metrics", slug)
    if not d:
        return None
    return AdvancedTeamProfile(
        team=team, n_matches=d["n_matches"], field_tilt=_ds(d["field_tilt"]),
        gk_goals_prevented_per_match=_ds(d["gk_goals_prevented_per_match"]),
        line_height=_ds(d.get("line_height")), directness=_ds(d.get("directness")),
        xg_share_leading=d["xg_share_leading"], xg_share_level=d["xg_share_level"],
        xg_share_trailing=d["xg_share_trailing"])


def load_cached_injuries(slug: str) -> list[Injury]:
    d = _load("availability", slug)
    if not d:
        return []
    return [Injury(name=e["name"], injury=e.get("injury") or "lesión",
                   until=e.get("until"), ongoing=e.get("ongoing", False))
            for e in d.get("injured", [])]


def load_tsv(slug: str) -> TeamStyleVector | None:
    f = CACHE / "profiles" / f"{slug}_tsv.json"
    return TeamStyleVector.model_validate_json(f.read_text()) if f.exists() else None


def load_tid(slug: str, team: str):
    f = CACHE / "profiles_statsbomb" / f"{slug}_sb.json"
    prof = TeamStatsBombProfile.model_validate_json(f.read_text()) if f.exists() else None
    return tactical_identity_for(team, prof)


def referee_by_name(name: str | None) -> RefereeTendency | None:
    if not name:
        return None
    # #7: prefiere la tabla de API-Football (n grande, mismo formato de nombre que el
    # fixture WC) y cae a la de StatsBomb por árbitro.
    af = CACHE / "referee_tendencies_af.json"
    sb = CACHE / "referee_tendencies.json"
    af_tbl = json.loads(af.read_text()) if af.exists() else {}
    sb_tbl = json.loads(sb.read_text()) if sb.exists() else {}
    if not af_tbl and not sb_tbl:
        return None

    def _hit(tbl: dict) -> str | None:
        return next((k for k in tbl
                     if name.lower() in k.lower() or k.lower() in name.lower()), None)

    key = _hit(af_tbl)
    table = af_tbl
    if key is None:
        key = _hit(sb_tbl)
        table = sb_tbl
    if key is None:
        return None
    r = table[key]
    return RefereeTendency(
        referee_name=key, country=r.get("country", ""), n_matches=r["n_matches"],
        confidence=r["confidence"], strictness=r["strictness"],
        cards_per_match=_ds(r["cards_per_match"]), yellows_per_match=_ds(r["yellows_per_match"]),
        reds_per_match=_ds(r["reds_per_match"]), fouls_per_match=_ds(r["fouls_per_match"]),
        penalties_per_match=_ds(r["penalties_per_match"]))


def load_press_resistance(team_name: str) -> PressResistance | None:
    """Look up a team's press-resistance profile (fuzzy name match)."""
    f = CACHE / "press_resistance.json"
    if not f.exists():
        return None
    table = json.loads(f.read_text())
    key = next((k for k in table
                if team_name.lower() in k.lower() or k.lower() in team_name.lower()), None)
    if key is None:
        return None
    r = table[key]
    return PressResistance(
        team=key, n_matches=r["n_matches"], confidence=r["confidence"],
        passes_under_pressure_per_match=_ds(r["passes_up_per_match"]),
        completion_under_pressure=_ds(r["completion_under_pressure"]),
        resistance=r["resistance"])


def fetch_live_injuries(team: str, today: date) -> tuple[list[Injury] | None, str | None]:
    """Live Transfermarkt pull. Returns (injuries, None) or (None, error-note)
    so the caller can fall back to cached availability without losing the team."""
    from bip.evaluation.tournaments.team_style_profiler.transfermarkt_scrape import (
        current_injury_status, fetch_injuries_html, fetch_squad_html,
        parse_injuries, parse_squad, resolve_kader_path,
    )
    try:
        kader = resolve_kader_path(team)
        if not kader:
            return None, f"{team}: no se resolvió el squad en Transfermarkt (usando caché)"
        squad = parse_squad(fetch_squad_html(kader))
        time.sleep(0.3)
        out: list[Injury] = []
        for p in squad:
            try:
                recs = parse_injuries(fetch_injuries_html(p.profile_path))
            except Exception:
                recs = []
            time.sleep(0.3)
            st = current_injury_status(recs, today)
            if st.injured:
                out.append(Injury(name=p.name, injury=st.injury or "lesión",
                                  until=st.until.isoformat() if st.until else None,
                                  ongoing=st.ongoing))
        return out, None
    except Exception as e:
        return None, f"{team}: fallo Transfermarkt ({type(e).__name__}) — usando caché"


# Display/corpus name -> the name API-Football resolves a national team by.
# Verified live against /teams?name= (2026-05-31): without these the search
# returns nothing → no fixture → no XI/referee. (Mexico/Panama/Jordan resolve
# fine via the j[0] fallback despite API's national:false quirk — not aliased.)
_API_FOOTBALL_TEAM_ALIAS = {
    "United States": "USA",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Turkey": "Türkiye",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
}


async def _confirmed_lineup_async(home: str, away: str, season: int):
    from bip.sports.football.client import ApiFootballClient
    key = _api_football_key()
    if not key:
        return None
    async with ApiFootballClient(api_key=key) as c:
        async def team_id(name: str):
            q = _API_FOOTBALL_TEAM_ALIAS.get(name, name)
            j = (await c._client.get("/teams", params={"name": q})).json().get("response", [])
            nat = [t for t in j if t.get("team", {}).get("national")]
            pick = nat[0] if nat else (j[0] if j else None)
            return pick["team"]["id"] if pick else None

        hid, aid = await team_id(home), await team_id(away)
        if not hid or not aid:
            return None
        fx = (await c._client.get("/fixtures", params={"team": hid, "season": season})).json().get("response", [])
        match = next((f for f in fx
                      if {f["teams"]["home"]["id"], f["teams"]["away"]["id"]} == {hid, aid}), None)
        if match is None:
            return None
        # Referee comes free on the same fixture object (None until assigned).
        referee = (match["fixture"].get("referee") or "").strip() or None
        ln = (await c._client.get(
            "/fixtures/lineups", params={"fixture": match["fixture"]["id"]})).json().get("response", [])
        home_xi, away_xi = [], []
        for t in ln:
            names = [p["player"]["name"] for p in (t.get("startXI") or []) if p.get("player")]
            if t["team"]["id"] == hid:
                home_xi = names
            elif t["team"]["id"] == aid:
                away_xi = names
        if not (home_xi or away_xi or referee):
            return None
        return (home_xi, away_xi, referee)


def fetch_confirmed_lineup(
    home: str, away: str, season: int
) -> tuple[list[str], list[str], str | None] | None:
    """Confirmed XI + referee from API-Football (lights up ~40' pre-KO / for played
    fixtures). Returns (home_starters, away_starters, referee_name) or None
    (pre-scheduled / no data / error). Referee is None until FIFA assigns one."""
    try:
        return asyncio.run(_confirmed_lineup_async(home, away, season))
    except Exception:
        return None


def build_match_intel(
    home: str, away: str, *,
    referee_name: str | None = None,
    pull_injuries: bool = True,
    fixture_date: date | None = None,
) -> MatchIntel:
    fixture_date = fixture_date or date(2026, 6, 11)
    today = date.today()
    hs, as_ = _slug(home), _slug(away)

    ctx = DossierContext(
        home_team=home, away_team=away, home_tsv=load_tsv(hs), away_tsv=load_tsv(as_),
        tournament_slug="world_cup_2026", fixture_date=fixture_date,
        home_tid=load_tid(hs, home), away_tid=load_tid(as_, away))
    dossier = generate_dossier(ctx)

    extra_notes: list[str] = []
    if pull_injuries:
        hi, hn = fetch_live_injuries(home, today)
        ai, an = fetch_live_injuries(away, today)
        if hi is None:
            extra_notes.append(hn or "")
            hi = load_cached_injuries(hs)
        if ai is None:
            extra_notes.append(an or "")
            ai = load_cached_injuries(as_)
    else:
        hi, ai = load_cached_injuries(hs), load_cached_injuries(as_)

    lineup = fetch_confirmed_lineup(home, away, fixture_date.year)
    home_xi, away_xi, ref_from_api = lineup if lineup else (None, None, None)
    # Explicit --referee wins; else use the one API-Football assigned (if any).
    referee = referee_by_name(referee_name or ref_from_api)

    intel = assemble_intel(
        dossier,
        home_props=load_props(hs), away_props=load_props(as_),
        home_props_recent=load_recent_props(hs), away_props_recent=load_recent_props(as_),
        home_advanced=load_advanced(hs), away_advanced=load_advanced(as_),
        home_team_adv=load_team_advanced(hs, home), away_team_adv=load_team_advanced(as_, away),
        home_injuries=hi, away_injuries=ai,
        referee=referee,
        home_lineup=home_xi, away_lineup=away_xi,
        home_press_resistance=load_press_resistance(home),
        away_press_resistance=load_press_resistance(away))
    intel.provenance_notes.extend(n for n in extra_notes if n)
    return intel
