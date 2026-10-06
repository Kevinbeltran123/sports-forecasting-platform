"""Curated set-piece intel — the rich qualitative detail the automated stats
pipeline cannot compute, structured from internal research notes so it
can be surfaced per fixture (not left buried in prose).

These are betting-relevant facts an elite analyst keeps in their head: who takes
penalties, the keeper's shootout pedigree, the team's set-piece threat tier, the
signature routine, and the exploitable weakness. Hand-curated (cross-source
verified per the catalog) — NOT auto-derived, by design.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SetPieceIntel:
    sp_threat_tier: str       # S / A / B / C / D
    penalty_taker: str        # #1 (+ caveat)
    keeper_shootout: str      # keeper + shootout pedigree
    signature: str            # main attacking routine
    exploitable: str          # their set-piece weakness to attack
    note: str = ""            # coach/availability caveat


# Keyed by TSV canonical team name. Source: SET_PIECE_CATALOG_WC2026.md (2026-05-28).
SET_PIECE_INTEL: dict[str, SetPieceIntel] = {
    "USA": SetPieceIntel(
        "S", "Pulisic", "Matt Turner — solid, no mythology",
        "Gianni Vio 'stack-and-shed' near-post — every corner is code-designed",
        "2nd balls post-corner-clear (WC22 NED pattern); CB pace",
        note="Vio retained by Pochettino — biggest set-piece edge of the tournament."),
    "Germany": SetPieceIntel(
        "S", "Havertz (#1 unclear post-Kroos)", "ter Stegen — no special shootout record",
        "Mittelstädt→Kimmich header-back→Wirtz volley (vs ESP EURO24); Buttgereit double-investment",
        "no clear penalty hierarchy post-Kroos (shootout mental risk); tall CBs vs low cross",
        note="Buttgereit+Schreuder reserved for SP — explicit priority signal."),
    "Portugal": SetPieceIntel(
        "A", "Ronaldo (41 — measurable decline; Bruno 90.5% is the sharper option)",
        "Diogo Costa — ELITE (3 saves one shootout, EURO24 record)",
        "Bruno Fernandes inswinger + MacPhee routines (ex-Villa top-3 PL SP)",
        "slow defensive recovery on 2nd balls (lost to FRA EURO24 QF)",
        note="BACK Portugal in any shootout (Diogo Costa)."),
    "England": SetPieceIntel(
        "A", "Kane", "Pickford — 5/20 shootout saves, 3 of 4 won; bottle with notes",
        "Barry deception routines + long throws (inswinger Trippier)",
        "NO elite direct-FK threat after Trent snub; zonal corner D leaves 2nd post",
        note="BACK England shootout above historic average (Pickford)."),
    "Spain": SetPieceIntel(
        "B", "Oyarzabal", "Unai Simón — strong central reader (Panenkas fail)",
        "Carvajal near-post blind-side flick (won vs CRO EURO24); 34% short corners",
        "pure-zonal corner D → runner to far post",
    ),
    "Argentina": SetPieceIntel(
        "B", "Messi (~81% but missed Copa24 Panenka + WC22 POL)",
        "Dibu Martínez — undefeated in Argentina shootouts",
        "near-post flick → far-post finish (WC22/Copa24 signature)",
        "slow short-corner defending (Colombia exploited Copa24 final)",
        note="BACK Argentina shootout (Dibu)."),
    "France": SetPieceIntel(
        "C", "Mbappé (~80%)", "Maignan — no shootout record, weak vs Panenka",
        "Griezmann inswing left → Upamecano/Saliba back-post",
        "pure-zonal corners overload near post → 2nd post free (ESP EURO24 SF)",
    ),
    "Croatia": SetPieceIntel(
        "B", "Modrić (~85%)", "Livaković — ELITE (3 saves vs JPN WC22)",
        "Modrić far-post inswinger → Gvardiol header-back; 22% set-piece xG",
        "ABSOLUTE Modrić dependency at 40 — ~60% SP threat gone if subbed/injured",
        note="LIVE: fade CRO SP threat once Modrić is off. BACK shootout (Livaković)."),
    "Morocco": SetPieceIntel(
        "C", "Hakimi (Panenka master vs ESP WC22)", "Bono — ELITE (3 saves vs ESP WC22)",
        "Hakimi inswinger → Aguerd far-post (era Regragui — in re-design)",
        "SP offense historically under-developed (reactive, not proactive)",
        note="Ouahbi replaced Regragui Mar 2026 — routine library NOT predictive; BACK shootout (Bono)."),
    "Uruguay": SetPieceIntel(
        "D", "Valverde", "Rochet — central-penalty save tendency",
        "minimal — Bielsa does NOT prioritise SP (build-through-play)",
        "Bielsa man-marking on D-corners exposed to blocks/picks",
        note="UNDER corner-goals URU has EV — elite aerial talent (Araújo/Giménez) WASTED."),
    "Mexico": SetPieceIntel(
        "C", "Raúl Jiménez (~95%, 42/44)", "Malagón/Rangel — no Ochoa pedigree",
        "inswinger near→far (Raúl Jiménez target)",
        "aerial vulnerability vs physical European sides on 2nd balls",
    ),
    "Netherlands": SetPieceIntel(
        "D", "Memphis (Gakpo confusion unresolved)", "Verbruggen — WEAKEST top-UEFA shootout keeper",
        "inswinger → Van Dijk back-post (VVD elite aerial target)",
        "deep-zonal corners cede 1st contact (2 corner goals conceded EURO24)",
        note="Most dead-ball-exploitable top-UEFA side; FADE NED shootout."),
    "Belgium": SetPieceIntel(
        "D", "De Bruyne (Lukaku erratic)", "Courtois — ~17% lifetime save rate",
        "KDB inswinger → Lukaku near-post flick",
        "aging CB pair + Courtois punching > catching; chained counter-corners",
        note="KDB delivery erratic when not fully fit."),
    "Japan": SetPieceIntel(
        "C", "Ueda", "Zion Suzuki — emerging, no shootout record",
        "far-post inswinger → CB/DM header (Itakura); Kubo direct FK",
        "small CB pair bullied aerially (Iran Asian Cup); zonal vs near-post traffic",
    ),
    "Senegal": SetPieceIntel(
        "C", "Mané (high volume, mid conversion — exploitable if keeper has homework)",
        "Édouard Mendy — elite shootout reputation",
        "Mané far-post inswinger → Koulibaly (if fit)",
        "aging spine + uncertain CB pairing — if Koulibaly out, loses primary target",
    ),
}


def set_piece_for(team: str) -> SetPieceIntel | None:
    return SET_PIECE_INTEL.get(team)
