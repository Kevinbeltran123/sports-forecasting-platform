"""Transfermarkt scraper — squad + injury/availability data.

Transfermarkt is NOT Cloudflare-blocked (unlike FBref): it serves real HTML to
a curl_cffi browser-TLS-impersonated request (verified 2026-05-30). This is the
working source for the biggest missing analysis input — current INJURIES /
AVAILABILITY for WC2026 squads (a key player out reshapes both the team dossier
and the player-prop board).

Pure parsers (offline-testable, bs4) are separated from the curl_cffi fetchers
so the suite tests parsing without network.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime

from bs4 import BeautifulSoup

_BASE = "https://www.transfermarkt.com"
_HEADERS = {"Accept-Language": "en-US,en;q=0.9"}
_NO_RETURN = {"", "-", "?", "unknown", "ausstehend"}
_SPIELER_RE = re.compile(r"/spieler/(\d+)")
_VEREIN_RE = re.compile(r"/([a-z0-9-]+)/(?:startseite|kader)/verein/(\d+)")
# Exclude youth / women / futsal sides so we resolve the SENIOR national team.
_NOT_SENIOR = ("u23", "u21", "u20", "u19", "u18", "u17", "u16", "frauen",
               "women", "futsal", "beach", "-fc", "olympic")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(t for t in s.lower().replace("-", " ").split() if t != "and")


@dataclass(frozen=True)
class SquadPlayer:
    name: str
    tm_id: str
    profile_path: str  # e.g. /erling-haaland/profil/spieler/418560


@dataclass(frozen=True)
class InjuryRecord:
    season: str
    injury: str
    from_date: str   # raw DD/MM/YYYY as shown
    until: str        # raw; may be "-"/"?" when ongoing
    days: str
    games_missed: str


@dataclass(frozen=True)
class InjuryStatus:
    injured: bool
    injury: str | None
    until: date | None
    ongoing: bool          # injury with no listed return date
    last_injury: str | None


def parse_squad(html: str) -> list[SquadPlayer]:
    """Players from a national-team /kader/ page (name + Transfermarkt id)."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.select_one("table.items")
    if tbl is None:
        return []
    out: list[SquadPlayer] = []
    seen: set[str] = set()
    for tr in tbl.select("tbody > tr"):
        a = tr.select_one("a[href*='/profil/spieler/']")
        if a is None:
            continue
        m = _SPIELER_RE.search(a.get("href", ""))
        if not m:
            continue
        tm_id = m.group(1)
        if tm_id in seen:
            continue
        seen.add(tm_id)
        out.append(SquadPlayer(name=a.get_text(strip=True), tm_id=tm_id,
                               profile_path=a["href"]))
    return out


def parse_injuries(html: str) -> list[InjuryRecord]:
    """Injury history rows (newest first) from a /verletzungen/ page."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.select_one("table.items")
    if tbl is None:
        return []
    out: list[InjuryRecord] = []
    for tr in tbl.select("tbody tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 6:
            continue
        out.append(InjuryRecord(
            season=cells[0], injury=cells[1], from_date=cells[2],
            until=cells[3], days=cells[4], games_missed=cells[5]))
    return out


def parse_search_for_team(html: str, team_name: str) -> str | None:
    """From a Transfermarkt quick-search page, return the SENIOR national team's
    kader path (/{slug}/kader/verein/{id}), skipping youth/women sides."""
    soup = BeautifulSoup(html, "html.parser")
    target = _norm(team_name)
    for a in soup.select("a[href*='/verein/']"):
        m = _VEREIN_RE.search(a.get("href", ""))
        if not m:
            continue
        slug, tid = m.group(1), m.group(2)
        if any(x in slug for x in _NOT_SENIOR):
            continue
        text = _norm(a.get_text(strip=True))
        if text and (text == target or target in text or text in target):
            return f"/{slug}/kader/verein/{tid}"
    return None


def _parse_dmy(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%d/%m/%Y").date()
    except (ValueError, AttributeError):
        return None


def current_injury_status(records: list[InjuryRecord], today: date) -> InjuryStatus:
    """Is the player currently injured? Uses the most-recent injury row.

    Ongoing (no return date) → injured. Otherwise injured iff `until` >= today.
    """
    if not records:
        return InjuryStatus(False, None, None, False, None)
    latest = records[0]
    until_raw = latest.until.strip().lower()
    if until_raw in _NO_RETURN:
        return InjuryStatus(True, latest.injury, None, True, latest.injury)
    until = _parse_dmy(latest.until)
    if until is not None and until >= today:
        return InjuryStatus(True, latest.injury, until, False, latest.injury)
    return InjuryStatus(False, None, None, False, latest.injury)


# ── Network (curl_cffi browser-TLS impersonation; lazy import) ──


def _get(url: str) -> str:
    from curl_cffi import requests as creq

    r = creq.get(url, impersonate="chrome", headers=_HEADERS, timeout=25)
    r.raise_for_status()
    return r.text


def fetch_squad_html(kader_path: str) -> str:
    """kader_path e.g. /norwegen/kader/verein/3440 (national-team squad page)."""
    return _get(f"{_BASE}{kader_path}")


def fetch_injuries_html(profile_path: str) -> str:
    inj_path = profile_path.replace("/profil/spieler/", "/verletzungen/spieler/")
    return _get(f"{_BASE}{inj_path}")


def resolve_kader_path(team_name: str) -> str | None:
    """Auto-resolve any national team's squad page via TM quick-search."""
    q = team_name.replace(" ", "+")
    html = _get(f"{_BASE}/schnellsuche/ergebnis/schnellsuche?query={q}")
    return parse_search_for_team(html, team_name)
