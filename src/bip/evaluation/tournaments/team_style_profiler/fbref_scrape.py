"""FBref scraper via FlareSolverr — the one source with real xG + minutes.

FBref is Cloudflare-locked to plain HTTP / curl_cffi / headless browsers, but a
FlareSolverr instance (dockerised headed browser that solves the challenge) at
http://localhost:8191 returns the full HTML. Verified 2026-05-30.

Two FBref quirks handled here:
  1. Data tables are wrapped in HTML comments → strip `<!--`/`-->` before parsing.
  2. The competition default points at the UPCOMING edition (no data) → always
     use the season-specific URL, e.g.
     /en/comps/1/2022/stats/2022-World-Cup-Stats   (not /en/comps/1/stats/...).

Pure parser (offline-testable) separated from the FlareSolverr fetch. FBref
gives season-aggregate per-90 rates directly (incl. xG) — richer than ESPN
(no xG) and StatsBomb (tournament-only), and the only xG source for non-StatsBomb
teams once scraped locally.
"""
from __future__ import annotations

import os

from bs4 import BeautifulSoup

FLARESOLVERR_URL = os.environ.get("FLARESOLVERR_URL", "http://localhost:8191/v1")


def parse_player_table(html: str, table_id: str, fields: list[str]) -> list[dict[str, str]]:
    """Extract `fields` (FBref data-stat names) per player row from `table_id`.

    Strips FBref's comment-wrapping first. Skips repeated header rows and rows
    without a player. Returns raw string values (caller coerces).
    """
    clean = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(clean, "lxml")
    table = soup.find("table", id=table_id)
    if table is None:
        return []
    out: list[dict[str, str]] = []
    for tr in table.select("tbody tr"):
        if "thead" in (tr.get("class") or []):
            continue
        player_cell = tr.find(attrs={"data-stat": "player"})
        if player_cell is None or not player_cell.get_text(strip=True):
            continue
        row: dict[str, str] = {}
        for f in fields:
            cell = tr.find(attrs={"data-stat": f})
            row[f] = cell.get_text(strip=True) if cell else ""
        out.append(row)
    return out


def season_stats_url(comp_id: int, year: str, comp_slug: str, kind: str = "stats") -> str:
    """Season-specific FBref URL (the one that actually carries data).

    e.g. season_stats_url(1, "2022", "World-Cup", "shooting") ->
    https://fbref.com/en/comps/1/2022/shooting/2022-World-Cup-Stats
    """
    return (f"https://fbref.com/en/comps/{comp_id}/{year}/{kind}/"
            f"{year}-{comp_slug}-Stats")


# ── FlareSolverr fetch (network; lazy import) ──


def fetch(url: str, max_timeout: int = 60000) -> str:
    """Fetch `url` through FlareSolverr, returning the solved HTML."""
    import requests

    r = requests.post(
        FLARESOLVERR_URL,
        json={"cmd": "request.get", "url": url, "maxTimeout": max_timeout},
        timeout=max_timeout / 1000 + 30,
    )
    j = r.json()
    if j.get("status") != "ok":
        raise RuntimeError(f"FlareSolverr error: {j.get('message')}")
    sol = j.get("solution", {})
    if sol.get("status") != 200:
        raise RuntimeError(f"FBref HTTP {sol.get('status')} for {url}")
    return sol.get("response", "")
