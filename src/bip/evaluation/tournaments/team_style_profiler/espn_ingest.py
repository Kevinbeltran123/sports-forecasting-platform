"""ESPN public-API ingestion → player-prop counts.

ESPN's open JSON API (site.api.espn.com) is the one international source that
works without Cloudflare/tokens and covers World Cup QUALIFIERS + friendlies —
the current-form data for the 8 WC2026 teams absent from StatsBomb open data
(Bosnia, Curaçao, Haiti, Iraq, Jordan, New Zealand, Norway, Uzbekistan).

Per-match player stats available: totalShots, shotsOnTarget, totalGoals,
foulsCommitted, yellowCards, goalAssists, saves, etc. NO xG, and no reliable
per-player minutes (keyEvents omit athlete attribution) → each appearance is
treated as 90' (conservative for subs) and anytime-scorer falls back to the
goals rate. Profiles carry source="espn" so consumers see the caveats.

This module is the PURE parser (offline-testable). Network fetching lives in
an offline research script (not published).
"""
from __future__ import annotations

# ESPN per-player stat name -> our PlayerPropProfile metric key.
ESPN_STAT_MAP = {
    "totalShots": "shots",
    "shotsOnTarget": "sot",
    "totalGoals": "goals",
    "foulsCommitted": "fouls",
    "yellowCards": "yellows",
    "goalAssists": "assists",
}


def match_player_counts(
    summary: dict, team_id: str
) -> tuple[dict[int, dict[str, float]], dict[int, tuple[str, str, str]], dict[int, float]]:
    """Extract one match's per-player (counts, meta, minutes) for `team_id`.

    Returns the same shape build_player_profiles expects. minutes = 90 for every
    player who appeared (ESPN gives no usable minutes). Players who did not
    appear (no appearance, not starter, not subbed in) are skipped.
    """
    counts: dict[int, dict[str, float]] = {}
    meta: dict[int, tuple[str, str, str]] = {}
    minutes: dict[int, float] = {}

    for team in summary.get("rosters", []):
        if str((team.get("team") or {}).get("id")) != str(team_id):
            continue
        team_name = (team.get("team") or {}).get("displayName", "")
        for entry in team.get("roster", []):
            ath = entry.get("athlete") or {}
            pid_raw = ath.get("id")
            if pid_raw is None:
                continue
            stat = {s.get("name"): float(s.get("value") or 0.0) for s in entry.get("stats", [])}
            appeared = (
                stat.get("appearances", 0) >= 1
                or entry.get("starter")
                or entry.get("subbedIn")
            )
            if not appeared:
                continue
            pid = int(pid_raw)
            c = {our: stat.get(espn, 0.0) for espn, our in ESPN_STAT_MAP.items()}
            c["xg"] = 0.0          # ESPN has no xG
            c["key_passes"] = 0.0  # ESPN has no key passes
            c["pens"] = 0.0        # penalty-taker not separable from this feed
            counts[pid] = c
            pos = (entry.get("position") or {}).get("abbreviation", "?")
            meta[pid] = (ath.get("displayName", str(pid)), team_name, pos)
            minutes[pid] = 90.0
    return counts, meta, minutes
