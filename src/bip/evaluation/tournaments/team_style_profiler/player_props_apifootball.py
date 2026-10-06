"""Parse API-Football /fixtures/players + /fixtures/events into the (counts, meta,
minutes) tuples that player_props.build_player_profiles consumes.

Fuente del prop board desde los fixtures RECIENTES de la selección bajo el DT actual
(reemplaza el StatsBomb-torneo viejo — mejora #2). Reusa toda la agregación
(build_player_profiles / prop_board); aquí solo se traduce el JSON de API-Football.

Calidad observada (probe 2026-06-01):
- minutos / posición / tiros / SoT / key_passes / asistencias: presentes.
- fouls.committed a veces None → 0 (no fabricar).
- TARJETAS desde /fixtures/events (player.id + minuto, filtrar por team.id): más fiable
  que players.cards. Un segundo amarillo cuenta como amarilla mostrada.
- xG no existe para internacionales → 0 (p_anytime_scorer cae a la tasa de goles).
"""
from __future__ import annotations


def _int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def parse_fixture(
    players_resp: list[dict],
    events_resp: list[dict],
    team_id: int,
) -> tuple[dict[int, dict[str, float]], dict[int, tuple[str, str, str]], dict[int, float]]:
    """Un fixture -> (counts, meta, minutes) solo para los jugadores de `team_id`.

    counts: {player_id: {metric: float}} con las claves de player_props._METRICS + 'pens'.
    meta:   {player_id: (name, team, position)}
    minutes:{player_id: float}  (excluye DNP / 0')
    """
    counts: dict[int, dict[str, float]] = {}
    meta: dict[int, tuple[str, str, str]] = {}
    minutes: dict[int, float] = {}

    block = next((b for b in players_resp if (b.get("team") or {}).get("id") == team_id), None)
    if block is None:
        return counts, meta, minutes
    team_name = (block.get("team") or {}).get("name", "")

    for p in block.get("players", []):
        pid = (p.get("player") or {}).get("id")
        if pid is None:
            continue
        st = (p.get("statistics") or [{}])[0] or {}
        games = st.get("games") or {}
        mins = _int(games.get("minutes"))
        if mins <= 0:  # no jugó
            continue
        shots = st.get("shots") or {}
        goals = st.get("goals") or {}
        fouls = st.get("fouls") or {}
        passes = st.get("passes") or {}
        pen = st.get("penalty") or {}
        counts[pid] = {
            "shots": float(_int(shots.get("total"))),
            "sot": float(_int(shots.get("on"))),
            "xg": 0.0,  # no hay xG en internacionales
            "goals": float(_int(goals.get("total"))),
            "fouls": float(_int(fouls.get("committed"))),
            "fouls_won": float(_int(fouls.get("drawn"))),
            "yellows": 0.0,  # se rellena desde /events
            "key_passes": float(_int(passes.get("key"))),
            "assists": float(_int(goals.get("assists"))),
            "pens": float(_int(pen.get("scored")) + _int(pen.get("missed"))),  # penaltis lanzados
        }
        meta[pid] = ((p.get("player") or {}).get("name") or str(pid), team_name,
                     games.get("position") or "?")
        minutes[pid] = float(mins)

    # Tarjetas desde eventos (más fiable que players.cards). Solo nuestro equipo.
    for e in events_resp:
        if e.get("type") != "Card":
            continue
        if (e.get("team") or {}).get("id") != team_id:
            continue
        pid = (e.get("player") or {}).get("id")
        if pid is None or pid not in counts:
            continue
        if "yellow" in (e.get("detail") or "").lower():
            counts[pid]["yellows"] += 1.0

    return counts, meta, minutes
