"""Log append-only de los reads MARCADOS por el analista, para medir hit-rate + ROI
por tipo de read en el tiempo.

POR QUÉ: la validez del flujo está cuellobotellada en MEDICIÓN, no en datos. Con n chico
y dos markdown a mano (match log, picks ledger), un buen tramo es
indistinguible de varianza y no se puede cortar el hit-rate por read-type. Este log estructurado
permite SLICE por read_level (team/prop) × tier (secundario/eficiente) × régimen.

QUÉ ES Y QUÉ NO: RECORDA lo que el analista marcó como evento más probable; NO decide picks ni
emite EV/probabilidad automática . Se pre-registra ANTES del KO
(``logged_at``) para evitar drift de retrospectiva. Validar por ROI REAL en 20-30+, NO por CLV
(demoted). Complementa, no reemplaza, los markdown que mantiene el operador.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# notes/analysis_log.jsonl a la raíz del repo (local, gitignored).
LOG_PATH = Path(__file__).resolve().parents[5] / "notes" / "analysis_log.jsonl"

Regime = Literal["amistoso", "internacional"]
Coverage = Literal["alta", "media", "baja"]
ReadLevel = Literal["team", "prop"]
Tier = Literal["secondary", "efficient"]
Status = Literal["pending", "HIT", "MISS", "void"]


class AnalysisRead(BaseModel):
    """Un read marcado (evento más probable), pre-registrado antes del KO."""

    model_config = ConfigDict(extra="forbid")

    id: str
    date: str  # fecha del partido YYYY-MM-DD
    fixture: str  # "Local vs Visitante"
    regime: Regime
    coverage: Coverage
    read_level: ReadLevel
    market: str  # córners | tarjetas | 1X2 | totales | BTTS | AH | prop:<jugador> ...
    selection: str  # el lado/línea marcado
    tier: Tier  # secundario (book no colapsa) vs eficiente (1X2/totales-main/BTTS-main)
    script_label: str | None = None  # p.ej. "dominador estéril", "host fade", "P5a"
    odds: float | None = None
    stake: float | None = None
    status: Status = "pending"
    roi: float | None = None  # P/L realizado en unidades (al graduar)
    notes: str | None = None
    logged_at: str  # ISO timestamp; pre-KO = pre-registro


def append_read(read: AnalysisRead, path: Path = LOG_PATH) -> None:
    """Añade una fila al JSONL (append-only). Crea el archivo si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(read.model_dump_json() + "\n")


def load_reads(path: Path = LOG_PATH) -> list[AnalysisRead]:
    """Lee todas las filas. Devuelve [] si el archivo no existe."""
    if not path.exists():
        return []
    out: list[AnalysisRead] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(AnalysisRead.model_validate_json(line))
    return out


def grade_read(
    read_id: str,
    status: Status,
    roi: float | None = None,
    path: Path = LOG_PATH,
) -> bool:
    """Marca el resultado de un read por id (reescribe el JSONL). Devuelve False si no existe."""
    reads = load_reads(path)
    found = False
    for r in reads:
        if r.id == read_id:
            r.status = status
            if roi is not None:
                r.roi = roi
            found = True
    if found:
        with path.open("w", encoding="utf-8") as f:
            for r in reads:
                f.write(r.model_dump_json() + "\n")
    return found


def slice_log(
    reads: list[AnalysisRead],
    by: tuple[str, ...] = ("read_level", "tier", "regime"),
) -> dict[tuple, dict]:
    """Agrupa los reads GRADUADOS (HIT/MISS) por los campos en ``by`` y computa, por grupo:
    n, hits, hit_rate, roi_sum, roi_per_bet (solo sobre filas con stake/roi). Ignora pending/void."""
    graded = [r for r in reads if r.status in ("HIT", "MISS")]
    groups: dict[tuple, list[AnalysisRead]] = {}
    for r in graded:
        key = tuple(getattr(r, field) for field in by)
        groups.setdefault(key, []).append(r)

    out: dict[tuple, dict] = {}
    for key, rs in groups.items():
        n = len(rs)
        hits = sum(1 for r in rs if r.status == "HIT")
        roied = [r for r in rs if r.roi is not None]
        roi_sum = round(sum(r.roi for r in roied), 2) if roied else None
        out[key] = {
            "n": n,
            "hits": hits,
            "hit_rate": round(hits / n, 3),
            "n_with_roi": len(roied),
            "roi_sum": roi_sum,
            "roi_per_bet": round(roi_sum / len(roied), 3) if roied else None,
        }
    return out
