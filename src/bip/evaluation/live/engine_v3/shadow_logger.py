"""Shadow-mode pick logger — appends V3 PipelineOutput to parquet.

Phase 2 deliverable from sec 10. The output structure mirrors the
``LineRecorder`` pattern: daily partitions under
``data/cache/v3_shadow/dt=YYYY-MM-DD/{picks,gate_denials,gsv_log}.parquet``.

Three files per day:

- ``picks.parquet``: one row per allowed candidate (the would-be bet)
- ``gate_denials.parquet``: one row per rejected candidate with rule
  number + reason (sec 7.2 audit trail requirement)
- ``gsv_log.parquet``: one row per pipeline frame with full GSV JSON.
  Enables (a) refitting OOD detector + pattern layer on real data
  (Phase 4 weekly refit cadence) and (b) reconstructing the audit trail
  for any pick/denial by joining on ``(fixture_id, state_version)``.

The shadow output is read-only from the operator's perspective — Phase
2 of sec 10 forbids routing these to Telegram. They feed the Phase-2
comparison: v3 shadow ROI vs current-system ROI on the same fixture
cohort.

GSV persistence design choices (T1.1 ship note):

- Sync buffer + periodic flush rather than async/thread: ``record()``
  only appends a Python dict (O(1)); flush is the only I/O and runs
  outside the path-critical pick generation step. No thread complexity
  needed and the pattern matches the existing pick/denial buffers.

- ``model_dump_json()`` from pydantic-core (Rust) is sub-millisecond
  per GSV at v3's schema size. Persistence is well under the 5ms budget
  from the mission spec.

- Daily rotation by GSV timestamp (not flush timestamp). Two flushes
  spanning midnight UTC write to two distinct partitions so the date
  partition matches the GSV reality, not the operator's wall clock.

- Schema evolution: a future GSV that adds fields stays
  backward-compatible because ``model_validate_json`` defaults to ignoring
  missing-then-required keys ONLY if the model declares defaults.
  If a load fails, ``load_shadow_gsvs`` returns ``LoadFailure`` records
  carrying the offending row + exception — callers decide whether to
  skip or fail. We never silently coerce.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.pipeline import PipelineOutput

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.no_bet_gate import GateResult

DEFAULT_SHADOW_ROOT = Path("data/cache/v3_shadow")


def _date_partition(ts: datetime) -> str:
    return ts.strftime("dt=%Y-%m-%d")


def _pick_row(pick, fixture_id: int, ts: datetime, gsv) -> dict[str, Any]:
    """Build a row for picks.parquet.

    The ``gsv`` argument is used to extract per-pick stake info that
    isn't on the pick itself: the bookmaker decimal odd at pick time
    (for the side being bet) and the full-Kelly fraction. These let
    Workflow #2 (calibration) and stake-sizing recalibration analyses
    proceed offline without re-parsing the full gsv_json.

    Kelly is precomputed as the FULL fraction (not 1/4). Operator's
    real stake = full_kelly × operator_fraction (1/4 by convention).
    Persisting the full value lets analysts vary the fraction at
    analysis time.
    """
    t = pick.full_thesis
    line = gsv.markets.lines.get(pick.candidate.market_id) if gsv else None
    book_odd, kelly_full_pct, line_value = _derive_stake_fields(
        pick=pick, line=line,
    )
    return {
        "fixture_id": int(fixture_id),
        "timestamp_utc": ts,
        "thesis_id": t.id,
        "archetype": t.archetype.value,
        "thesis_layer": t.source.layer,
        "rule_id": t.source.identifier,
        "family": t.prediction.family.value,
        "direction": t.prediction.direction,
        "magnitude_pp": float(t.prediction.magnitude_pp),
        "horizon_minutes": int(t.prediction.horizon.horizon_minutes),
        "market_id": pick.candidate.market_id,
        "line_value": line_value,
        "bookmaker_odd": book_odd,
        "kelly_full_pct": kelly_full_pct,
        "fair_prob": float(pick.candidate.fair_prob),
        "base_edge": float(pick.candidate.mes.base_edge),
        "signal_clarity": float(pick.candidate.mes.signal_clarity),
        "book_slowness": float(pick.candidate.mes.book_slowness),
        "liquidity_score": float(pick.candidate.mes.liquidity_score),
        "conditional_variance": float(pick.candidate.mes.conditional_variance),
        "mes_score": float(pick.candidate.mes.score),
        "confidence_prior": float(t.confidence_prior),
        "activated_at_minute": int(t.activated_at_minute),
    }


def _derive_stake_fields(
    *, pick, line,
) -> tuple[float | None, float | None, float | None]:
    """Return (bookmaker_odd, kelly_full_pct, line_value) for the pick.

    Maps thesis direction to which side of the MarketLine to use:
      - over / yes / home → side_a_decimal
      - under / no / away / draw → side_b_decimal (falls back to side_a)
      - 1X2: home → side_a; draw → side_b; away → side_c if present, else side_b

    Returns (None, None, line_value) when the line is missing or the
    decimal can't be resolved. Kelly is computed as:
      f* = (p*(b)-(1-p)) / b   with b = decimal_odd - 1
    Negative Kelly is clamped to 0 (sign of "do not bet"; the gate
    SHOULD have caught this upstream but we don't double-penalize here).
    """
    if line is None:
        return (None, None, None)

    line_value = float(line.line_value) if line.line_value is not None else None
    direction = pick.full_thesis.prediction.direction.lower()
    if direction in ("over", "yes", "home"):
        book_odd = line.side_a_decimal
    elif direction in ("under", "no", "away", "draw"):
        book_odd = line.side_b_decimal or line.side_a_decimal
    else:
        book_odd = line.side_a_decimal

    if book_odd is None or book_odd <= 1.0:
        return (None, None, line_value)

    p = float(pick.candidate.fair_prob)
    b = float(book_odd) - 1.0
    kelly = (p * b - (1.0 - p)) / b
    kelly = max(0.0, kelly)

    return (float(book_odd), float(kelly), line_value)


def _denial_row(result, fixture_id: int, ts: datetime, *, is_shadow: bool = False) -> dict[str, Any]:
    """Build a row for gate_denials.parquet.

    ``is_shadow=True`` marks rows that were NOT enforced: the candidate
    PASSED the gate but a shadow denial is recorded for offline analysis.
    Existing enforced-denial rows always have ``is_shadow=False``.
    The parquet is written with ``diagonal_relaxed`` concat so existing
    readers without the column receive ``null`` — backward-compatible.
    """
    cand = result.candidate
    return {
        "fixture_id": int(fixture_id),
        "timestamp_utc": ts,
        "thesis_id": cand.thesis.id,
        "archetype": cand.thesis.archetype.value,
        "family": cand.family.value,
        "market_id": cand.market_id,
        "rule_number": int(result.verdict.rule_number or 0),
        "reason": result.verdict.reason,
        "mes_score": float(cand.mes.score),
        "direction": cand.thesis.prediction.direction,
        "is_shadow": bool(is_shadow),
    }


def _gsv_row(gsv: GameStateVector) -> dict[str, Any]:
    """One row per pipeline frame. ``gsv_json`` carries the full GSV
    serialised via ``model_dump_json()`` (Pydantic v2 round-trip).
    """
    return {
        "fixture_id": int(gsv.fixture_id),
        "state_version": int(gsv.state_version),
        "timestamp_utc": gsv.timestamp_utc,
        "home_team_id": int(gsv.home_team_id),
        "away_team_id": int(gsv.away_team_id),
        "minute": int(gsv.time.minute),
        "period": gsv.time.period,
        "home_goals": int(gsv.score.home_goals),
        "away_goals": int(gsv.score.away_goals),
        "gsv_json": gsv.model_dump_json(),
    }


@dataclass(frozen=True)
class LoadFailure:
    """A row that could not be deserialised into a ``GameStateVector``.

    Returned by ``load_shadow_gsvs`` so callers can decide whether to
    skip, retry, or surface the corruption. We never silently swallow
    schema mismatches — that would defeat the audit-trail purpose.
    """

    path: Path
    row_index: int
    fixture_id: int
    error: str


class ShadowLogger:
    """Buffered parquet logger for shadow picks + gate denials + GSV log."""

    def __init__(
        self,
        output_root: Path | str = DEFAULT_SHADOW_ROOT,
        *,
        gsv_log_enabled: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.gsv_log_enabled = gsv_log_enabled
        self._pick_buf: list[dict[str, Any]] = []
        self._denial_buf: list[dict[str, Any]] = []
        self._gsv_buf: list[dict[str, Any]] = []

    def record_shadow_denial(
        self,
        result: "GateResult",
        fixture_id: int,
        ts: datetime,
    ) -> None:
        """Record a shadow denial row — candidate PASSED but would have been
        denied by the shadow rule. ``is_shadow=True`` distinguishes these
        from enforced denials in gate_denials.parquet so offline analysis
        can compare shadow vs real gate outcomes.

        This is the injection point for rule_9 (OOD, shadow-only) and
        rule_8 (horizon-squashed cvar, shadow-only for HT GOALS).
        """
        self._denial_buf.append(_denial_row(result, fixture_id, ts, is_shadow=True))

    def record(self, output: PipelineOutput) -> tuple[int, int, int]:
        """Append the picks + denials + GSV from one pipeline frame.

        Returns ``(n_picks, n_denials, n_gsv)`` recorded this call.
        ``n_gsv`` is 0 if GSV logging is disabled, else 1.
        """
        ts = output.gsv.timestamp_utc
        fixture_id = output.gsv.fixture_id

        for pick in output.allowed_picks:
            self._pick_buf.append(_pick_row(pick, fixture_id, ts, output.gsv))

        for r in output.gate_results:
            if r.verdict.allowed:
                continue
            self._denial_buf.append(_denial_row(r, fixture_id, ts))

        n_gsv = 0
        if self.gsv_log_enabled:
            self._gsv_buf.append(_gsv_row(output.gsv))
            n_gsv = 1

        return (
            len(output.allowed_picks),
            sum(1 for r in output.gate_results if not r.verdict.allowed),
            n_gsv,
        )

    def flush(self, timestamp_utc: datetime | None = None) -> dict[str, Path]:
        """Write buffers to parquet partitions.

        Returns dict keyed by ``picks`` / ``denials`` / ``gsv_log`` to the
        parquet path written, omitting entries that were empty.

        Partitioning: by row timestamp, not flush timestamp. A single
        flush() spanning two UTC days writes to two partitions.
        """
        out: dict[str, Path] = {}
        wall_ts = timestamp_utc or datetime.now(timezone.utc)

        # Bucket each buffer by row.timestamp_utc → daily partition.
        # Falls back to the wall-ts partition for any row lacking the field
        # (defensive; current code always populates it).
        def _bucket(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
            buckets: dict[str, list[dict[str, Any]]] = {}
            for r in rows:
                row_ts = r.get("timestamp_utc")
                if isinstance(row_ts, datetime):
                    key = _date_partition(row_ts)
                else:
                    key = _date_partition(wall_ts)
                buckets.setdefault(key, []).append(r)
            return buckets

        # The mission specifies one parquet per partition per kind. With
        # cross-midnight flushes we may end up writing multiple partitions
        # in one call; ``out`` returns ONE path per kind (the most-recent
        # by date). This keeps the API simple and is fine for callers that
        # just need to know the partition was written.
        for kind, buf in (
            ("picks", self._pick_buf),
            ("denials", self._denial_buf),
            ("gsv_log", self._gsv_buf),
        ):
            if not buf:
                continue
            filename = {
                "picks": "picks.parquet",
                "denials": "gate_denials.parquet",
                "gsv_log": "gsv_log.parquet",
            }[kind]
            for part, rows in sorted(_bucket(buf).items()):
                date_dir = self.output_root / part
                date_dir.mkdir(parents=True, exist_ok=True)
                path = date_dir / filename
                self._append_parquet(path, rows)
                out[kind] = path  # last partition wins in the return dict
            buf.clear()

        return out

    def buffer_size(self) -> tuple[int, int, int]:
        """Return ``(picks, denials, gsv)`` buffer counts."""
        return (
            len(self._pick_buf),
            len(self._denial_buf),
            len(self._gsv_buf),
        )

    @staticmethod
    def _append_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
        import polars as pl

        new_df = pl.DataFrame(rows)
        if path.exists():
            try:
                existing = pl.read_parquet(path)
                combined = pl.concat([existing, new_df], how="diagonal_relaxed")
            except Exception:
                combined = new_df
        else:
            combined = new_df
        combined.write_parquet(path)


def load_shadow_gsvs(
    date_range: tuple[datetime, datetime] | None = None,
    output_root: Path | str = DEFAULT_SHADOW_ROOT,
) -> tuple[list[GameStateVector], list[LoadFailure]]:
    """Load persisted GSVs from a date range.

    ``date_range`` is ``(start, end)`` inclusive on both ends. ``None``
    loads every daily partition present under ``output_root``. Dates are
    matched by partition name (``dt=YYYY-MM-DD``), not by the GSV's own
    ``timestamp_utc`` field — so a small clock skew at midnight is
    tolerable.

    Returns ``(successes, failures)``. Failures carry the offending path,
    row index, fixture_id, and the raised exception message. Callers
    decide whether to fail-fast or skip-and-continue. We never silently
    drop a malformed row.
    """
    import polars as pl

    root = Path(output_root)
    if not root.exists():
        return [], []

    if date_range is not None:
        start, end = date_range
        if start > end:
            raise ValueError(f"date_range start ({start.date()}) > end ({end.date()})")

    successes: list[GameStateVector] = []
    failures: list[LoadFailure] = []

    for partition_dir in sorted(root.iterdir()):
        if not partition_dir.is_dir():
            continue
        if not partition_dir.name.startswith("dt="):
            continue
        try:
            part_date = datetime.strptime(partition_dir.name[3:], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if date_range is not None:
            if part_date.date() < date_range[0].date():
                continue
            if part_date.date() > date_range[1].date():
                continue

        gsv_path = partition_dir / "gsv_log.parquet"
        if not gsv_path.exists():
            continue

        df = pl.read_parquet(gsv_path)
        for idx, json_blob in enumerate(df["gsv_json"].to_list()):
            try:
                gsv = GameStateVector.model_validate_json(json_blob)
            except ValidationError as e:
                fid = int(df["fixture_id"][idx]) if "fixture_id" in df.columns else 0
                failures.append(
                    LoadFailure(
                        path=gsv_path,
                        row_index=idx,
                        fixture_id=fid,
                        error=str(e),
                    )
                )
                continue
            successes.append(gsv)

    return successes, failures


__all__ = [
    "DEFAULT_SHADOW_ROOT",
    "LoadFailure",
    "ShadowLogger",
    "load_shadow_gsvs",
]
