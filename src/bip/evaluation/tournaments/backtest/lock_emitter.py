"""Lock JSON emitter — Phase 6 final artifact for the WC2026 calibration spike.

Phase 6 (hard deadline 2026-06-08) closes the spike by writing per-fixture
locked predictions to ``locked_predictions/world_cup_2026/`` with a
SHA-256 content hash for tamper-evidence. Post-tournament scoring uses
the locked artifact as the ground-truth record of what the system
predicted *before* the tournament started — no goalpost moving.

Design properties:

- **Tamper-evident**: ``content_hash`` is SHA-256 over the canonical
  JSON serialization of every field except ``content_hash`` itself,
  with sorted keys and ISO 8601 datetimes. ``verify_lock_json()`` returns
  ``(lock, is_valid)``; ``is_valid=False`` means the file was modified
  after lock.
- **Schema-versioned**: ``schema_version="1.0"`` so future format changes
  don't break post-tournament scoring scripts. Bump on breaking changes;
  loaders can dispatch by version.
- **Always-emittable, safe by default**: when ``LockDecision.passes_lock``
  is False (e.g., below-gate, coverage-partial), ``emit_lock_json()``
  raises by default. ``force=True`` opts into the spike R-08 slip plan:
  *"lock anyway with explicit calibration_status flag in the JSON"* —
  the override is recorded in ``forced_emit_reason`` so the audit trail
  is preserved.
- **Calibration audit trail surfaced**: the full ``LockDecision``
  (per-predictor verdicts, market-level ECE/Brier, operator overrides)
  is embedded so reviewers can interrogate exactly why each prediction
  was locked.

References:
- internal research notes Conclusion 1 (CLV chain composition)
- internal planning notes §6 LOCK + R-08 slip plan
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bip.evaluation.tournaments.backtest.lock_gate import (
    LockDecision,
    PredictorGateVerdict,
)

# Bump when the LockJSON shape changes in a non-backward-compatible way.
# Loaders SHOULD dispatch by schema_version when adding fields that change
# the canonical-form hash computation.
LOCK_SCHEMA_VERSION = "1.0"

# Default destination for committed lock artifacts (per spike §10).
DEFAULT_LOCK_DIR = (
    Path(__file__).resolve().parent.parent
    / "locked_predictions"
    / "world_cup_2026"
)


# ── Per-fixture predictions ──────────────────────────────────────────────────


class FixtureLockedPredictions(BaseModel):
    """All predictors' market probabilities for one tournament fixture.

    The ``predictions`` map is intentionally schemaless beyond the per-market
    float values — different predictors emit different markets, and the
    audit trail is what the lock JSON exists to preserve. Validation of
    individual market keys / probability ranges is the predictor's
    responsibility (see ``predictors/base.py``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_id: str
    tournament_phase: str  # 'group_a' | 'group_b' | ... | 'r16' | 'qf' | 'sf' | 'final'
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str

    # predictor_name -> {market_name -> probability}
    # Example: {"bivariate_poisson": {"p_home_win": 0.42, "p_draw": 0.28, ...}}
    predictions: dict[str, dict[str, float]]

    @field_validator("predictions")
    @classmethod
    def _all_probs_are_valid(
        cls, v: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Each probability must be in [0, 1]; reject NaN / negative / > 1."""
        for predictor, markets in v.items():
            for market, p in markets.items():
                if not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"{predictor}.{market}: probability {p} out of [0, 1]"
                    )
        return v


# ── Top-level lock artifact ──────────────────────────────────────────────────


class LockJSON(BaseModel):
    """The committed lock — what the system predicted before WC2026 kicks off."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = LOCK_SCHEMA_VERSION
    tournament_slug: str  # 'world_cup_2026'
    locked_at: datetime
    git_sha: str | None = None

    # Calibration audit trail — embedded LockDecision contents
    calibration_status: str
    held_out_tournaments: tuple[str, ...]
    n_fixtures_total: int = Field(ge=0)
    n_fixtures_with_predictions: int = Field(ge=0)
    coverage_threshold: float = Field(ge=0.0, le=1.0)
    predictor_verdicts: tuple[PredictorGateVerdict, ...]
    operator_overrides: tuple[str, ...]

    # If emit_lock_json was called with force=True over a non-passing decision,
    # this carries the operator's stated reason for shipping anyway.
    forced_emit_reason: str | None = None

    # The locked predictions
    fixtures: tuple[FixtureLockedPredictions, ...]

    # Tamper-evidence
    content_hash: str = Field(
        description="SHA-256 of canonical JSON of all fields except content_hash",
    )

    # ── derived ───────────────────────────────────────────────────────────

    @property
    def coverage_pct(self) -> float:
        if self.n_fixtures_total == 0:
            return 0.0
        return self.n_fixtures_with_predictions / self.n_fixtures_total

    # ── persistence ───────────────────────────────────────────────────────

    def to_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        return cls.model_validate_json(Path(path).read_text())


# ── Content hashing ──────────────────────────────────────────────────────────


def _canonical_payload(lock_data: dict[str, Any]) -> str:
    """Produce the canonical JSON string used for content_hash computation.

    Excludes ``content_hash`` itself; sorts keys deterministically; coerces
    datetimes to ISO 8601 strings. Two LockJSONs with identical contents
    (modulo content_hash) yield identical canonical strings → identical
    hashes.
    """
    sanitized = {k: v for k, v in lock_data.items() if k != "content_hash"}
    return json.dumps(sanitized, sort_keys=True, default=str, separators=(",", ":"))


def _compute_content_hash(lock_data: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical payload."""
    payload = _canonical_payload(lock_data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Emitter ──────────────────────────────────────────────────────────────────


def emit_lock_json(
    decision: LockDecision,
    fixtures: Iterable[FixtureLockedPredictions],
    *,
    tournament_slug: str = "world_cup_2026",
    git_sha: str | None = None,
    output_path: Path | None = None,
    force: bool = False,
    forced_emit_reason: str | None = None,
    locked_at: datetime | None = None,
) -> LockJSON:
    """Build and optionally write the lock JSON.

    Args:
        decision: The Phase 5 LockDecision aggregating per-predictor verdicts.
        fixtures: Per-fixture locked predictions for every tournament match
            the system predicted.
        tournament_slug: Identifier for the tournament (default 'world_cup_2026').
        git_sha: Commit hash for reproducibility. Falls back to decision.git_sha.
        output_path: Where to write the JSON. If None, returns the LockJSON
            without persisting (useful for tests / dry-runs).
        force: When True, emits even if ``decision.passes_lock`` is False.
            Required to lock a below-gate / coverage-partial decision per
            R-08 slip plan. Records ``forced_emit_reason`` in the artifact.
        forced_emit_reason: Operator-supplied rationale for ``force=True``.
            Required when ``force=True`` so the audit trail is preserved.
        locked_at: Override the lock timestamp (mainly for deterministic
            tests). Defaults to current UTC.

    Returns:
        LockJSON with content_hash computed.

    Raises:
        ValueError: If the decision does not pass and ``force`` is False, or
            if ``force=True`` is set without ``forced_emit_reason``.
    """
    if not decision.passes_lock and not force:
        raise ValueError(
            f"LockDecision does not pass: calibration_status="
            f"{decision.calibration_status!r}. "
            "Pass force=True with forced_emit_reason='...' to emit anyway "
            "(per spike R-08 slip plan)."
        )
    if force and not forced_emit_reason:
        raise ValueError(
            "force=True requires forced_emit_reason='...' so the audit "
            "trail records why the lock was emitted despite a non-passing "
            "calibration verdict."
        )

    fixtures_tuple = tuple(fixtures)
    effective_git_sha = git_sha if git_sha is not None else decision.git_sha
    effective_locked_at = locked_at if locked_at is not None else datetime.now(UTC)

    # Build the artifact with a placeholder hash, compute the real hash,
    # then construct the final immutable model.
    placeholder = LockJSON(
        schema_version=LOCK_SCHEMA_VERSION,
        tournament_slug=tournament_slug,
        locked_at=effective_locked_at,
        git_sha=effective_git_sha,
        calibration_status=decision.calibration_status,
        held_out_tournaments=decision.held_out_tournaments,
        n_fixtures_total=decision.n_fixtures_total,
        n_fixtures_with_predictions=decision.n_fixtures_with_predictions,
        coverage_threshold=decision.coverage_threshold,
        predictor_verdicts=decision.predictor_verdicts,
        operator_overrides=decision.operator_overrides,
        forced_emit_reason=forced_emit_reason,
        fixtures=fixtures_tuple,
        content_hash="",  # placeholder; recomputed below
    )

    real_hash = _compute_content_hash(placeholder.model_dump(mode="json"))

    final = placeholder.model_copy(update={"content_hash": real_hash})

    if output_path is not None:
        final.to_json(output_path)

    return final


# ── Verifier ─────────────────────────────────────────────────────────────────


def verify_lock_json(path: Path | str) -> tuple[LockJSON, bool]:
    """Load a lock JSON and verify the content_hash hasn't been tampered with.

    Returns:
        ``(lock, is_valid)``. When ``is_valid=False``, the file's contents
        diverged from the original lock — somebody edited the JSON after
        the lock was written. Post-tournament scoring should treat this
        as evidence of manipulation and refuse to grade the predictions.
    """
    lock = LockJSON.from_json(path)
    expected_hash = _compute_content_hash(lock.model_dump(mode="json"))
    return lock, lock.content_hash == expected_hash
