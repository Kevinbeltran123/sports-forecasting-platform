"""MundialModel — BaseModel wrapper around the WC2026 locked predictions.

Sprint 0 Ola C. Loads the SHA-256-verified lock and emits PredictionRecord
rows using Max-P selection (spec §5.6) over the markets the predictor
covers: 1x2, btts, ou_2.5. The lock currently carries
calibration_status='below-gate' — that flag is propagated transparently
in `payload.calibration_status` so downstream gates can choose to drop
or shadow these picks.

This model does NOT implement train()/evaluate() — Mundial uses a shipped
artifact. Post-tournament scoring (after 2026-07-19) will be a separate
evaluation task, not a BaseModel.evaluate() call.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from bip.models.base import BaseModel, PredictionRecord

log = structlog.get_logger(__name__)

# Canonical lock location
DEFAULT_LOCK_PATH = Path(
    "src/bip/evaluation/tournaments/locked_predictions/world_cup_2026/lock.json"
)

# Max-P threshold from the implementation spec §5.6.
# Threshold is tunable but defaults to 0.65 (≥65% model probability to
# emit a pick). The lock is below-gate so few fixtures may clear this in
# practice; that is acceptable — the registry simply yields empty.
P_MAX_THRESHOLD = 0.65

# Predictor key inside lock.json["fixtures"][i]["predictions"]
_PREDICTOR_KEY = "bayesian_bivariate_xg_blended"

# Competition aliases that route to MundialModel
_COMPETITION_ALIASES = frozenset({"WC2026", "FIFA World Cup", "World Cup", "wc2026"})


def _canonical_content_for_hash(data: dict[str, Any]) -> str:
    """Recompute the lock content string used to derive content_hash.

    Mirrors the lock generator: strip the stored content_hash before
    hashing the rest as canonical JSON.
    """
    cleaned = {k: v for k, v in data.items() if k != "content_hash"}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _verify_lock_integrity(lock: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, computed_prefix) for content_hash integrity check.

    The stored content_hash is the first 16 chars of sha256 over the
    canonical JSON (matches the lock-emitter
    behavior). If a future lock-format version differs, this function
    is the single place to update.
    """
    stored = str(lock.get("content_hash", ""))
    if not stored:
        return False, ""
    payload = _canonical_content_for_hash(lock)
    full = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    computed_prefix = full[: len(stored)]
    return (stored == computed_prefix), computed_prefix


class MundialModel(BaseModel):
    """BaseModel impl that emits frozen picks from the WC2026 lock."""

    name = "mundial_wc2026"

    def __init__(
        self,
        lock_path: Path | str = DEFAULT_LOCK_PATH,
        *,
        p_max_threshold: float = P_MAX_THRESHOLD,
        verify_hash: bool = True,
    ) -> None:
        self.lock_path = Path(lock_path)
        self.p_max_threshold = p_max_threshold
        if not self.lock_path.exists():
            raise FileNotFoundError(f"Lock not found at {self.lock_path}")

        lock = json.loads(self.lock_path.read_text())
        self._raw = lock
        self.version = f"lock-{lock.get('git_sha', 'unknown')[:7]}"

        # Verify SHA-256 integrity (warn-not-block — the lock may have
        # been regenerated with a slightly different canonical form, and
        # we don't want to brick the model on a format drift).
        if verify_hash:
            ok, computed = _verify_lock_integrity(lock)
            if not ok:
                log.warning(
                    "mundial_lock_hash_mismatch",
                    stored=lock.get("content_hash"),
                    computed=computed,
                    lock_path=str(self.lock_path),
                )

        # Index fixtures by match_id for O(1) lookup
        self._fixtures_by_id: dict[str, dict[str, Any]] = {
            f["match_id"]: f for f in lock.get("fixtures", [])
        }
        self.calibration_status = lock.get("calibration_status", "unknown")
        log.info(
            "mundial_model_loaded",
            n_fixtures=len(self._fixtures_by_id),
            calibration_status=self.calibration_status,
            version=self.version,
        )

    # ------------------------------------------------------------------
    # BaseModel contract
    # ------------------------------------------------------------------

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        comp = fixture.get("competition")
        match_id = fixture.get("match_id") or fixture.get("fixture_id")
        if comp not in _COMPETITION_ALIASES:
            return False
        return match_id in self._fixtures_by_id

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        match_id = fixture.get("match_id") or fixture.get("fixture_id")
        if match_id not in self._fixtures_by_id:
            return []

        locked = self._fixtures_by_id[match_id]
        preds = locked.get("predictions", {}).get(_PREDICTOR_KEY)
        if not preds:
            log.info("mundial_no_predictor_data", match_id=match_id)
            return []

        # Max-P scan across covered markets
        market_probs = self._scan_market_probabilities(preds)
        if not market_probs:
            return []
        best_market, best_selection, best_p = max(market_probs, key=lambda x: x[2])
        if best_p < self.p_max_threshold:
            log.info(
                "mundial_below_threshold",
                match_id=match_id,
                best_market=best_market,
                best_selection=best_selection,
                best_p=round(best_p, 4),
                threshold=self.p_max_threshold,
            )
            return []

        record = PredictionRecord(
            source="mundial",
            fixture_id=str(match_id),
            competition="WC2026",
            home_team=str(locked.get("home_team_name", "")),
            away_team=str(locked.get("away_team_name", "")),
            match_datetime=self._parse_kickoff(locked.get("kickoff_utc")),
            market=best_market,
            selection=best_selection,
            p_model=float(best_p),
            ev=None,  # EV computed later if odds are present in the fixture context
            odds_at_pick=None,
            payload={
                "predictor": _PREDICTOR_KEY,
                "tournament_phase": locked.get("tournament_phase"),
                "calibration_status": self.calibration_status,
                "lock_content_hash": self._raw.get("content_hash"),
                "p_max_threshold": self.p_max_threshold,
                "scanned_markets": [
                    {"market": m, "selection": s, "p": round(p, 4)}
                    for (m, s, p) in market_probs
                ],
            },
            model_version=self.version,
        )
        return [record]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_market_probabilities(
        preds: dict[str, float],
    ) -> list[tuple[str, str, float]]:
        """Flatten predictor output into (market, selection, p) tuples.

        Markets covered today by bayesian_bivariate_xg_blended:
          - 1x2: p_home_win, p_draw, p_away_win
          - btts: p_btts (yes) — implicit no = 1 - p_btts
          - ou_2.5: p_over_2_5 (over) — implicit under = 1 - p_over_2_5

        Missing fields are silently skipped — the predictor may grow
        over time.
        """
        out: list[tuple[str, str, float]] = []

        if "p_home_win" in preds:
            out.append(("1x2", "home", float(preds["p_home_win"])))
        if "p_draw" in preds:
            out.append(("1x2", "draw", float(preds["p_draw"])))
        if "p_away_win" in preds:
            out.append(("1x2", "away", float(preds["p_away_win"])))

        if "p_btts" in preds:
            p_yes = float(preds["p_btts"])
            out.append(("btts", "yes", p_yes))
            out.append(("btts", "no", 1.0 - p_yes))

        if "p_over_2_5" in preds:
            p_over = float(preds["p_over_2_5"])
            out.append(("ou_2.5", "over", p_over))
            out.append(("ou_2.5", "under", 1.0 - p_over))

        return out

    @staticmethod
    def _parse_kickoff(raw: Any) -> datetime:
        if isinstance(raw, datetime):
            return raw
        if not raw:
            raise ValueError("kickoff_utc missing from locked fixture")
        text = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(text)

    # ------------------------------------------------------------------
    # Introspection helpers (useful for smoke tests + dashboards)
    # ------------------------------------------------------------------

    @property
    def fixture_ids(self) -> list[str]:
        return list(self._fixtures_by_id.keys())


__all__ = ["DEFAULT_LOCK_PATH", "MundialModel", "P_MAX_THRESHOLD"]
