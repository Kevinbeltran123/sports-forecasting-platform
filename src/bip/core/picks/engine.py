"""PickEngine — the Phase 3 orchestration heart (PICK-01 through PICK-05, CLAUDE-01).

Single async entry point: evaluate(prediction, opening_odds) -> Pick | None.

Filter chain (D-02 + D-09 + D-10 + D-11 + D-07 + D-14):
  1. simulate_pick edge filter — per-market threshold from LeagueRegistry
     (G-MAINT-01 / CORE-05). simulate_pick still imported from bip.train.backtest
     (D-02 no duplication); EDGE_THRESHOLD_PCT remains the fallback default.
  2. quarter_kelly sizing
  3. exceeds_60pct_cap (D-09 — drop on bind)
  4. ClaudeValidator.validate — None -> filtered/claude_api_unavailable (D-07);
     REJECT -> rejected (D-03, D-14)
  5. CONFIRM/FLAG -> status=pending, schedule DateTrigger send

D-04: every branch persists a Pick row.
specifics §195: idempotent on (fixture_id, market, prediction_id).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from bip.core.claude.validator import ClaudeValidator, ClaudeVerdict
from bip.core.errors import PickError
from bip.core.picks.staking import (
    exceeds_60pct_cap,
    quarter_kelly_units,
    round_to_nearest_half_unit,
)
from bip.core.settings import Settings
from bip.core.storage.models import Pick, Prediction
from bip.core.storage.repositories import PickRepository
from bip.core.telegram.sender import TelegramSender
from bip.core.types import MarketKey, PickStatus
from bip.sports.football.config.league_registry import LeagueRegistry
from bip.train.backtest import EDGE_THRESHOLD_PCT, simulate_pick

logger = structlog.get_logger(__name__)

_SELECTION_KEYS = ("1", "X", "2")


class PickEngine:
    """Orchestrates pick evaluation end-to-end."""

    def __init__(
        self,
        pick_repo: PickRepository,
        validator: ClaudeValidator,
        scheduler: AsyncIOScheduler,
        sender: TelegramSender,
        settings: Settings,
        league_registry: LeagueRegistry,
    ) -> None:
        self._pick_repo = pick_repo
        self._validator = validator
        self._scheduler = scheduler
        self._sender = sender
        self._settings = settings
        self._league_registry = league_registry

    async def evaluate(self, prediction: Prediction, opening_odds: dict[str, Any]) -> Pick | None:
        fixture_id = prediction.fixture_id
        market = prediction.market

        try:
            probs = np.array([prediction.probabilities[k] for k in _SELECTION_KEYS], dtype=float)
            odds = np.array([float(opening_odds[k]) for k in _SELECTION_KEYS], dtype=float)
        except (KeyError, TypeError) as exc:
            raise PickError(
                f"Missing 1X2 keys in prediction.probabilities or opening_odds for "
                f"fixture_id={fixture_id}: {exc}"
            ) from exc

        # G-MAINT-01 / CORE-05: per-market threshold from league YAML.
        # G-MAINT-05: market normalization delegated to MarketKey enum.
        try:
            market_key = MarketKey.from_str(market)
            league_cfg = self._league_registry.get(prediction.league)
            threshold = getattr(
                league_cfg.model_params,
                market_key.to_threshold_attr(),
                EDGE_THRESHOLD_PCT,
            )
        except (KeyError, ValueError):
            # Unknown league OR unknown market alias -- fall back to backtest default.
            threshold = EDGE_THRESHOLD_PCT

        idx = simulate_pick(probs, odds, threshold=threshold)
        if idx is None:
            return self._persist_filtered(prediction, opening_odds, "no_edge")

        selection = _SELECTION_KEYS[idx]
        edge = float(probs[idx] * odds[idx] - 1.0)
        raw_stake = quarter_kelly_units(edge, float(odds[idx]), self._settings.max_kelly_fraction)
        stake = round_to_nearest_half_unit(raw_stake)
        kelly_fraction = raw_stake

        if exceeds_60pct_cap(self._pick_repo, market=market, sport=prediction.sport):
            return self._persist_filtered(prediction, opening_odds, "market_cap")

        pick_summary = self._build_pick_summary(prediction, selection, odds[idx], edge, stake)
        curated_signals = self._build_curated_signals(prediction)
        verdict = await self._validator.validate(pick_summary, curated_signals)

        if verdict is None:
            # D-01: claude_failure_mode controls behavior when Claude API is unreachable.
            if self._settings.claude_failure_mode == "filter":
                # Default — Phase 3 D-07 preserved: conservative (no unvalidated alerts).
                return self._persist_filtered(
                    prediction, opening_odds, "claude_api_unavailable"
                )
            # claude_failure_mode == "skip" (D-01 opt-in): persist as pending+SKIPPED, send.
            skipped_verdict = ClaudeVerdict(
                verdict="SKIPPED",
                reason_code="claude_api_unavailable",
                reasoning="",
                summary="🤖❌ Claude validation unavailable; manual review recommended.",
            )
            pick = self._persist_pending(
                prediction, opening_odds, selection, idx, edge, kelly_fraction, stake,
                skipped_verdict,
            )
            send_at = datetime.now(UTC)
            self._schedule_send(pick, prediction, send_at)
            return pick

        if verdict.verdict == "REJECT":
            return self._persist_rejected(
                prediction, opening_odds, selection, idx, edge, kelly_fraction, stake, verdict
            )

        pick = self._persist_pending(
            prediction, opening_odds, selection, idx, edge, kelly_fraction, stake, verdict
        )
        send_at = datetime.now(UTC)
        self._schedule_send(pick, prediction, send_at)
        return pick

    def _persist_filtered(self, prediction: Any, opening_odds: dict[str, Any], reason_code: str) -> Pick:
        pick = self._build_pick_skeleton(
            prediction, opening_odds,
            selection=None, idx=None, edge=0.0, kelly_fraction=None, stake=None,
            status=PickStatus.filtered,
        )
        pick.claude_reasoning = f"reason_code={reason_code}"
        self._pick_repo.insert(pick)
        logger.info(
            "pick_filtered", fixture_id=prediction.fixture_id, market=prediction.market,
            reason_code=reason_code,
        )
        return pick

    def _persist_rejected(self, prediction: Any, opening_odds: dict[str, Any], selection: str,
                          idx: int, edge: float, kelly_fraction: float, stake: float,
                          verdict: Any) -> Pick:
        pick = self._build_pick_skeleton(
            prediction, opening_odds,
            selection=selection, idx=idx, edge=edge,
            kelly_fraction=kelly_fraction, stake=stake,
            status=PickStatus.rejected,
        )
        pick.claude_validation = "REJECT"
        pick.claude_reasoning = verdict.reasoning
        pick.claude_summary = verdict.summary
        pick.claude_validated_at = datetime.now(UTC)
        self._pick_repo.insert(pick)
        logger.info(
            "pick_rejected", fixture_id=prediction.fixture_id, market=prediction.market,
            reason_code=verdict.reason_code,
        )
        return pick

    def _persist_pending(self, prediction: Any, opening_odds: dict[str, Any], selection: str,
                         idx: int, edge: float, kelly_fraction: float, stake: float,
                         verdict: Any) -> Pick:
        pick = self._build_pick_skeleton(
            prediction, opening_odds,
            selection=selection, idx=idx, edge=edge,
            kelly_fraction=kelly_fraction, stake=stake,
            status=PickStatus.pending,
        )
        pick.claude_validation = verdict.verdict
        pick.claude_reasoning = verdict.reasoning
        pick.claude_summary = verdict.summary
        pick.claude_validated_at = datetime.now(UTC)
        self._pick_repo.insert(pick)
        logger.info(
            "pick_persisted_pending", fixture_id=prediction.fixture_id, market=prediction.market,
            verdict=verdict.verdict,
        )
        return pick

    def _build_pick_skeleton(self, prediction: Any, opening_odds: dict[str, Any],
                             selection: str | None, idx: int | None, edge: float,
                             kelly_fraction: float | None, stake: float | None,
                             status: PickStatus) -> Pick:
        if selection is None:
            selection = "X"
            best_odds = 1.0
            implied = 1.0
            model_prob = 0.0
        else:
            best_odds = float(opening_odds[_SELECTION_KEYS[idx]])
            implied = 1.0 / best_odds
            model_prob = float(prediction.probabilities[selection])
        return Pick(
            prediction_id=getattr(prediction, "id", None),
            fixture_id=prediction.fixture_id,
            league=prediction.league,
            sport=prediction.sport,
            market=prediction.market,
            selection=selection,
            model_probability=model_prob,
            implied_probability=implied,
            edge=edge,
            best_odds=best_odds,
            bookmaker=str(opening_odds.get("bookmaker", "betano")),
            kelly_fraction=kelly_fraction,
            suggested_stake=stake,
            status=status,
        )

    def _build_pick_summary(self, prediction: Any, selection: str, odds: float,
                            edge: float, stake: float) -> str:
        return (
            f"{prediction.home_team} vs {prediction.away_team} "
            f"({prediction.league}, {prediction.market})\n"
            f"Selection: {selection} @ {odds:.2f}\n"
            f"Edge: {edge*100:.1f}% | Suggested stake: {stake:.1f}u\n"
            f"Model version: {prediction.model_version}"
        )

    def _build_curated_signals(self, prediction: Any) -> str:
        probs = prediction.probabilities
        return (
            "Curated signals (D-05):\n"
            f"- Model probabilities: 1={probs.get('1', 0):.2%} "
            f"X={probs.get('X', 0):.2%} 2={probs.get('2', 0):.2%}\n"
            f"- Kickoff (UTC): {prediction.kickoff_utc.isoformat()}\n"
            f"- Lineup adjusted: {prediction.is_lineup_adjusted}\n"
        )

    def _schedule_send(self, pick: Pick, prediction: Any, send_at: datetime) -> None:
        job_id = f"send_pick_{pick.fixture_id}_{pick.market}"
        self._scheduler.add_job(
            self._sender.send_pick,
            trigger=DateTrigger(run_date=send_at),
            kwargs={
                "pick": pick,
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "model_version": prediction.model_version,
                "reason_code": None,
            },
            id=job_id,
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info(
            "pick_send_scheduled", fixture_id=pick.fixture_id, market=pick.market,
            job_id=job_id, send_at=send_at.isoformat(),
        )
