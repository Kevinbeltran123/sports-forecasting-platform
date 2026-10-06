"""Live runner — orchestrates state poll + residual predict + alert emit.

Architecture:
  1. Pull current fixture state via API-Football
  2. Build MatchState
  3. Compute live market probabilities (residual_predictor)
  4. Pull fresh odds (Pinnacle) — every poll, since live odds shift
  5. Run value_detector across live markets
  6. Filter alerts through AlertHistory rate-limit
  7. Format + send via Telegram bot v2

Designed for one fixture per ``LiveRunner`` instance. The caller is
expected to spawn one runner per concurrent live match.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    MarketPredictions,
)
from bip.evaluation.tournaments.team_style_profiler.live.alert_state import (
    AlertHistory,
    should_alert,
)
from bip.evaluation.tournaments.team_style_profiler.live.match_state import (
    MatchState,
    build_match_state,
    next_poll_seconds,
)
from bip.evaluation.tournaments.team_style_profiler.live.residual_predictor import (
    LiveMarketPredictions,
    LiveMarketProb,
    predict_live_markets,
)
from bip.evaluation.tournaments.team_style_profiler.odds_provider import (
    extract_pinnacle_canonical_odds,
    implied_probability,
)
from bip.evaluation.tournaments.team_style_profiler.value_detector import (
    MIN_Z_SCORE,
    ACCEPTED_CONFIDENCE,
    ValueAlert,
)

logger = logging.getLogger(__name__)


# ─── Bot interface protocol (so we can inject the existing bot v2) ──────


class BotProtocol(Protocol):
    async def send_html(self, text: str) -> None: ...


# ─── API-Football client interface protocol ─────────────────────────────


class ApiClientProtocol(Protocol):
    async def get_fixture_by_id(self, fixture_id: int) -> dict: ...
    async def get_fixture_events(self, fixture_id: int) -> dict: ...
    async def get_odds(
        self, fixture_id: int | None = None, **kwargs
    ) -> dict: ...


# ─── Configuration ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveRunnerConfig:
    fixture_id: int
    pre_match_predictions: MarketPredictions
    pre_match_cards_lambda_total: float | None = None
    """Sum of both teams' yellow_cards_per_match means at pre-match
    time — used to scale cards residual. Pass None to disable cards."""
    rho: float = 0.0
    max_polls: int | None = None
    """If set, stop after this many polls (used for testing). None
    means run until fixture is finished."""


# ─── Alert formatting ───────────────────────────────────────────────────


def format_alert_html(
    alert: ValueAlert,
    state: MatchState,
    re_alert_reason: str | None = None,
) -> str:
    """Build the HTML message body sent to Telegram."""
    edge = alert.edge_pct or 0.0
    z = alert.z_score or 0.0
    odd_str = f"{alert.decimal_odd:.2f}" if alert.decimal_odd else "?"
    score = f"{state.home_team} {state.home_goals} - {state.away_goals} {state.away_team}"

    re_alert_line = ""
    if re_alert_reason is not None:
        re_alert_line = f"\n<i>{re_alert_reason}</i>"

    return (
        f"<b>LIVE EDGE</b> @ min {state.minute}\n"
        f"<i>{score}</i>\n"
        f"\n"
        f"Market: <b>{alert.market}</b>\n"
        f"Pinnacle cuota: {odd_str} (implied {alert.p_implied:.1%})\n"
        f"TSP empirical: {alert.p_empirical:.1%}\n"
        f"Edge: <b>{edge:+.2%}</b>\n"
        f"Z-score: {z:.2f} (gate {MIN_Z_SCORE})\n"
        f"Confidence: {alert.confidence}"
        f"{re_alert_line}"
    )


# ─── Z-score recomputation for live markets ─────────────────────────────


def _std_from_live(prob: LiveMarketProb) -> float:
    """Same approximation as value_detector._std_from_market but for
    LiveMarketProb (kept separate to avoid coupling)."""
    half_widths = {"HIGH": 0.05, "MEDIUM": 0.10, "LOW": 0.15}
    return half_widths.get(prob.confidence, 0.15) / 1.96


def _evaluate_live_market(
    live_prob: LiveMarketProb,
    odds: dict[str, float],
) -> ValueAlert:
    decimal_odd = odds.get(live_prob.market)
    std_emp = _std_from_live(live_prob)
    p_emp = live_prob.probability

    if decimal_odd is None:
        return ValueAlert(
            market=live_prob.market, decision="SKIP_NO_ODDS",
            p_empirical=p_emp, p_implied=0.0, std_empirical=std_emp,
            decimal_odd=None, z_score=None, edge_pct=None,
            confidence=live_prob.confidence,
            rationale=f"No Pinnacle odd for {live_prob.market}",
        )

    p_implied = implied_probability(decimal_odd)
    z = (p_emp - p_implied) / std_emp if std_emp > 0 else 0.0
    edge = p_emp * decimal_odd - 1.0

    if live_prob.confidence not in ACCEPTED_CONFIDENCE:
        decision = "SKIP_CONFIDENCE"
        rationale = f"confidence={live_prob.confidence}"
    elif z < MIN_Z_SCORE:
        decision = "SKIP_EDGE"
        rationale = f"Z={z:.2f} < {MIN_Z_SCORE}, edge={edge:.2%}"
    else:
        decision = "ALERT"
        rationale = (
            f"Z={z:.2f} >= {MIN_Z_SCORE}, P_emp={p_emp:.3f} "
            f"vs P_implied={p_implied:.3f} (cuota {decimal_odd:.2f}). "
            f"{live_prob.rationale}"
        )

    return ValueAlert(
        market=live_prob.market, decision=decision,
        p_empirical=p_emp, p_implied=p_implied, std_empirical=std_emp,
        decimal_odd=decimal_odd, z_score=z, edge_pct=edge,
        confidence=live_prob.confidence, rationale=rationale,
    )


# ─── Runner ─────────────────────────────────────────────────────────────


@dataclass
class LiveRunner:
    """One async run-loop per fixture. Stops when match finishes or
    max_polls reached."""

    config: LiveRunnerConfig
    client: ApiClientProtocol
    bot: BotProtocol
    history: AlertHistory = field(default_factory=AlertHistory)
    sleep_fn: Callable[[float], "asyncio.Future"] = field(default=asyncio.sleep)  # type: ignore[assignment]

    async def run(self) -> list[ValueAlert]:
        """Drive the poll loop. Returns list of emitted ALERT alerts."""
        emitted: list[ValueAlert] = []
        poll_count = 0
        while True:
            poll_count += 1
            if self.config.max_polls is not None and poll_count > self.config.max_polls:
                logger.info("live_runner_max_polls_reached", count=poll_count)
                break

            state = await self._poll_state()
            if state is None:
                # Couldn't fetch state — back off and retry
                await self.sleep_fn(30)
                continue

            if state.is_finished:
                logger.info("live_runner_match_finished", fixture_id=state.fixture_id)
                break

            if state.is_pre_match:
                # No live edges to compute pre-match — we wait
                await self.sleep_fn(next_poll_seconds(state))
                continue

            live_preds = predict_live_markets(
                self.config.pre_match_predictions,
                state,
                pre_match_cards_lambda_total=self.config.pre_match_cards_lambda_total,
                rho=self.config.rho,
            )

            # Pull live odds
            odds = await self._fetch_pinnacle_odds()

            # Evaluate every market
            for live_prob in live_preds.all_markets:
                alert = _evaluate_live_market(live_prob, odds)
                if alert.decision != "ALERT":
                    continue
                ok, reason = should_alert(
                    self.history, self.config.fixture_id, alert.market,
                    alert.edge_pct or 0.0,
                )
                if not ok:
                    continue
                # Emit
                msg = format_alert_html(
                    alert, state,
                    re_alert_reason=reason if "grew" in reason else None,
                )
                try:
                    await self.bot.send_html(msg)
                    self.history.record(
                        self.config.fixture_id, alert.market,
                        edge_pct=alert.edge_pct or 0.0,
                        z_score=alert.z_score or 0.0,
                    )
                    emitted.append(alert)
                except Exception as exc:  # noqa: BLE001
                    logger.error("live_runner_bot_failed", error=str(exc))

            await self.sleep_fn(next_poll_seconds(state))
        return emitted

    async def _poll_state(self) -> MatchState | None:
        try:
            fix_payload = await self.client.get_fixture_by_id(self.config.fixture_id)
            evt_payload = await self.client.get_fixture_events(self.config.fixture_id)
            return build_match_state(fix_payload, evt_payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("live_runner_state_fetch_failed", error=str(exc))
            return None

    async def _fetch_pinnacle_odds(self) -> dict[str, float]:
        try:
            payload = await self.client.get_odds(fixture_id=self.config.fixture_id)
            return extract_pinnacle_canonical_odds(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("live_runner_odds_fetch_failed", error=str(exc))
            return {}
