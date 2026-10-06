"""DeliveryWorker — consume predictions_raw, gate, deliver (or shadow).

Sprint 2 Ola B. Polls predictions_raw for pending rows in a delivery
window, groups by fixture, calls the Claude validator ONCE per fixture
(spec §4.6 — not once per pick), applies the verdict + Kelly + safety
gates, and marks every row as 'sent', 'shadow', or 'killed'.

Shadow mode (default) writes status='shadow' and never invokes the
Telegram sender. Live mode (shadow_mode=False) sends + writes 'sent'.

Per memory `project_data_blocker`, this worker is wired against mocks in
tests. Real Anthropic / Supabase / Telegram integration is verified end-
to-end once those credentials are unblocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from bip.core.picks.staking import (
    quarter_kelly_units,
    round_to_nearest_half_unit,
)
from bip.pipeline.orchestrator import PREDICTIONS_RAW_TABLE
from bip.pipeline.protocols import (
    ClaudeValidatorProtocol,
    SupabaseClientProtocol,
    TelegramSenderProtocol,
)

log = structlog.get_logger(__name__)


# Delivery window — match_datetime ∈ (now + lower, now + upper)
DEFAULT_WINDOW_LOWER = timedelta(minutes=30)
DEFAULT_WINDOW_UPPER = timedelta(hours=4)

# Daily exposure cap as a fraction of bankroll (spec §4.7)
DEFAULT_DAILY_EXPOSURE_CAP = 0.15

# Max stake per pick as a fraction of bankroll (spec §4.7)
DEFAULT_MAX_STAKE_PER_PICK = 0.05


@dataclass
class DeliveryRunSummary:
    """Per-call counters and reasons."""

    n_pending_rows: int = 0
    n_fixtures_processed: int = 0
    n_validator_calls: int = 0
    n_sent: int = 0
    n_shadow: int = 0
    n_killed: int = 0
    n_errors: int = 0
    kill_reasons: dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _row_id(row: dict) -> str:
    return str(row.get("id", ""))


def _coerce_match_dt(row: dict) -> datetime:
    val = row.get("match_datetime")
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)


def _build_pick_summary(row: dict) -> str:
    """Compact one-line summary used as Claude input + Telegram body."""
    market = row.get("market", "?")
    selection = row.get("selection", "?")
    p = row.get("p_model")
    ev = row.get("ev")
    odds = row.get("odds_at_pick")
    home = row.get("home_team", "?")
    away = row.get("away_team", "?")
    parts = [
        f"{home} vs {away}",
        f"{market}/{selection}",
        f"p={float(p):.3f}" if p is not None else "p=n/a",
        f"odds={float(odds):.2f}" if odds is not None else "odds=n/a",
        f"ev={float(ev):+.3f}" if ev is not None else "ev=n/a",
    ]
    return " | ".join(parts)


def _build_curated_signals(rows: list[dict]) -> str:
    """Aggregate per-fixture context lines for the validator system prompt."""
    if not rows:
        return ""
    first = rows[0]
    lines = [
        f"competition={first.get('competition')}",
        f"home={first.get('home_team')} away={first.get('away_team')}",
        f"kickoff={first.get('match_datetime')}",
        f"n_picks_in_fixture={len(rows)}",
    ]
    for row in rows:
        payload = row.get("payload") or {}
        source = row.get("source")
        market = row.get("market")
        sel = row.get("selection")
        p = row.get("p_model")
        ev = row.get("ev")
        lines.append(
            f"- {source}/{market}/{sel} p={p} ev={ev} payload_keys={sorted(list(payload.keys()))[:5]}"
        )
    return "\n".join(lines)


def _format_telegram_text(row: dict, stake_units: float) -> str:
    """Plain-text body (no emojis by convention)."""
    source = (row.get("source") or "").upper()
    comp = row.get("competition", "")
    return (
        f"[{source}] {comp}\n"
        f"{row.get('home_team')} vs {row.get('away_team')}\n"
        f"Pick: {row.get('market')}/{row.get('selection')} "
        f"@ {row.get('odds_at_pick') or 'n/a'}\n"
        f"p_model: {row.get('p_model')}\n"
        f"Stake: {stake_units:.1f}u"
    )


def _verdict_field(verdict_obj: Any, name: str, default: Any = None) -> Any:
    """Read attr or key from a verdict-like object (pydantic or dict)."""
    if verdict_obj is None:
        return default
    if hasattr(verdict_obj, name):
        return getattr(verdict_obj, name)
    if isinstance(verdict_obj, dict):
        return verdict_obj.get(name, default)
    return default


# ──────────────────────────────────────────────────────────────────────
# DeliveryWorker
# ──────────────────────────────────────────────────────────────────────


class DeliveryWorker:
    """Consumes predictions_raw and dispatches picks (shadow or live)."""

    def __init__(
        self,
        *,
        supabase_client: SupabaseClientProtocol,
        validator: ClaudeValidatorProtocol | None,
        sender: TelegramSenderProtocol | None,
        shadow_mode: bool = True,
        ev_threshold: float = 0.03,
        daily_exposure_cap: float = DEFAULT_DAILY_EXPOSURE_CAP,
        max_stake_per_pick: float = DEFAULT_MAX_STAKE_PER_PICK,
        window_lower: timedelta = DEFAULT_WINDOW_LOWER,
        window_upper: timedelta = DEFAULT_WINDOW_UPPER,
        table_name: str = PREDICTIONS_RAW_TABLE,
    ) -> None:
        self._client = supabase_client
        self._validator = validator
        self._sender = sender
        self.shadow_mode = shadow_mode
        self.ev_threshold = float(ev_threshold)
        self.daily_exposure_cap = float(daily_exposure_cap)
        self.max_stake_per_pick = float(max_stake_per_pick)
        self.window_lower = window_lower
        self.window_upper = window_upper
        self._table_name = table_name

    async def run_once(
        self, *, now: datetime | None = None
    ) -> DeliveryRunSummary:
        """Single polling pass. Caller schedules this periodically."""
        now = now or datetime.now(UTC)
        summary = DeliveryRunSummary()

        pending = self._fetch_pending(now=now)
        summary.n_pending_rows = len(pending)
        by_fixture = self._group_by_fixture(pending)
        cumulative_exposure = 0.0

        for fixture_id, rows in by_fixture.items():
            summary.n_fixtures_processed += 1
            try:
                verdict = await self._call_validator(rows, summary=summary)
            except Exception as exc:  # noqa: BLE001
                summary.n_errors += 1
                log.warning(
                    "delivery_validator_error",
                    fixture_id=fixture_id,
                    error=str(exc),
                )
                verdict = None

            verdict_str = _verdict_field(verdict, "verdict", "SKIPPED")
            confidence_modifier = float(
                _verdict_field(verdict, "confidence_modifier", 0.0) or 0.0
            )

            if verdict_str == "REJECT":
                self._kill_all(rows, summary, reason="claude_reject")
                continue

            for row in rows:
                stake_units, kill_reason = self._compute_stake_or_kill_reason(
                    row,
                    confidence_modifier=confidence_modifier,
                    cumulative_exposure=cumulative_exposure,
                )
                if kill_reason is not None:
                    self._mark_killed(row, reason=kill_reason, summary=summary)
                    continue

                cumulative_exposure += stake_units / 100.0  # stake as fraction of 100u bankroll

                if self.shadow_mode:
                    self._mark_shadow(row, stake_units=stake_units, summary=summary)
                else:
                    sent_ok = await self._send_telegram(row, stake_units=stake_units)
                    if sent_ok:
                        self._mark_sent(row, stake_units=stake_units, summary=summary)
                    else:
                        self._mark_killed(row, reason="telegram_failure", summary=summary)

        log.info(
            "delivery_worker_run_complete",
            n_pending=summary.n_pending_rows,
            n_sent=summary.n_sent,
            n_shadow=summary.n_shadow,
            n_killed=summary.n_killed,
            n_errors=summary.n_errors,
            shadow_mode=self.shadow_mode,
        )
        return summary

    # ------------------------------------------------------------------
    # IO + helpers
    # ------------------------------------------------------------------

    def _fetch_pending(self, *, now: datetime) -> list[dict]:
        lower = (now + self.window_lower).isoformat()
        upper = (now + self.window_upper).isoformat()
        query = (
            self._client.table(self._table_name)
            .select("*")
            .eq("status", "pending")
            .gte("match_datetime", lower)
            .lte("match_datetime", upper)
        )
        resp = query.execute()
        return list(getattr(resp, "data", None) or [])

    @staticmethod
    def _group_by_fixture(rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in rows:
            key = str(r.get("fixture_id", ""))
            out.setdefault(key, []).append(r)
        return out

    async def _call_validator(
        self, rows: list[dict], *, summary: DeliveryRunSummary
    ) -> Any:
        if self._validator is None:
            return None
        summary.n_validator_calls += 1
        pick_summary = _build_pick_summary(rows[0])  # spec §4.6: one call per fixture
        curated = _build_curated_signals(rows)
        return await self._validator.validate(
            pick_summary=pick_summary, curated_signals=curated
        )

    def _compute_stake_or_kill_reason(
        self,
        row: dict,
        *,
        confidence_modifier: float,
        cumulative_exposure: float,
    ) -> tuple[float, str | None]:
        """Apply confidence modifier, EV gate, Kelly, and exposure cap.

        Returns (stake_units, kill_reason). When kill_reason is not None,
        stake_units is 0.
        """
        ev = row.get("ev")
        odds = row.get("odds_at_pick")

        if ev is None or odds is None:
            return 0.0, "missing_ev_or_odds"

        ev_adjusted = float(ev) + confidence_modifier
        if ev_adjusted < self.ev_threshold:
            return 0.0, "ev_below_threshold"

        # Quarter Kelly (staking)
        kelly_fraction = quarter_kelly_units(edge=ev_adjusted, odds=float(odds))
        if kelly_fraction <= 0:
            return 0.0, "kelly_zero"

        # Cap per-pick fraction
        safe_fraction = min(kelly_fraction, self.max_stake_per_pick)

        # Convert to units on a notional 100-unit bankroll, round to 0.5u
        stake_units = round_to_nearest_half_unit(safe_fraction * 100.0)
        if stake_units <= 0:
            return 0.0, "stake_rounded_to_zero"

        # Daily exposure cap check
        projected = cumulative_exposure + safe_fraction
        if projected > self.daily_exposure_cap:
            return 0.0, "daily_exposure_cap"

        return stake_units, None

    def _kill_all(
        self, rows: list[dict], summary: DeliveryRunSummary, *, reason: str
    ) -> None:
        for row in rows:
            self._mark_killed(row, reason=reason, summary=summary)

    def _mark_killed(self, row: dict, *, reason: str, summary: DeliveryRunSummary) -> None:
        self._update_status(row, payload={"status": "killed", "kill_reason": reason})
        summary.n_killed += 1
        summary.kill_reasons[reason] = summary.kill_reasons.get(reason, 0) + 1

    def _mark_shadow(self, row: dict, *, stake_units: float, summary: DeliveryRunSummary) -> None:
        self._update_status(
            row,
            payload={
                "status": "shadow",
                "stake_units": stake_units,
            },
        )
        summary.n_shadow += 1

    def _mark_sent(self, row: dict, *, stake_units: float, summary: DeliveryRunSummary) -> None:
        self._update_status(
            row,
            payload={
                "status": "sent",
                "stake_units": stake_units,
                "sent_at": datetime.now(UTC).isoformat(),
            },
        )
        summary.n_sent += 1

    def _update_status(self, row: dict, *, payload: dict) -> None:
        try:
            (
                self._client.table(self._table_name)
                .update(payload)
                .eq("id", _row_id(row))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "delivery_update_error",
                row_id=_row_id(row),
                error=str(exc),
            )

    async def _send_telegram(self, row: dict, *, stake_units: float) -> bool:
        if self._sender is None:
            return False
        try:
            await self._sender.send_pick(text=_format_telegram_text(row, stake_units))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "delivery_telegram_error",
                row_id=_row_id(row),
                error=str(exc),
            )
            return False


__all__ = ["DeliveryRunSummary", "DeliveryWorker"]
