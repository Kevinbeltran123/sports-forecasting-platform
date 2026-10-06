"""WC2026 Pattern Runner — pipeline integration for the validated rules.

Wires the four pure rules in ``wc2026_patterns`` into a stateful runner
that the live engine can call per fixture-frame. The runner:

1. Loads MatchContext per fixture from the locked-predictions JSON.
2. Joins squad market values from the Transfermarkt parquet.
3. Evaluates all four rules on every (fixture, odds_snapshot) event.
4. Emits ``PatternPick`` per trigger and dedupes by
   ``(fixture_id, rule_id, market_family, direction)``.

Output is **shadow-mode**: the recorder writes to
``data/cache/wc2026_patterns/pattern_picks.parquet`` (date-partitioned).
Wiring to the Telegram alert path is left to the caller — the runner
exposes ``PatternPick`` objects and the alert composer can decide which
rule IDs go live.

Lifecycle (per process):

    runner = WC2026PatternRunner.from_lock_json()
    for fixture_event in live_stream:
        odds = fixture_event.to_snapshot()
        picks = runner.evaluate(fixture_event.match_id, odds)
        recorder.record(picks)
        for pick in picks:
            if pick.rule_id == "A1":  # operator decides which to alert
                telegram.send(...)
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_avoids import (
    AvoidVerdict,
    LineMoveContext,
    is_blocked,
    run_avoids,
)
from bip.evaluation.live.engine_v3.wc2026_bias_flags import (
    BiasFlag,
    LockPrediction,
    flag_b1_strong_favorite_vs_debutant,
    flag_b2_lock_tilts_against_elite,
    flag_b3_post_lock_events,
)
from bip.evaluation.live.engine_v3.wc2026_contingencies import (
    ContingencyVerdict,
    FriendlyResult,
    LineupSnapshot,
    MD1Result,
    StakeAdjustment,
    run_contingencies,
    worst_adjustment,
)
from bip.evaluation.live.engine_v3.wc2026_patterns import (
    MatchContext,
    OddsSnapshot,
    PatternTrigger,
    enrich_context,
    load_market_values,
    rule_a1_ht00_live_u25,
    rule_a2_favorite_corner_tilt,
    rule_a3_favorite_team_total,
    rule_a4_afcon_parity,
)

_logger = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parents[5]
    / "src" / "bip" / "evaluation" / "tournaments" / "locked_predictions"
    / "world_cup_2026" / "lock.json"
)

DEFAULT_PATTERN_OUTPUT_ROOT = Path("data/cache/wc2026_patterns")


@dataclass(frozen=True)
class PatternPick:
    """One pattern-rule-driven candidate pick. Shadow-mode by default;
    promotion to live alerts is the caller's responsibility.

    ``avoid_verdicts`` carries any C-rule that blocked the pick. When
    non-empty, callers should treat the pick as suppressed.

    ``contingency_adjustment`` carries the worst D-rule adjustment so
    callers can scale the recommended stake (REDUCE_50 halves it,
    CANCEL drops it).
    """

    fixture_id: str
    timestamp_utc: datetime
    rule_id: str
    home_team: str
    away_team: str
    tournament_slug: str
    phase: str
    market_family: MarketFamily
    direction: str
    decimal_odds: float
    max_kelly: float
    reason: str
    breakeven_odds: float | None = None
    home_market_value_eur: float | None = None
    away_market_value_eur: float | None = None
    avoid_verdicts: tuple[AvoidVerdict, ...] = field(default_factory=tuple)
    contingency_adjustment: StakeAdjustment = StakeAdjustment.HOLD
    blocked: bool = False

    @property
    def effective_kelly(self) -> float:
        """Kelly fraction after applying contingency adjustment + block."""
        if self.blocked or self.contingency_adjustment == StakeAdjustment.CANCEL:
            return 0.0
        if self.contingency_adjustment == StakeAdjustment.REDUCE_50:
            return self.max_kelly * 0.5
        return self.max_kelly

    def to_row(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "timestamp_utc": self.timestamp_utc,
            "rule_id": self.rule_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "tournament_slug": self.tournament_slug,
            "phase": self.phase,
            "market_family": self.market_family.value,
            "direction": self.direction,
            "decimal_odds": float(self.decimal_odds),
            "max_kelly": float(self.max_kelly),
            "effective_kelly": float(self.effective_kelly),
            "reason": self.reason,
            "breakeven_odds": (
                float(self.breakeven_odds) if self.breakeven_odds is not None else None
            ),
            "home_market_value_eur": self.home_market_value_eur,
            "away_market_value_eur": self.away_market_value_eur,
            "blocked": bool(self.blocked),
            "contingency_adjustment": self.contingency_adjustment.value,
            "avoid_block_reasons": " | ".join(
                f"{v.rule_id}:{v.reason}" for v in self.avoid_verdicts if v.blocked
            ),
        }


@dataclass(frozen=True)
class FixtureContingencyInputs:
    """Optional inputs supplied by the operator/feed per fixture."""

    lock_prediction: LockPrediction | None = None
    home_lineup: LineupSnapshot | None = None
    away_lineup: LineupSnapshot | None = None
    home_md1: MD1Result | None = None
    away_md1: MD1Result | None = None
    home_friendly: FriendlyResult | None = None
    away_friendly: FriendlyResult | None = None
    line_move: LineMoveContext | None = None


@dataclass(frozen=True)
class FixtureFlags:
    """Bias flags + contingency verdicts for one fixture. Cached after
    first evaluation per fixture-frame to avoid recomputation when
    multiple odds snapshots arrive."""

    bias_flags: tuple[BiasFlag, ...]
    contingency_verdicts: tuple[ContingencyVerdict, ...]
    worst_adjust: StakeAdjustment


# Stage-aware rule dispatch. A4 is a filter; A2 and A3 invoke it
# internally via _gate_a2_a3, so it does not appear as an emitting rule.
_RULES_PRE_MATCH = (
    rule_a2_favorite_corner_tilt,
    rule_a3_favorite_team_total,
)
_RULES_LIVE = (rule_a1_ht00_live_u25,)


class WC2026PatternRunner:
    """Stateful runner with per-fixture MatchContext + dedup."""

    def __init__(
        self,
        contexts: dict[str, MatchContext] | None = None,
        *,
        market_values: dict[str, float] | None = None,
    ) -> None:
        self.contexts: dict[str, MatchContext] = dict(contexts or {})
        self.market_values: dict[str, float] = dict(market_values or {})
        # Dedup keyed by (fixture_id, rule_id, market_family.value, direction).
        # Re-firing of the same rule on the same market is suppressed
        # within process lifetime, matching the V3Pipeline trio-dedup
        # semantics.
        self._emitted: set[tuple[str, str, str, str]] = set()
        # Per-fixture B+D evaluation is expensive (registry walk, lineup
        # check, etc) and ALL its inputs are fixture-scoped, not odds-
        # scoped. Cache once per fixture per operator-set contingency
        # input bundle.
        self._contingency_inputs: dict[str, FixtureContingencyInputs] = {}
        self._flag_cache: dict[str, FixtureFlags] = {}

    @classmethod
    def from_lock_json(
        cls,
        lock_path: Path = DEFAULT_LOCK_PATH,
        *,
        market_values_path: Path | None = None,
    ) -> "WC2026PatternRunner":
        """Build a runner with all WC2026 fixtures pre-registered.

        Reads the lock JSON's fixtures and joins Transfermarkt values.
        Fixtures in the lock are group-stage only; knockout matches will
        need a separate ingestion when the bracket resolves.
        """
        if not lock_path.exists():
            raise FileNotFoundError(f"lock not found at {lock_path}")
        if market_values_path is None:
            mvs = load_market_values()
        else:
            mvs = load_market_values(market_values_path)

        with open(lock_path) as fh:
            lock = json.load(fh)

        tournament_slug = lock.get("tournament_slug", "world_cup_2026")
        contexts: dict[str, MatchContext] = {}
        for fx in lock.get("fixtures", []):
            ctx_partial = {
                "match_id": str(fx["match_id"]),
                "tournament_slug": tournament_slug,
                "home_team": fx["home_team_name"],
                "away_team": fx["away_team_name"],
                "phase": fx.get("tournament_phase", "group"),
            }
            contexts[str(fx["match_id"])] = enrich_context(ctx_partial, mvs)

        return cls(contexts=contexts, market_values=mvs)

    def register_fixture(self, ctx: MatchContext) -> None:
        """Add or replace a single fixture context. Resets dedup +
        flag cache for that fixture."""
        self.contexts[ctx.match_id] = ctx
        self._emitted = {key for key in self._emitted if key[0] != ctx.match_id}
        self._flag_cache.pop(ctx.match_id, None)

    def set_contingency_inputs(
        self,
        fixture_id: str,
        inputs: FixtureContingencyInputs,
    ) -> None:
        """Attach lock prediction + lineups + MD1 results + friendlies to
        a fixture. Invalidates the flag cache for that fixture."""
        self._contingency_inputs[fixture_id] = inputs
        self._flag_cache.pop(fixture_id, None)

    def _compute_flags(self, fixture_id: str) -> FixtureFlags:
        cached = self._flag_cache.get(fixture_id)
        if cached is not None:
            return cached
        ctx = self.contexts[fixture_id]
        inputs = self._contingency_inputs.get(fixture_id, FixtureContingencyInputs())

        bias_flags: list[BiasFlag] = []
        if inputs.lock_prediction is not None:
            bias_flags.append(
                flag_b1_strong_favorite_vs_debutant(ctx, inputs.lock_prediction)
            )
            bias_flags.append(
                flag_b2_lock_tilts_against_elite(ctx, inputs.lock_prediction)
            )
        bias_flags.append(flag_b3_post_lock_events(ctx))

        contingencies = run_contingencies(
            ctx,
            home_lineup=inputs.home_lineup,
            away_lineup=inputs.away_lineup,
            home_md1=inputs.home_md1,
            away_md1=inputs.away_md1,
            home_friendly=inputs.home_friendly,
            away_friendly=inputs.away_friendly,
        )
        worst = worst_adjustment(contingencies)
        flags = FixtureFlags(
            bias_flags=tuple(bias_flags),
            contingency_verdicts=tuple(contingencies),
            worst_adjust=worst,
        )
        self._flag_cache[fixture_id] = flags
        return flags

    def flags_for(self, fixture_id: str) -> FixtureFlags | None:
        """Return cached B+D verdicts for a fixture; compute on first call."""
        if fixture_id not in self.contexts:
            return None
        return self._compute_flags(fixture_id)

    def reset_dedupe(self, fixture_id: str | None = None) -> None:
        """Clear dedup state. ``fixture_id=None`` clears all fixtures."""
        if fixture_id is None:
            self._emitted.clear()
        else:
            self._emitted = {key for key in self._emitted if key[0] != fixture_id}

    def evaluate(
        self,
        fixture_id: str,
        odds: OddsSnapshot,
        *,
        now_utc: datetime | None = None,
    ) -> list[PatternPick]:
        """Run all stage-appropriate rules; return triggered picks (post-dedup).

        Each emitted pick is annotated with:
        - C-rule blocks (if any avoid rule triggered)
        - D-rule contingency adjustment (HOLD/REDUCE_50/CANCEL)
        - B-rule bias flags can be queried separately via ``flags_for(fixture_id)``
        """
        ctx = self.contexts.get(fixture_id)
        if ctx is None:
            _logger.warning(
                "wc2026_pattern_runner: fixture %s not registered", fixture_id
            )
            return []

        rules = _RULES_LIVE if odds.stage == "live" else _RULES_PRE_MATCH
        triggers: list[PatternTrigger] = [rule(ctx, odds) for rule in rules]

        # B+D flags are fixture-scoped; cache once per fixture.
        flags = self._compute_flags(fixture_id)
        inputs = self._contingency_inputs.get(fixture_id, FixtureContingencyInputs())

        # C-rule avoid pass: needs the current odds snapshot.
        avoid_verdicts = tuple(
            run_avoids(
                ctx, odds,
                lock=inputs.lock_prediction,
                line_move=inputs.line_move,
            )
        )
        any_block = is_blocked(list(avoid_verdicts))

        ts = now_utc or datetime.now(timezone.utc)
        picks: list[PatternPick] = []
        for trig in triggers:
            if not trig.triggered or trig.market_family is None or trig.direction is None:
                continue
            key = (
                fixture_id,
                trig.rule_id,
                trig.market_family.value,
                trig.direction,
            )
            if key in self._emitted:
                continue
            picks.append(
                PatternPick(
                    fixture_id=fixture_id,
                    timestamp_utc=ts,
                    rule_id=trig.rule_id,
                    home_team=ctx.home_team,
                    away_team=ctx.away_team,
                    tournament_slug=ctx.tournament_slug,
                    phase=ctx.phase,
                    market_family=trig.market_family,
                    direction=trig.direction,
                    decimal_odds=odds.decimal_odds,
                    max_kelly=trig.max_kelly,
                    reason=trig.reason,
                    breakeven_odds=trig.breakeven_odds,
                    home_market_value_eur=ctx.home_market_value_eur,
                    away_market_value_eur=ctx.away_market_value_eur,
                    avoid_verdicts=avoid_verdicts,
                    contingency_adjustment=flags.worst_adjust,
                    blocked=any_block,
                )
            )
            self._emitted.add(key)
        return picks

    def afcon_filter_status(self, fixture_id: str) -> PatternTrigger | None:
        """Expose A4's filter verdict for the fixture (audit only)."""
        ctx = self.contexts.get(fixture_id)
        if ctx is None:
            return None
        return rule_a4_afcon_parity(ctx)


class PatternPickRecorder:
    """Date-partitioned parquet sink for PatternPick. Mirrors the
    LineRecorder shape used elsewhere in engine_v3."""

    def __init__(self, output_root: Path = DEFAULT_PATTERN_OUTPUT_ROOT) -> None:
        self.output_root = output_root
        self._buffer: list[dict[str, Any]] = []

    def record(self, picks: Iterable[PatternPick]) -> None:
        """Buffer one or more picks. ``flush()`` writes to disk."""
        for pick in picks:
            self._buffer.append(pick.to_row())

    def flush(self, *, ts: datetime | None = None) -> Path | None:
        """Write buffered rows to the date partition. Returns the path
        or None if there was nothing to write."""
        if not self._buffer:
            return None
        ts = ts or datetime.now(timezone.utc)
        partition = ts.strftime("dt=%Y-%m-%d")
        out_dir = self.output_root / partition
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "pattern_picks.parquet"
        new_rows = pl.DataFrame(self._buffer)
        if path.exists():
            existing = pl.read_parquet(path)
            combined = pl.concat([existing, new_rows], how="diagonal_relaxed")
        else:
            combined = new_rows
        combined.write_parquet(path)
        self._buffer.clear()
        return path

    def __len__(self) -> int:
        return len(self._buffer)


__all__ = [
    "DEFAULT_LOCK_PATH",
    "DEFAULT_PATTERN_OUTPUT_ROOT",
    "FixtureContingencyInputs",
    "FixtureFlags",
    "PatternPick",
    "PatternPickRecorder",
    "WC2026PatternRunner",
]
