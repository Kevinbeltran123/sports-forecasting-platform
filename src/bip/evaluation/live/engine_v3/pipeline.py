"""End-to-end V3 shadow pipeline.

Wires together the components defined in this package:

    LiveMatchState + priors + market_snapshot
        ↓ GSVBuilder
    GameStateVector
        ↓ generate_theses (rule layer)
    List[Thesis]
        ↓ select_markets (MES + family routing) + conditional predictor
    List[MarketCandidate]
        ↓ run_gate (8 no-bet rules)
    List[GateResult]   (allowed + denied; reasons logged)
        ↓ ShadowPick synthesizer
    List[ShadowPick]   (audit-only, never sent to Telegram in Phase 1)

This is **shadow-mode** (sec 7.5 + Phase 2 of sec 10): the output goes
to a log / parquet sink, not to the operator. The whole point is to
let the v3 run alongside the current system for 2-4 weeks and compare
ROIs cohort-against-cohort without risk.

The pipeline is a single class so callers can wire dependencies once
and call ``run(...)`` per fixture-frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from bip.evaluation.live.engine_v3.archetypes import generate_theses
from bip.evaluation.live.engine_v3.conditional_predictor import (
    ConditionalPredictor,
    make_fair_prob_provider,
)
from bip.evaluation.live.engine_v3.pattern_layer import (
    PatternLayer,
    generate_theses_hybrid,
)
from bip.evaluation.live.engine_v3.gsv import (
    CriticalEvent,
    GameStateVector,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.evaluation.live.engine_v3.gsv_builder import GSVBuilder
from bip.evaluation.live.engine_v3.market_selector import (
    MarketCandidate,
    select_markets,
)
from bip.evaluation.live.engine_v3.mispricing_window import (
    MispricingWindowConfig,
    WindowResult,
    classify_gsv,
)
from bip.evaluation.live.engine_v3.no_bet_gate import GateResult, run_gate
from bip.evaluation.live.engine_v3.ood_detector import OODDetector
from bip.evaluation.live.engine_v3.thesis import Thesis
from bip.evaluation.live.match_state import LiveMatchState

# Phase-3 surfaces. Imported lazily-as-types to keep Phase-1/2 callers
# (which never wire a promoter) free of any phase3 import overhead.
try:  # pragma: no cover — type-only import path
    from bip.evaluation.live.engine_v3.phase3.tier_promoter import (
        PromotedPick,
        TierDPromoter,
    )
except ImportError:  # pragma: no cover
    PromotedPick = None  # type: ignore[assignment]
    TierDPromoter = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.drift_monitor import (
        CalibrationDriftMonitor,
    )


@dataclass(frozen=True)
class ShadowPick:
    """The minimal audit-trail entry per allowed candidate.

    Phase 2 will extend this with the predictor's fair_prob CI, the
    operator's Kelly fraction, and the calibrated isotonic bucket."""

    fixture_id: int
    timestamp_utc: datetime
    candidate: MarketCandidate
    full_thesis: Thesis = field(repr=False)


@dataclass(frozen=True)
class PipelineOutput:
    """All outputs of one frame, for audit + telemetry.

    Sec 7.2 requires that **every** rejected candidate be logged with
    its rule number. ``gate_results`` carries that.

    ``promoted_picks`` is populated only when a Phase-3 ``TierDPromoter``
    is wired to the pipeline. Otherwise it stays empty — Phase 1/2
    callers see no behaviour change.
    """

    gsv: GameStateVector
    theses: list[Thesis]
    candidates: list[MarketCandidate]
    gate_results: list[GateResult]
    allowed_picks: list[ShadowPick]
    promoted_picks: list = field(default_factory=list)
    mispricing_window: WindowResult | None = None
    deduped_count: int = 0


class V3Pipeline:
    """One pipeline instance per process. Holds a stateful GSVBuilder
    (for ``state_version``) and reusable components.

    Emission dedup: ``_emitted_trios`` tracks ``{fixture_id: {(arch, mkt_id)}}``
    so a thesis re-firing across consecutive frames does not produce
    duplicate "picks". A trio is "one bet" in operator terms — the
    parquet sink and Telegram delivery downstream both count rows, so
    re-emitting 11× the same A12 cruise → match_goals_under_X.5 trio
    inflates P/L 11× when graded raw. Day-4 (2026-05-13) measured this
    inflation directly: 271 raw vs 45 unique trios.

    Reset semantics: dedup is permanent within process lifetime per
    fixture. If a condition re-emerges after a critical event resets
    the state (goal, red card), we deliberately do NOT re-emit — the
    line will have moved and an operator would have either already
    taken the bet or moved on. ``deduped`` is exposed on
    ``PipelineOutput`` for telemetry.
    """

    def __init__(
        self,
        *,
        gsv_builder: GSVBuilder | None = None,
        conditional_predictor: ConditionalPredictor | None = None,
        tier_promoter: "TierDPromoter | None" = None,
        ood_detector: OODDetector | None = None,
        pattern_layer: PatternLayer | None = None,
        mispricing_window_cfg: MispricingWindowConfig | None = None,
        drift_monitor: "CalibrationDriftMonitor | None" = None,
        mes_threshold: float = 0.6,
        target_stake: float = 100.0,
        line_max_age_sec: float | None = None,
        commentary_required: bool = False,
        uncertainty_band: float = 0.08,
        top_k_per_thesis: int = 3,
        dedupe_emissions: bool = True,
    ) -> None:
        self.gsv_builder = gsv_builder or GSVBuilder()
        self.predictor = conditional_predictor or ConditionalPredictor.default()
        self.tier_promoter = tier_promoter
        self.ood_detector = ood_detector
        self.pattern_layer = pattern_layer
        self.mispricing_window_cfg = mispricing_window_cfg or MispricingWindowConfig()
        self.drift_monitor = drift_monitor
        self.mes_threshold = mes_threshold
        self.target_stake = target_stake
        self.line_max_age_sec = line_max_age_sec
        self.commentary_required = commentary_required
        self.uncertainty_band = uncertainty_band
        self.top_k_per_thesis = top_k_per_thesis
        self.dedupe_emissions = dedupe_emissions
        self._emitted_trios: dict[int, set[tuple[str, str]]] = {}

    def reset_dedupe(self, fixture_id: int | None = None) -> None:
        """Clear the dedup set. ``fixture_id=None`` clears all fixtures.

        Tests use this to assert behaviour without instantiating a fresh
        pipeline. Production should not call it; per-process lifetime is
        the contract."""
        if fixture_id is None:
            self._emitted_trios.clear()
        else:
            self._emitted_trios.pop(fixture_id, None)

    def run(
        self,
        state: LiveMatchState,
        *,
        priors: PreMatchPriors,
        markets: MarketSnapshot,
        dominant_team_id: int | None = None,
        ref_card_rate_prior: float = 0.0,
        last_critical_event: CriticalEvent | None = None,
        last_critical_event_age_sec: float | None = None,
        now_utc: datetime | None = None,
    ) -> PipelineOutput:
        """Run one frame end-to-end."""
        gsv = self.gsv_builder.build(
            state,
            priors=priors,
            markets=markets,
            dominant_team_id=dominant_team_id,
            ref_card_rate_prior=ref_card_rate_prior,
            last_critical_event=last_critical_event,
            last_critical_event_age_sec=last_critical_event_age_sec,
            now_utc=now_utc,
        )
        return self.run_on_gsv(gsv)

    def run_on_gsv(self, gsv: GameStateVector) -> PipelineOutput:
        """Replay-entry point that consumes a pre-built GSV.

        Bypasses ``GSVBuilder``. Useful for: (1) historical replays
        against ``gsv_log.parquet`` where the original LiveMatchState is
        no longer available, (2) testing with hand-constructed GSVs.

        The dedup state is still tracked per fixture so a replay over N
        frames of the same fixture matches the production behaviour."""
        if self.pattern_layer is not None and self.pattern_layer.is_fitted:
            theses = generate_theses_hybrid(
                gsv,
                pattern_layer=self.pattern_layer,
                ood_detector=self.ood_detector,
            )
        else:
            theses = generate_theses(gsv)
        provider = make_fair_prob_provider(self.predictor)
        candidates = select_markets(
            theses, gsv, provider,
            top_k=self.top_k_per_thesis,
            target_stake=self.target_stake,
            mes_threshold=self.mes_threshold,
        )
        gate_results = run_gate(
            theses, candidates, gsv,
            mes_threshold=self.mes_threshold,
            line_max_age_sec=self.line_max_age_sec,
            commentary_required=self.commentary_required,
            uncertainty_band=self.uncertainty_band,
            ood_detector=self.ood_detector,
            mispricing_window_cfg=self.mispricing_window_cfg,
            drift_monitor=self.drift_monitor,
        )
        window = classify_gsv(gsv, self.mispricing_window_cfg)
        emitted = self._emitted_trios.setdefault(gsv.fixture_id, set())
        allowed: list[ShadowPick] = []
        promoted: list = []
        deduped_count = 0
        for r in gate_results:
            if not r.verdict.allowed:
                continue
            trio = (r.candidate.thesis.archetype.value, r.candidate.market_id)
            if self.dedupe_emissions and trio in emitted:
                deduped_count += 1
                continue
            allowed.append(ShadowPick(
                fixture_id=gsv.fixture_id,
                timestamp_utc=gsv.timestamp_utc,
                candidate=r.candidate,
                full_thesis=r.candidate.thesis,
            ))
            if self.dedupe_emissions:
                emitted.add(trio)
            if self.tier_promoter is not None:
                promoted.append(self.tier_promoter.promote(r.candidate, gsv))
        return PipelineOutput(
            gsv=gsv,
            theses=theses,
            candidates=candidates,
            gate_results=gate_results,
            allowed_picks=allowed,
            promoted_picks=promoted,
            mispricing_window=window,
            deduped_count=deduped_count,
        )


__all__ = ["PipelineOutput", "ShadowPick", "V3Pipeline"]
