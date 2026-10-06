"""Pattern Layer — kNN secondary hypothesis generator.

Section 3 of the v3 engine design notes, the "Pattern layer (SECUNDARIO,
ML ligero)" lane of the Causal Hypothesis Generator:

    Embedding GSV → kNN sobre estados pasados →
        recupera tesis activadas en estados similes

The rule layer (12 archetypes in archetypes.py) covers the codified
domain knowledge. The pattern layer is the complement: when a GSV
"rhymes with" historical states that fired a particular thesis, we
recover that thesis even if the rule layer didn't catch the regime.

This breaks the hard ceiling the rule layer imposes on the universe of
opportunities. But it must do so WITHOUT becoming a back-door past
the No-Bet Gate — see the cold-start defenses below.

Design doc lives at internal research notes (local-only).
Decisions encoded here:

1. Embedding: 17-feature vector from ``vectorize_gsv`` (DRY w/ OOD detector).
2. Distance: euclidean over z-scored features.
3. k = 10 default. Auto-reduced to ``min(k, n//5)`` while 50 ≤ n < 200.
4. Recovery: vote by archetype → re-evaluate premise on CURRENT gsv.
5. Cold-start defenses:
   - n < min_samples_active → DEACTIVATED, returns [].
   - OOD detector says is_ood → returns []. Closes the rule-#9 backdoor.
   - max neighbor distance gate prevents "k furthest" promotion.
6. Composition: PARALLEL to rule layer; merge by archetype with
   max(confidence). Pattern NEVER reduces rule-layer confidence.
7. Pattern theses are minted with ``source.layer="pattern"`` for audit.

Constraint respected: no autoencoders, no LSH/faiss. sklearn KDTree on
1K-10K samples is sufficient and already on the stack.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.neighbors import KDTree

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.ood_detector import (
    N_FEATURES,
    OODDetector,
    vectorize_gsv,
)
from bip.evaluation.live.engine_v3.thesis import (
    ConditionalShift,
    GSVPredicate,
    Thesis,
    ThesisArchetype,
    ThesisSource,
    build_horizon,
)

_log = structlog.get_logger(__name__)


# ── Config ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatternLayerConfig:
    """Tunable knobs in one place. Defaults documented in module docstring."""

    k: int = 10
    max_neighbor_distance: float = 3.0  # z-score units; ~3σ avg per dim
    k_min_consensus: int = 3
    min_samples_active: int = 50
    min_samples_full: int = 200


# ── Records ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatternRecord:
    """One historical (state, thesis) pair used as a training row."""

    feature_vector: np.ndarray  # 17-dim, z-scored
    thesis: Thesis


# ── Predicate evaluator ─────────────────────────────────────────────────


def _resolve_path(gsv: GameStateVector, path: str) -> Any:
    """Walk a dotted path on the GSV (e.g. ``score.dominant_losing``)."""
    obj: Any = gsv
    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _evaluate_predicate(p: GSVPredicate, gsv: GameStateVector) -> bool:
    """Evaluate a single ``GSVPredicate`` against the GSV.

    Returns False on any missing attribute — a missing field is a
    failed predicate, never a true one. The op set is the closed
    literal from thesis.GSVPredicate.
    """
    val = _resolve_path(gsv, p.path)
    op = p.op
    if op == "eq":
        return val == p.value
    if op == "ne":
        return val != p.value
    if op == "lt":
        return val is not None and val < p.value
    if op == "le":
        return val is not None and val <= p.value
    if op == "gt":
        return val is not None and val > p.value
    if op == "ge":
        return val is not None and val >= p.value
    if op == "in":
        return val in p.value
    if op == "not_in":
        return val not in p.value
    if op == "between":
        lo, hi = p.value
        return val is not None and lo <= val <= hi
    if op == "contains":
        return val is not None and p.value in val
    return False  # unreachable due to Literal type, defensive


def _all_predicates_hold(thesis: Thesis, gsv: GameStateVector) -> bool:
    return all(_evaluate_predicate(p, gsv) for p in thesis.premise)


# ── Thesis minting ───────────────────────────────────────────────────────


def _mint_pattern_thesis(
    template: Thesis,
    *,
    minute: int,
    vote_count: int,
    boost_confidence: float | None = None,
) -> Thesis:
    """Build a new Thesis with pattern-layer provenance.

    The id, source, horizon, and activated_at_minute are refreshed.
    The premise, mechanism, prediction direction, and invalidation
    triggers are inherited verbatim — they are the thesis content.
    """
    refreshed_horizon = build_horizon(
        template.prediction.horizon.label, minute,
    )
    new_prediction = ConditionalShift(
        family=template.prediction.family,
        direction=template.prediction.direction,
        magnitude_pp=template.prediction.magnitude_pp,
        horizon=refreshed_horizon,
    )
    confidence = (
        boost_confidence
        if boost_confidence is not None
        else template.confidence_prior
    )
    return Thesis(
        id=f"PATTERN:{template.archetype.value}@m{minute}",
        archetype=template.archetype,
        premise=list(template.premise),
        mechanism=template.mechanism,
        prediction=new_prediction,
        invalidation_triggers=list(template.invalidation_triggers),
        confidence_prior=confidence,
        source=ThesisSource(
            layer="pattern",
            identifier=f"knn:vote={vote_count}",
        ),
        activated_at_minute=minute,
    )


# ── Pattern layer ────────────────────────────────────────────────────────


@dataclass
class PatternLayer:
    """kNN-based hypothesis recovery from historical (GSV, Thesis) pairs.

    Fit once with ``(gsv, fired_thesis)`` pairs; then call ``propose(gsv)``
    to recover theses by archetype vote among nearest neighbours, with
    the current GSV's premise check as the final filter.
    """

    cfg: PatternLayerConfig = field(default_factory=PatternLayerConfig)
    records: list[PatternRecord] = field(default_factory=list)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    is_fitted: bool = False
    _tree: KDTree | None = field(default=None, repr=False)

    # ── fit ────────────────────────────────────────────────────────

    def fit(self, pairs: list[tuple[GameStateVector, Thesis]]) -> "PatternLayer":
        """Build the kNN index from (GSV, fired Thesis) pairs.

        ``fit`` is idempotent — calling it again rebuilds the index
        from scratch.
        """
        if not pairs:
            raise ValueError("PatternLayer.fit: no (gsv, thesis) pairs provided")

        X = np.asarray([vectorize_gsv(g) for g, _ in pairs], dtype=np.float64)
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"feature width mismatch: got {X.shape[1]}, want {N_FEATURES}"
            )

        # z-score normalisation — euclidean distance demands scale parity.
        self.mean = X.mean(axis=0)
        # ddof=1 sample std; add eps to avoid divide-by-zero on constant cols.
        self.std = X.std(axis=0, ddof=1) + 1e-8
        X_norm = (X - self.mean) / self.std

        self.records = [
            PatternRecord(feature_vector=X_norm[i], thesis=t)
            for i, (_, t) in enumerate(pairs)
        ]
        # KDTree is only built when we have enough samples to ask k of them.
        if len(self.records) >= self.cfg.min_samples_active:
            self._tree = KDTree(X_norm)
        else:
            self._tree = None
        self.is_fitted = True
        return self

    # ── propose ────────────────────────────────────────────────────

    def propose(
        self,
        gsv: GameStateVector,
        *,
        ood_detector: OODDetector | None = None,
    ) -> list[Thesis]:
        """Recover theses by kNN vote, filtered through cold-start defenses
        and premise re-evaluation. Returns ``[]`` when in any guard branch.
        """
        if not self.is_fitted:
            return []
        n = len(self.records)
        if n < self.cfg.min_samples_active or self._tree is None:
            return []
        # Cold-start guard B: OOD piggyback.
        if (
            ood_detector is not None
            and ood_detector.is_fitted
            and ood_detector.is_ood(gsv)
        ):
            _log.debug("pattern_layer_skipped_ood", n_records=n)
            return []

        assert self.mean is not None and self.std is not None
        # k auto-reduce while between active and full thresholds.
        if n >= self.cfg.min_samples_full:
            k = min(self.cfg.k, n)
        else:
            k = max(3, n // 5)
            if k > n:
                k = n

        v = vectorize_gsv(gsv)
        v_norm = (v - self.mean) / self.std
        distances, indices = self._tree.query(v_norm.reshape(1, -1), k=k)
        distances = distances[0]
        indices = indices[0]

        # Cold-start guard A: distance gate.
        valid_pairs = [
            (d, i) for d, i in zip(distances, indices)
            if d <= self.cfg.max_neighbor_distance
        ]
        if len(valid_pairs) < self.cfg.k_min_consensus:
            return []

        # Vote by archetype among valid neighbours.
        votes: Counter[ThesisArchetype] = Counter()
        examples: dict[ThesisArchetype, Thesis] = {}
        for _, idx in valid_pairs:
            rec = self.records[idx]
            votes[rec.thesis.archetype] += 1
            # Keep the first example per archetype — they share archetype
            # invariants, so any one is a fine template to re-mint from.
            examples.setdefault(rec.thesis.archetype, rec.thesis)

        proposed: list[Thesis] = []
        for archetype, vote_count in votes.items():
            if vote_count < self.cfg.k_min_consensus:
                continue
            template = examples[archetype]
            # Final filter: the recovered thesis's premise must hold on
            # the CURRENT GSV, not just on its neighbours.
            if not _all_predicates_hold(template, gsv):
                continue
            proposed.append(_mint_pattern_thesis(
                template,
                minute=gsv.time.minute,
                vote_count=vote_count,
            ))
        return proposed

    # ── persistence ────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PatternLayer":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"loaded object is {type(obj)!r}, not PatternLayer")
        return obj


# ── Hybrid hypothesis generator ─────────────────────────────────────────


def merge_theses(
    rule_theses: list[Thesis],
    pattern_theses: list[Thesis],
) -> list[Thesis]:
    """Merge rule + pattern outputs by archetype.

    Decision (design doc #6):
    - If both fire the same archetype, keep one thesis with
      ``confidence_prior = max(rule.confidence, pattern.confidence)``.
      Provenance defaults to the rule layer (which is auditable) unless
      pattern's confidence is strictly higher, in which case we keep
      pattern's provenance (otherwise the layer wouldn't earn the
      higher prior).
    - If only one layer fires for an archetype, that thesis goes through.
    """
    by_archetype: dict[ThesisArchetype, Thesis] = {}
    for t in rule_theses:
        by_archetype[t.archetype] = t
    for pt in pattern_theses:
        existing = by_archetype.get(pt.archetype)
        if existing is None:
            by_archetype[pt.archetype] = pt
            continue
        # Keep the higher-confidence variant.
        if pt.confidence_prior > existing.confidence_prior:
            by_archetype[pt.archetype] = pt
        # else: existing (rule) wins; do nothing
    return list(by_archetype.values())


def generate_theses_hybrid(
    gsv: GameStateVector,
    *,
    pattern_layer: PatternLayer | None = None,
    ood_detector: OODDetector | None = None,
) -> list[Thesis]:
    """Run rule layer + (optional) pattern layer and merge by archetype.

    When ``pattern_layer`` is None or unfitted, behaviour is exactly
    equivalent to ``generate_theses(gsv)`` — Phase-1 callers see no
    change.
    """
    # Local import to avoid pulling archetypes into modules that don't need them.
    from bip.evaluation.live.engine_v3.archetypes import generate_theses

    rule = generate_theses(gsv)
    if pattern_layer is None or not pattern_layer.is_fitted:
        return rule
    pattern = pattern_layer.propose(gsv, ood_detector=ood_detector)
    return merge_theses(rule, pattern)


__all__ = [
    "PatternLayer",
    "PatternLayerConfig",
    "PatternRecord",
    "generate_theses_hybrid",
    "merge_theses",
]
