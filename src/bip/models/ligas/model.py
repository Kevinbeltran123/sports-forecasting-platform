"""LigasModel — BaseModel impl routing 5 top European leagues.

Sprint 1. Adapter over the existing train/ stack (StackedEnsemble +
calibrator + walkforward + simulate_pick). One Market1x2 + one
MarketOverUnder per league (5 leagues × 2 markets = 10 sub-models when
fully trained). Each sub-model is independently fit on its league
corpus; LigasModel.predict() routes by competition + fixture metadata.

Until the real data corpus is connected (Sprint 1 Layer-2, requires real
historical data), train/evaluate operate on synthetic / mock inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from bip.models.base import BaseModel, ModelMetrics, PredictionRecord

log = structlog.get_logger(__name__)

# Local imports for late binding inside methods (avoid circular hits at
# import time with bip.models.ligas.__init__ which re-exports LigasModel)


class LigasModel(BaseModel):
    """Plugin-registry adapter over the legacy ensemble + calibrator stack."""

    name = "ligas_phase1"
    version = "0.1.0"

    def __init__(
        self,
        *,
        ev_threshold: float = 0.05,
    ) -> None:
        from bip.models.ligas.markets.market_1x2 import Market1x2
        from bip.models.ligas.markets.market_ou import MarketOverUnder

        self.ev_threshold = float(ev_threshold)
        # Per-(league, market) sub-models. Populated by train(); empty
        # until then. Keys: (league_str, market_id_str).
        self._submodels: dict[tuple[str, str], Any] = {}
        self._submodel_factory = {
            "1x2": Market1x2,
            "ou_2.5": lambda: MarketOverUnder(line=2.5),
        }

    # ------------------------------------------------------------------
    # BaseModel contract
    # ------------------------------------------------------------------

    def can_handle(self, fixture: dict[str, Any]) -> bool:
        from bip.models.ligas import SUPPORTED_LEAGUES

        comp = fixture.get("competition") or fixture.get("league")
        return comp in SUPPORTED_LEAGUES

    def predict(self, fixture: dict[str, Any]) -> list[PredictionRecord]:
        """Produce 0..N PredictionRecord rows for the fixture.

        Expects the fixture dict to carry:
          - fixture_id, competition, home_team, away_team
          - match_datetime (datetime) or kickoff_utc
          - features (np.ndarray or list[float]) — pre-computed by the
            feature engineering layer; the orchestrator hydrates this.
          - odds_at_pick (dict[selection, decimal_odds]) — optional; when
            present we compute ev = p_model * odds - 1.

        Returns empty list when:
          - no sub-models are trained for the (league, market),
          - features are missing (no fallback heuristic in Sprint 1),
          - or no selection passes the EV threshold.
        """
        from bip.models.ligas import MARKETS

        comp = fixture.get("competition") or fixture.get("league")
        if not self.can_handle(fixture):
            return []

        features = fixture.get("features")
        if features is None:
            log.info("ligas_skipped_missing_features", fixture_id=fixture.get("fixture_id"))
            return []
        x = np.asarray(features, dtype=float).reshape(1, -1)

        kickoff = self._coerce_kickoff(fixture)
        odds_book = fixture.get("odds_at_pick") or {}

        records: list[PredictionRecord] = []
        for market_id in MARKETS:
            sub_key = (comp, market_id)
            if sub_key not in self._submodels:
                continue
            sub = self._submodels[sub_key]
            try:
                probs = sub.predict_for_fixture(x)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "ligas_predict_error",
                    fixture_id=fixture.get("fixture_id"),
                    market=market_id,
                    error=str(exc),
                )
                continue

            for selection, p_model in probs.items():
                odds = odds_book.get(market_id, {}).get(selection)
                ev: float | None = None
                if odds is not None and float(odds) > 1.0:
                    ev = p_model * float(odds) - 1.0
                    if ev < self.ev_threshold:
                        continue
                elif self.ev_threshold > 0:
                    # No odds: cannot compute EV. Without EV we skip in
                    # Sprint 1 (strict gate); ev_threshold=0 lets it
                    # through for shadow-mode tests.
                    continue

                records.append(
                    PredictionRecord(
                        source="ligas",
                        fixture_id=str(fixture.get("fixture_id") or ""),
                        competition=str(comp),
                        home_team=str(fixture.get("home_team", "")),
                        away_team=str(fixture.get("away_team", "")),
                        match_datetime=kickoff,
                        market=market_id,
                        selection=selection,
                        p_model=float(p_model),
                        ev=ev,
                        odds_at_pick=float(odds) if odds is not None else None,
                        payload={
                            "league_id_map_key": comp,
                            "ev_threshold": self.ev_threshold,
                            "submodel_version": getattr(sub, "version", "n/a"),
                        },
                        model_version=self.version,
                    )
                )
        return records

    # ------------------------------------------------------------------
    # Training + evaluation
    # ------------------------------------------------------------------

    def train(self, data: Any) -> None:
        """Fit a (league, market) sub-model from a training payload.

        Expected payload shape — dict with keys:
            league: str       — must be in SUPPORTED_LEAGUES
            market: str       — must be in MARKETS
            X: np.ndarray     — (n_samples, n_features)
            y: np.ndarray     — (n_samples,) — class labels matching
                                the market's class space:
                                  1x2: 0/1/2 (home/draw/away)
                                  ou_2.5: 0/1 (under/over)

        Repeated calls with the same (league, market) re-fit and replace.
        """
        from bip.models.ligas import MARKETS, SUPPORTED_LEAGUES

        if not isinstance(data, dict):
            raise TypeError("train() expects a dict payload")
        league = data.get("league")
        market = data.get("market")
        X = data.get("X")
        y = data.get("y")
        if league not in SUPPORTED_LEAGUES:
            raise ValueError(f"Unsupported league: {league!r}")
        if market not in MARKETS:
            raise ValueError(f"Unsupported market: {market!r}")
        if X is None or y is None:
            raise ValueError("X and y are required")

        factory = self._submodel_factory[market]
        sub = factory()
        sub.fit(np.asarray(X), np.asarray(y))
        self._submodels[(league, market)] = sub
        log.info(
            "ligas_train_complete",
            league=league,
            market=market,
            n_samples=int(np.asarray(X).shape[0]),
        )

    def evaluate(self, holdout: Any) -> ModelMetrics:
        """Run walk-forward backtest + bootstrap-CI gate on the holdout.

        Delegates to bip.models.ligas.backtest.run_walkforward (Ola C).
        The expected holdout payload mirrors train() with an additional
        `dates: np.ndarray` of kickoff timestamps for the leakage check.
        """
        from bip.models.ligas.backtest import run_walkforward

        metrics, _decision = run_walkforward(holdout, ev_threshold=self.ev_threshold)
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_kickoff(fixture: dict[str, Any]) -> datetime:
        for key in ("match_datetime", "kickoff_utc"):
            v = fixture.get(key)
            if isinstance(v, datetime):
                return v
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                except ValueError:
                    continue
        return datetime.now(UTC)

    @property
    def trained_submodels(self) -> list[tuple[str, str]]:
        return list(self._submodels.keys())


__all__ = ["LigasModel"]
