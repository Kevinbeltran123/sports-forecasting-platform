"""Lock_v2.json emitter — FAIL verdict transparent audit artifact.

Per internal planning notes Ola 7, this module emits a
lock_v2.json file that:

1. Predicts every WC2026 group-stage fixture (from lock_v1's fixture list)
   under the CalibratedDIBPPredictor (v2 full configuration).
2. Records the lock-gate metrics from the Ola 5 backtest + the ablation
   verdict from Ola 6.
3. Calibration_status = ``below-gate`` (consistent with lock_v1; the v2
   ablation showed none of the proposed components materially improve over
   the baseline — see internal research notes).
4. SHA-256 content_hash via the existing ``emit_lock_json`` API.

This artifact is **not** registered in the ModelRegistry. lock_v1 remains
the WC2026 production path (per the FAIL verdict in PLAN.md §"Gate
verdict written to ... 'Gate Verdict'"). lock_v2 exists purely for
post-tournament scoring + the audit trail.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from bip.evaluation.tournaments.backtest.lock_emitter import (
    FixtureLockedPredictions,
    emit_lock_json,
)
from bip.evaluation.tournaments.backtest.lock_gate import (
    DEFAULT_COVERAGE_THRESHOLD,
    LockDecision,
    PredictorGateVerdict,
)

from .v2_predictor import CalibratedDIBPPredictor

_REPO_ROOT = Path(__file__).resolve().parents[5]
_LOCK_V1_PATH = (
    _REPO_ROOT
    / "src"
    / "bip"
    / "evaluation"
    / "tournaments"
    / "locked_predictions"
    / "world_cup_2026"
    / "lock.json"
)
LOCK_V2_PATH = (
    _REPO_ROOT
    / "src"
    / "bip"
    / "evaluation"
    / "tournaments"
    / "locked_predictions"
    / "world_cup_2026"
    / "lock_v2.json"
)

PREDICTOR_NAME = "wc2026_v2_calibrated_dibp"

# Ola 5 backtest results on the 199-match hold-out (full v2 configuration).
# These drive the lock_v2 calibration_status field. Sourced from
# internal research notes row ``full``.
BACKTEST_METRICS = {
    "1x2": {"brier_for_gate": 0.2126, "ece": 0.1145},
    "btts": {"brier_for_gate": 0.2586, "ece": 0.0802},
    "goals_total_o_u_2_5": {"brier_for_gate": 0.2508, "ece": 0.0668},
}

OPERATOR_OVERRIDES = (
    "Ola 6 ablation (internal research notes): paired delta-Brier "
    "1X2 vs full configuration showed DIBP contributes +0.0000 [-0.0002, +0.0003] "
    "(not significant), beta calibration -0.0000 [-0.0000, +0.0000] (not significant), "
    "and match-importance weighting -0.0011 [-0.0020, -0.0001] (statistically "
    "significant but goes the wrong direction — removing it improves Brier).",
    "Honest FAIL verdict: none of the three proposed signals (Karlis-Ntzoufras "
    "diagonal-inflation, Kull et al. 2017 beta calibration, Ley et al. 2019 "
    "match-importance weighted MLE) materially improve over the lock_v1 baseline "
    "on this hold-out corpus.",
    "Predictor architecture differs from lock_v1: DIBP wraps the same "
    "Bivariate Poisson core (predictors/bivariate_poisson.py) with optional "
    "diagonal inflation (Karlis-Ntzoufras 2003 J(k;θ) = (1−θ)·θ^k). The fit "
    "returned π_diag=0 on the calibration corpus (WC2018+Euro2020 n=115), so "
    "the deployed predictor reduces to plain BP with a Ley 2019 weighted-MLE "
    "strength prior in place of lock_v1's xG-blended Bivariate (α=0.20).",
    "Lock_v2 emitted as audit artifact only — not registered in the "
    "ModelRegistry. Lock_v1 (commit 8c0ce19) remains the WC2026 production path "
    "per PLAN.md Ola 7 §'Gate Verdict' = FAIL branch.",
)

FORCED_EMIT_REASON = (
    "Ola 7 FAIL-verdict audit emission. The v2 spike (Olas 0-6) explored three "
    "academic signals — DIBP draw inflation, beta calibration, and match-"
    "importance weighting — on the WC2026 hold-out (AFCON 2023 + Copa 2024 + "
    "Euro 2024 + WC 2022, n=199). Walk-forward backtest with 4-tournament "
    "rolling-origin CV yielded 1X2 Brier=0.2126 [CI 0.1938-0.2325] vs lock_v1's "
    "0.2156 [CI 0.2027-0.2290] — a Δ=-0.003 point estimate with overlapping "
    "CIs (not significant). Per-component ablation: DIBP +0.0000, beta -0.0000, "
    "match-importance -0.0011 (wrong direction). FAIL verdict; lock_v1 ships. "
    "lock_v2.json preserved as audit trail per PLAN.md."
)


def _load_lock_v1_fixtures() -> list[dict]:
    """Load the 72 WC2026 fixture stubs from lock_v1 (team names + kickoff)."""

    with _LOCK_V1_PATH.open() as f:
        lock_v1 = json.load(f)
    return list(lock_v1["fixtures"])


def _build_fixture_predictions(
    predictor: CalibratedDIBPPredictor,
    lock_v1_fixtures: list[dict],
) -> tuple[FixtureLockedPredictions, ...]:
    """Apply the v2 predictor to every lock_v1 fixture; return wrapped records."""

    out: list[FixtureLockedPredictions] = []
    for fx in lock_v1_fixtures:
        pred = predictor.predict(fx["home_team_name"], fx["away_team_name"])
        out.append(
            FixtureLockedPredictions(
                match_id=fx["match_id"],
                tournament_phase=fx["tournament_phase"],
                kickoff_utc=datetime.fromisoformat(fx["kickoff_utc"].replace("Z", "+00:00")),
                home_team_id=fx["home_team_id"],
                away_team_id=fx["away_team_id"],
                home_team_name=fx["home_team_name"],
                away_team_name=fx["away_team_name"],
                predictions={
                    PREDICTOR_NAME: {
                        "p_home_win": pred.p_home_win,
                        "p_draw": pred.p_draw,
                        "p_away_win": pred.p_away_win,
                        "p_btts": pred.p_btts,
                        "p_over_2_5": pred.p_over_2_5,
                    }
                },
            )
        )
    return tuple(out)


def _build_lock_decision() -> LockDecision:
    verdict = PredictorGateVerdict(
        predictor_name=PREDICTOR_NAME,
        n_markets=3,
        market_statuses={
            "1X2": "below-gate",
            "btts": "below-gate",
            "goals_total_o_u_2_5": "below-gate",
        },
        market_brier_for_gate={
            "1X2": BACKTEST_METRICS["1x2"]["brier_for_gate"],
            "btts": BACKTEST_METRICS["btts"]["brier_for_gate"],
            "goals_total_o_u_2_5": BACKTEST_METRICS["goals_total_o_u_2_5"]["brier_for_gate"],
        },
        market_classwise_ece={
            "1X2": BACKTEST_METRICS["1x2"]["ece"],
            "btts": BACKTEST_METRICS["btts"]["ece"],
            "goals_total_o_u_2_5": BACKTEST_METRICS["goals_total_o_u_2_5"]["ece"],
        },
        overall_status="below-gate",
    )
    return LockDecision(
        evaluated_at=datetime.now(UTC),
        git_sha=None,
        held_out_tournaments=("wc_2022", "afcon_2023", "copa_2024", "euro_2024"),
        n_fixtures_total=72,
        n_fixtures_with_predictions=72,
        coverage_threshold=DEFAULT_COVERAGE_THRESHOLD,
        predictor_verdicts=(verdict,),
        calibration_status="below-gate",
        operator_overrides=OPERATOR_OVERRIDES,
    )


def emit_lock_v2(
    predictor: CalibratedDIBPPredictor,
    *,
    git_sha: str | None = None,
    output_path: Path | None = None,
    locked_at: datetime | None = None,
) -> Path:
    """Emit lock_v2.json with FAIL-verdict audit metadata.

    Returns the path written. ``output_path`` defaults to
    ``LOCK_V2_PATH`` next to lock.json (does NOT overwrite lock.json).
    """

    target = output_path or LOCK_V2_PATH
    lock_v1_fixtures = _load_lock_v1_fixtures()
    fixtures = _build_fixture_predictions(predictor, lock_v1_fixtures)
    decision = _build_lock_decision()
    emit_lock_json(
        decision=decision,
        fixtures=fixtures,
        tournament_slug="world_cup_2026",
        git_sha=git_sha,
        output_path=target,
        force=True,  # below-gate calibration_status requires force
        forced_emit_reason=FORCED_EMIT_REASON,
        locked_at=locked_at,
    )
    return target
