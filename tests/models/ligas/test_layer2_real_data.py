"""Layer-2 placeholder tests — require real historical corpus.

Sprint 1 Ola C. These tests scaffold the integration with the real
historical corpus once Sprint 2+ wires data ingest. Each is marked
``xfail(strict=False)`` with reason ``requires-real-data`` so the
suite passes on green builds AND becomes an immediate signal when
real data lands (xpassed will be reported by pytest).

Convention: ship Layer-1 scaffolding tested
with mocks/synthetic; queue Layer-2 real-data validation as xfail.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REAL_CORPUS_PATH = Path("data/cache/historical/ligas_corpus.parquet")


def _real_corpus_present() -> bool:
    return REAL_CORPUS_PATH.exists()


@pytest.mark.xfail(
    not _real_corpus_present(),
    reason="requires-real-data: historical ligas corpus not yet ingested",
    strict=False,
)
def test_walkforward_on_real_pl_data_passes_or_shadows():
    """LigasModel(PL).evaluate() must NOT FAIL the gate on real PL data.

    The exact verdict (PASS vs SHADOW) depends on calibration quality
    of the trained sub-models. FAIL means the entire pipeline is
    publishing picks worse than random — that is a release blocker.
    """
    import polars as pl

    from bip.models.ligas import LigasModel
    from bip.models.ligas.gate import GateVerdict

    df = pl.read_parquet(REAL_CORPUS_PATH).filter(pl.col("league") == "PL")
    # Caller in Sprint 2+ will build the payload from df; for now we
    # just assert presence + run a stub. The xfail above prevents this
    # from running when the file is absent.
    assert df.height > 200
    model = LigasModel()
    payload = {
        "X": df.drop(["league", "kickoff_utc", "label_1x2"]).to_numpy(),
        "y": df["label_1x2"].to_numpy(),
        "dates": df["kickoff_utc"].to_numpy(),
    }
    metrics = model.evaluate(payload)
    assert "verdict=" in metrics.notes
    # The decision should be SHADOW or PASS, not FAIL.
    assert "FAIL" not in metrics.notes.split(";")[0], metrics.notes


@pytest.mark.xfail(
    not _real_corpus_present(),
    reason="requires-real-data: historical ligas corpus not yet ingested",
    strict=False,
)
def test_brier_ci_on_real_data_is_finite():
    """Brier CI must be finite on real data (no NaN propagation)."""
    import polars as pl

    from bip.models.ligas import LigasModel

    df = pl.read_parquet(REAL_CORPUS_PATH).filter(pl.col("league") == "PL")
    model = LigasModel()
    payload = {
        "X": df.drop(["league", "kickoff_utc", "label_1x2"]).to_numpy(),
        "y": df["label_1x2"].to_numpy(),
        "dates": df["kickoff_utc"].to_numpy(),
    }
    metrics = model.evaluate(payload)
    assert metrics.brier_score is not None
    assert metrics.brier_score > 0
