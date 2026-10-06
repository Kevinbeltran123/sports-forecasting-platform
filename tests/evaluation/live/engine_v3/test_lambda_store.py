"""Tests for Wave-3 Task 3.3: ML lambda store + dominant-team ML prior tier.

All tests are offline/synthetic — no real penaltyblog corpus required.
Uses generate_synthetic_corpus() from build_lambda_store to create a
deterministic training set.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_state(
    home_team_id: int = 100,
    away_team_id: int = 200,
    home_goals: int = 0,
    away_goals: int = 0,
    home_team_name: str = "Home FC",
    away_team_name: str = "Away FC",
):
    from bip.evaluation.live.match_state import LiveMatchState
    return LiveMatchState(
        fixture_id=999,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals=home_goals,
        away_goals=away_goals,
        minute=60,
        period_id=2,
        is_live=True,
        is_half_time=False,
        is_finished=False,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
    )


def _make_markets_empty():
    from bip.evaluation.live.engine_v3.gsv import MarketSnapshot
    return MarketSnapshot()


# ──────────────────────────────────────────────────────────────────────
# build_lambda_store.py unit tests
# ──────────────────────────────────────────────────────────────────────


def test_generate_synthetic_corpus_structure():
    """Synthetic corpus has the required keys and shapes."""
    from scripts.build_lambda_store import generate_synthetic_corpus
    corpus = generate_synthetic_corpus(n_teams=4, n_fixtures=12)
    assert "results" in corpus and "teams" in corpus
    results = corpus["results"]
    assert len(results) == 12
    for r in results:
        assert "home_team" in r and "away_team" in r
        assert "home_goals" in r and "away_goals" in r
        assert isinstance(r["home_goals"], int)
        assert r["home_team"] != r["away_team"]


def test_fit_dixon_coles_on_synthetic_corpus():
    """Dixon-Coles model fits successfully on synthetic data."""
    from scripts.build_lambda_store import (
        generate_synthetic_corpus, fit_dixon_coles,
    )
    corpus = generate_synthetic_corpus(n_teams=6, n_fixtures=40)
    model = fit_dixon_coles(corpus["results"], min_matches=2)
    assert model is not None, "Dixon-Coles fit must succeed on synthetic corpus"
    assert model.fitted, "Model must be fitted"
    assert model.n_teams > 0


def test_fit_dixon_coles_too_few_returns_none():
    """Corpus with fewer than 10 rows returns None."""
    from scripts.build_lambda_store import fit_dixon_coles
    tiny = [{"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0}]
    model = fit_dixon_coles(tiny, min_matches=1)
    assert model is None, "Too-small corpus must return None"


def test_predict_lambdas_emits_correct_row(tmp_path):
    """predict_lambdas returns a LambdaRow with plausible goal expectations."""
    from scripts.build_lambda_store import (
        generate_synthetic_corpus, fit_dixon_coles, predict_lambdas,
    )
    corpus = generate_synthetic_corpus(n_teams=6, n_fixtures=40)
    model = fit_dixon_coles(corpus["results"], min_matches=2)
    assert model is not None

    teams = corpus["teams"]
    fixtures = [{"fixture_id": 1001, "home_team": teams[0], "away_team": teams[1]}]
    rows = predict_lambdas(model, fixtures)
    assert len(rows) == 1, "Expected exactly 1 lambda row"
    row = rows[0]
    assert row.fixture_id == 1001
    assert 0.1 < row.lambda_home < 6.0, f"Implausible lambda_home={row.lambda_home}"
    assert 0.1 < row.lambda_away < 6.0, f"Implausible lambda_away={row.lambda_away}"
    assert row.model_version == "dixon_coles_v1"


def test_predict_lambdas_skips_unknown_team():
    """Teams not in model vocabulary produce no row (with a warning, not crash)."""
    from scripts.build_lambda_store import (
        generate_synthetic_corpus, fit_dixon_coles, predict_lambdas,
    )
    corpus = generate_synthetic_corpus(n_teams=6, n_fixtures=40)
    model = fit_dixon_coles(corpus["results"], min_matches=2)
    assert model is not None

    fixtures = [{"fixture_id": 9999, "home_team": "Unknown FC", "away_team": corpus["teams"][0]}]
    rows = predict_lambdas(model, fixtures)
    assert rows == [], "Unknown team must produce no lambda row (no crash)"


def test_build_lambda_store_writes_parquet(tmp_path):
    """End-to-end: build_lambda_store writes a readable parquet file."""
    import polars as pl
    from scripts.build_lambda_store import (
        generate_synthetic_corpus, build_lambda_store,
    )
    corpus = generate_synthetic_corpus(n_teams=6, n_fixtures=40)
    teams = corpus["teams"]
    fixtures = [
        {"fixture_id": 1001, "home_team": teams[0], "away_team": teams[1]},
        {"fixture_id": 1002, "home_team": teams[2], "away_team": teams[3]},
    ]
    output = tmp_path / "lambda_store.parquet"
    rows = build_lambda_store(corpus["results"], fixtures, output, min_matches=2)
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert output.exists()

    df = pl.read_parquet(output)
    assert set(df.columns) >= {"fixture_id", "lambda_home", "lambda_away", "model_version"}
    assert df.height == 2
    assert set(df["fixture_id"].to_list()) == {1001, 1002}


def test_read_lambda_store_roundtrip(tmp_path):
    """read_lambda_store correctly deserializes what write_lambda_store wrote."""
    import polars as pl
    from scripts.build_lambda_store import (
        LambdaRow, write_lambda_store, read_lambda_store,
    )
    rows = [
        LambdaRow(fixture_id=100, lambda_home=1.8, lambda_away=1.1),
        LambdaRow(fixture_id=200, lambda_home=0.9, lambda_away=1.4),
    ]
    path = tmp_path / "lambda_store.parquet"
    write_lambda_store(rows, path)
    loaded = read_lambda_store(path)
    assert 100 in loaded
    assert 200 in loaded
    assert loaded[100].lambda_home == pytest.approx(1.8)
    assert loaded[200].lambda_away == pytest.approx(1.4)


def test_read_lambda_store_missing_file():
    """read_lambda_store returns empty dict when file doesn't exist."""
    from scripts.build_lambda_store import read_lambda_store
    result = read_lambda_store(Path("/tmp/nonexistent_lambda_store_xyz.parquet"))
    assert result == {}


# ──────────────────────────────────────────────────────────────────────
# derive_priors_from_fixture integration tests
# ──────────────────────────────────────────────────────────────────────


def _make_simple_fixture(fixture_id: int):
    """Minimal duck-typed fixture object with no predictions."""
    class FakeFixture:
        def __init__(self, fid):
            self.id = fid
            self.predictions = []
    return FakeFixture(fixture_id)


def test_derive_priors_uses_lambda_store_when_available(tmp_path):
    """derive_priors_from_fixture reads λ-store row when it exists
    and populates PreMatchPriors from it."""
    from scripts.build_lambda_store import LambdaRow, write_lambda_store
    from bip.evaluation.live.engine_v3.runtime.dual_write import derive_priors_from_fixture

    # Write a lambda store with one fixture
    rows = [LambdaRow(fixture_id=999, lambda_home=1.95, lambda_away=0.85)]
    store_path = tmp_path / "lambda_store.parquet"
    write_lambda_store(rows, store_path)

    fixture = _make_simple_fixture(999)
    priors = derive_priors_from_fixture(fixture, lambda_store_path=store_path)
    assert priors.lambda_home_prematch == pytest.approx(1.95, abs=0.01)
    assert priors.lambda_away_prematch == pytest.approx(0.85, abs=0.01)


def test_derive_priors_falls_back_when_fixture_not_in_store(tmp_path):
    """derive_priors_from_fixture uses Sportmonks/defaults when
    fixture_id is absent from the λ-store."""
    from scripts.build_lambda_store import LambdaRow, write_lambda_store
    from bip.evaluation.live.engine_v3.runtime.dual_write import derive_priors_from_fixture

    rows = [LambdaRow(fixture_id=100, lambda_home=2.0, lambda_away=0.8)]
    store_path = tmp_path / "lambda_store.parquet"
    write_lambda_store(rows, store_path)

    # fixture_id=200 is NOT in the store → falls back to neutral defaults
    fixture = _make_simple_fixture(200)
    priors = derive_priors_from_fixture(fixture, lambda_store_path=store_path)
    # Neutral defaults: 1.35 / 1.15
    assert priors.lambda_home_prematch == pytest.approx(1.35, abs=0.01)
    assert priors.lambda_away_prematch == pytest.approx(1.15, abs=0.01)


def test_derive_priors_works_without_store(tmp_path):
    """derive_priors_from_fixture works when no store exists (graceful fallback)."""
    from bip.evaluation.live.engine_v3.runtime.dual_write import derive_priors_from_fixture

    missing_store = tmp_path / "nonexistent_store.parquet"
    fixture = _make_simple_fixture(555)
    # Must not raise, falls back to neutral defaults
    priors = derive_priors_from_fixture(fixture, lambda_store_path=missing_store)
    assert priors.lambda_home_prematch == pytest.approx(1.35, abs=0.01)


# ──────────────────────────────────────────────────────────────────────
# _choose_dominant_team_id ML-lambda tier tests
# ──────────────────────────────────────────────────────────────────────


def _priors(lh: float, la: float, elo: float = 0.0):
    from bip.evaluation.live.engine_v3.gsv import PreMatchPriors
    return PreMatchPriors(
        lambda_home_prematch=lh,
        lambda_away_prematch=la,
        elo_diff=elo,
    )


def test_dominant_resolves_to_higher_lambda_team_decisive_gap():
    """ML-lambda tier: when |lh - la| >= min_gap, the higher-λ team wins.

    Away team has clearly higher λ (2.1 vs 1.2, gap=0.9 >> 0.15) →
    away should be dominant regardless of market signal.
    """
    from bip.evaluation.live.engine_v3.gsv_builder import _choose_dominant_team_id

    state = _make_state(home_team_id=100, away_team_id=200)
    priors = _priors(lh=1.2, la=2.1)   # away has higher λ
    markets = _make_markets_empty()     # no market signal

    result = _choose_dominant_team_id(state, priors, markets, ml_lambda_min_gap=0.15)
    assert result == 200, (
        f"Higher-λ away team must be dominant when gap is decisive, got {result}"
    )


def test_dominant_falls_back_to_market_when_lambda_coin_flip():
    """ML-lambda tier: when |lh - la| < min_gap (coin-flip λ), falls back
    to market signal.

    Both lambdas are near-equal (gap=0.05 < 0.15). The market clearly
    says home is favourite → home must win via tier 3.
    """
    from bip.evaluation.live.engine_v3.gsv_builder import (
        _choose_dominant_team_id, _MARKET_DOM_MIN_PROB_GAP,
    )
    from bip.evaluation.live.engine_v3.gsv import MarketLine, MarketSnapshot
    from bip.evaluation.live.match_state import LiveMatchState

    state = _make_state(
        home_team_id=100, away_team_id=200,
        home_team_name="Home FC", away_team_name="Away FC",
    )
    priors = _priors(lh=1.35, la=1.30)  # gap=0.05 < 0.15 → coin-flip

    # Market strongly favours home via team-named BTTSxResult lines
    now = datetime.now(timezone.utc)
    lines = {
        "result___both_teams_to_score_home_fc_/_yes": MarketLine(
            market_id="result___both_teams_to_score_home_fc_/_yes",
            side_a_decimal=2.5, max_stake_cap=500.0, last_update_utc=now,
        ),
        "result___both_teams_to_score_home_fc_/_no": MarketLine(
            market_id="result___both_teams_to_score_home_fc_/_no",
            side_a_decimal=1.8, max_stake_cap=500.0, last_update_utc=now,
        ),
        "result___both_teams_to_score_away_fc_/_yes": MarketLine(
            market_id="result___both_teams_to_score_away_fc_/_yes",
            side_a_decimal=8.0, max_stake_cap=500.0, last_update_utc=now,
        ),
        "result___both_teams_to_score_away_fc_/_no": MarketLine(
            market_id="result___both_teams_to_score_away_fc_/_no",
            side_a_decimal=5.0, max_stake_cap=500.0, last_update_utc=now,
        ),
    }
    markets = MarketSnapshot(lines=lines)

    result = _choose_dominant_team_id(state, priors, markets, ml_lambda_min_gap=0.15)
    assert result == 100, (
        f"Market favourite (home) must win when λ is coin-flip, got {result}"
    )


def test_dominant_ml_lambda_above_market():
    """ML-lambda tier sits ABOVE bookmaker market tier.

    λ clearly favours home (lh=2.0, la=1.0, gap=1.0 >> 0.15).
    Market has no team-named lines → ML-λ must win.
    This confirms tier ordering: ELO > ML-λ > market > Sportmonks-λ.
    """
    from bip.evaluation.live.engine_v3.gsv_builder import _choose_dominant_team_id

    state = _make_state(home_team_id=100, away_team_id=200)
    priors = _priors(lh=2.0, la=1.0)   # home clearly favoured by ML-λ
    markets = _make_markets_empty()     # no market signal available

    result = _choose_dominant_team_id(state, priors, markets, ml_lambda_min_gap=0.15)
    assert result == 100, (
        f"ML-λ tier must pick home when gap is decisive and market is empty, got {result}"
    )


def test_dominant_elo_overrides_ml_lambda():
    """ELO tier (tier 1) beats ML-lambda tier (tier 2).

    ELO strongly favours away (elo_diff=-50 < -25 threshold).
    λ favours home (gap decisive). ELO must override.
    """
    from bip.evaluation.live.engine_v3.gsv_builder import _choose_dominant_team_id

    state = _make_state(home_team_id=100, away_team_id=200)
    priors = _priors(lh=2.0, la=1.0, elo=-50.0)  # ELO says away; λ says home
    markets = _make_markets_empty()

    result = _choose_dominant_team_id(state, priors, markets, ml_lambda_min_gap=0.15)
    assert result == 200, (
        f"ELO must override ML-λ when ELO is decisive, got {result}"
    )


def test_dominant_ml_lambda_zero_gap_min_falls_back():
    """When ml_lambda_min_gap=inf, ML-λ tier never fires → falls to market/λ."""
    from bip.evaluation.live.engine_v3.gsv_builder import _choose_dominant_team_id

    state = _make_state(home_team_id=100, away_team_id=200)
    priors = _priors(lh=2.5, la=0.8)  # decisive gap but min_gap=inf
    markets = _make_markets_empty()

    result = _choose_dominant_team_id(state, priors, markets, ml_lambda_min_gap=float("inf"))
    # Falls to tier 4 (raw λ): home > away → home wins
    assert result == 100, (
        f"Tier 4 raw-λ fallback must pick higher λ when ML tier is bypassed, got {result}"
    )
