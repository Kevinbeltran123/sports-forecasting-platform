"""GREEN tests for CORNERS-01 D-17b coverage logic."""
from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl


class TestThresholdLogic:
    def test_pass_when_every_league_has_3_plus_seasons_above_95pct(self):
        from scripts.corners_gate_coverage import gate_decision
        # Synthetic coverage table -- every league has 3 seasons at 96%
        rows = []
        for league in ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"):
            for season in (2023, 2024, 2025):
                rows.append({"league": league, "season": str(season), "ft_count": 380,
                              "coverage_pct": 0.96, "missing_cols": ""})
        cov = pl.DataFrame(rows)
        passed, md = gate_decision(cov)
        assert passed is True
        assert "**Overall verdict:** PASS" in md

    def test_fail_when_one_league_has_only_2_good_seasons(self):
        from scripts.corners_gate_coverage import gate_decision
        rows = []
        for league in ("premier_league", "la_liga", "bundesliga", "serie_a"):
            for season in (2023, 2024, 2025):
                rows.append({"league": league, "season": str(season), "ft_count": 380,
                              "coverage_pct": 0.96, "missing_cols": ""})
        # Ligue 1 only has 2 good seasons
        rows.append({"league": "ligue_1", "season": "2024", "ft_count": 380,
                      "coverage_pct": 0.96, "missing_cols": ""})
        rows.append({"league": "ligue_1", "season": "2025", "ft_count": 380,
                      "coverage_pct": 0.96, "missing_cols": ""})
        cov = pl.DataFrame(rows)
        passed, md = gate_decision(cov)
        assert passed is False
        assert "**Overall verdict:** FAIL" in md

    def test_fail_when_coverage_below_95(self):
        from scripts.corners_gate_coverage import gate_decision
        rows = [
            {"league": l, "season": str(s), "ft_count": 380, "coverage_pct": 0.94,
             "missing_cols": ""}
            for l in ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1")
            for s in (2023, 2024, 2025)
        ]
        cov = pl.DataFrame(rows)
        passed, _ = gate_decision(cov)
        assert passed is False

    def test_fail_when_all_columns_missing(self):
        """A5 + D-17b: corner cols absent -> coverage_pct=0 -> FAIL across the board."""
        from scripts.corners_gate_coverage import gate_decision
        rows = [
            {"league": l, "season": str(s), "ft_count": 380, "coverage_pct": 0.0,
             "missing_cols": "home_corners,away_corners"}
            for l in ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1")
            for s in (2023, 2024, 2025)
        ]
        cov = pl.DataFrame(rows)
        passed, md = gate_decision(cov)
        assert passed is False
        assert "**Overall verdict:** FAIL" in md
        assert "home_corners,away_corners" in md

    def test_empty_coverage_table_fails(self):
        from scripts.corners_gate_coverage import gate_decision
        cov = pl.DataFrame(schema={"league": pl.String, "season": pl.String,
                                      "ft_count": pl.Int64, "coverage_pct": pl.Float64,
                                      "missing_cols": pl.String})
        passed, md = gate_decision(cov)
        assert passed is False
        assert "no data" in md.lower()


class TestComputeCoverage:
    def test_ft_only_filter(self, tmp_path):
        """specifics §199: PST/CANC/ABD must NOT count as missing-coverage rows."""
        from scripts.corners_gate_coverage import compute_coverage

        store = MagicMock()
        # Mix FT + PST rows -- PST should be excluded from denominator
        df = pl.DataFrame({
            "fixture_id": [1, 2, 3],
            "season": [2024, 2024, 2024],
            "status": ["FT", "FT", "PST"],
            "home_corners": [5, 6, None],
            "away_corners": [3, 4, None],
        })
        # Return same df for every league for simplicity
        store.read_results.return_value = df

        cov = compute_coverage(store)
        # 2 FT rows, both with non-null corners -> coverage 100%
        pl_rows = cov.filter(pl.col("league") == "premier_league")
        assert pl_rows.height >= 1
        assert pl_rows["coverage_pct"][0] == 1.0
        assert pl_rows["ft_count"][0] == 2  # PST excluded

    def test_missing_columns_fail_gate(self):
        """A5: home_corners/away_corners absent -> coverage_pct=0 -> FAIL."""
        from scripts.corners_gate_coverage import compute_coverage

        store = MagicMock()
        # No corner columns
        df = pl.DataFrame({
            "fixture_id": [1, 2, 3],
            "season": [2024, 2024, 2024],
            "status": ["FT", "FT", "FT"],
        })
        store.read_results.return_value = df

        cov = compute_coverage(store)
        # Every row should have coverage_pct=0 and missing_cols mentioning home_corners/away_corners
        for league in ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"):
            league_rows = cov.filter(pl.col("league") == league)
            assert league_rows.height >= 1
            assert league_rows["coverage_pct"][0] == 0.0
            assert "home_corners" in league_rows["missing_cols"][0]
