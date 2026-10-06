"""Seed script checkpoint resume — D-01 (historical data acquisition).

Phase 02.1-09 expanded the on-disk checkpoint from a single
``completed_fixture_ids`` set to a 3-key dict so feature, results, and odds
writes can each resume independently. The Phase 2 single-key shape must still
load cleanly (backward-compat).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_seed_module():
    """Import scripts/seed_historical.py as a module (not in sys.path by default)."""
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "seed_historical", root / "scripts" / "seed_historical.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_historical"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _empty_state() -> dict[str, set[int]]:
    return {
        "completed_fixture_ids": set(),
        "completed_results_fixture_ids": set(),
        "completed_odds_fixture_ids": set(),
    }


class TestSeedCheckpoint:
    """Seed script can resume after interruption via checkpoint file."""

    def test_checkpoint_resume(self, tmp_path, monkeypatch):
        """Resume round-trips the 3-key dict — Phase 02.1-09 D-17."""
        seed = _load_seed_module()
        cp = tmp_path / "seed_checkpoint.json"
        monkeypatch.setattr(seed, "CHECKPOINT_PATH", cp)

        state = {
            "completed_fixture_ids": {101, 102, 103},
            "completed_results_fixture_ids": {101, 102},
            "completed_odds_fixture_ids": {101},
        }
        seed.save_checkpoint(state)
        reloaded = seed.load_checkpoint()
        assert reloaded == state

    def test_checkpoint_atomic_write(self, tmp_path, monkeypatch):
        """save_checkpoint writes to .tmp then renames — D-01 durability."""
        seed = _load_seed_module()
        cp = tmp_path / "seed_checkpoint.json"
        monkeypatch.setattr(seed, "CHECKPOINT_PATH", cp)

        state = _empty_state()
        state["completed_fixture_ids"].add(200)
        seed.save_checkpoint(state)

        # Final file exists; no stray .tmp left behind
        assert cp.exists()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"

    def test_checkpoint_empty_when_missing(self, tmp_path, monkeypatch):
        """load_checkpoint returns the 3-key empty dict when file is absent."""
        seed = _load_seed_module()
        cp = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(seed, "CHECKPOINT_PATH", cp)

        assert seed.load_checkpoint() == _empty_state()

    def test_checkpoint_backward_compat_with_old_shape(self, tmp_path, monkeypatch):
        """Old Phase 2 checkpoint (only completed_fixture_ids) loads cleanly."""
        seed = _load_seed_module()
        cp = tmp_path / "seed_checkpoint.json"
        monkeypatch.setattr(seed, "CHECKPOINT_PATH", cp)

        # Phase 2 on-disk shape: only the legacy single key.
        cp.write_text(json.dumps({"completed_fixture_ids": [101, 102]}))
        reloaded = seed.load_checkpoint()
        assert reloaded["completed_fixture_ids"] == {101, 102}
        assert reloaded["completed_results_fixture_ids"] == set()
        assert reloaded["completed_odds_fixture_ids"] == set()

    def test_pre_cr01_checkpoint_resets_results_and_odds(
        self, tmp_path, monkeypatch,
    ):
        """WR-03: a pre-CR-01 checkpoint (no schema_version) auto-resets the
        results/odds tracking sets so the seed re-writes those fixtures.

        Features tracking is preserved because the features write path was
        never affected by CR-01.
        """
        seed = _load_seed_module()
        cp = tmp_path / "seed_checkpoint.json"
        monkeypatch.setattr(seed, "CHECKPOINT_PATH", cp)

        # Pre-CR-01 shape: 3-key dict but no schema_version field.
        cp.write_text(
            json.dumps(
                {
                    "completed_fixture_ids": [501, 502],
                    "completed_results_fixture_ids": [501, 502],
                    "completed_odds_fixture_ids": [501],
                    # Note: schema_version intentionally absent.
                }
            )
        )
        reloaded = seed.load_checkpoint()
        assert reloaded["completed_fixture_ids"] == {501, 502}, (
            "features tracking must be preserved across the version bump"
        )
        assert reloaded["completed_results_fixture_ids"] == set(), (
            "WR-03: pre-CR-01 results tracking is untrusted; must reset"
        )
        assert reloaded["completed_odds_fixture_ids"] == set(), (
            "WR-03: pre-CR-01 odds tracking is untrusted; must reset"
        )

    def test_post_cr01_checkpoint_round_trips_with_version(
        self, tmp_path, monkeypatch,
    ):
        """A checkpoint we wrote ourselves carries schema_version and survives
        a save/load cycle without the WR-03 reset firing.
        """
        seed = _load_seed_module()
        cp = tmp_path / "seed_checkpoint.json"
        monkeypatch.setattr(seed, "CHECKPOINT_PATH", cp)

        state = {
            "completed_fixture_ids": {601},
            "completed_results_fixture_ids": {601},
            "completed_odds_fixture_ids": {601},
        }
        seed.save_checkpoint(state)
        on_disk = json.loads(cp.read_text())
        assert on_disk.get("schema_version") == seed.CHECKPOINT_SCHEMA_VERSION
        reloaded = seed.load_checkpoint()
        assert reloaded == state
