"""Snapshot cache for Sportmonks fixtures.

We persist every fixture pull as a timestamped JSON snapshot so:
- Historical replay is possible (drive predictor over past frames)
- Lossless audit (raw API payload preserved for forensics)
- Backtesting needs only the snapshots, not live API access

Layout::

    data/cache/sportmonks/
        snapshots/
            {fixture_id}/
                {iso_timestamp}.json   — raw Fixture JSON
        catalog/
            {date}.parquet             — daily catalog: fid → metadata
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.sports.football.sportmonks.schemas import Fixture

logger = logging.getLogger(__name__)


DEFAULT_CACHE_ROOT = Path("data/cache/sportmonks")


class SportmonksCache:
    """File-system cache for fixture snapshots."""

    def __init__(self, root: Path = DEFAULT_CACHE_ROOT) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.catalog_dir = self.root / "catalog"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

    # ── persistence ──────────────────────────────────────────────────────

    def save_snapshot(
        self,
        fixture_id: int,
        payload: dict[str, Any],
        *,
        captured_at: datetime | None = None,
    ) -> Path:
        """Write the raw API payload for one fixture to disk.

        The filename is the ISO-8601 UTC timestamp (sortable).
        """
        captured_at = captured_at or datetime.now(timezone.utc)
        # Filesystem-safe timestamp: replace ':' with '-'
        ts = captured_at.strftime("%Y%m%dT%H%M%SZ")
        target_dir = self.snapshots_dir / str(fixture_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{ts}.json"
        path.write_text(json.dumps(payload, default=str))
        return path

    def list_fixtures(self) -> list[int]:
        return sorted(
            int(p.name) for p in self.snapshots_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )

    def list_snapshots(self, fixture_id: int) -> list[Path]:
        d = self.snapshots_dir / str(fixture_id)
        if not d.exists():
            return []
        return sorted(d.glob("*.json"))

    def load_snapshot(self, path: Path) -> Fixture:
        raw = json.loads(path.read_text())
        # API response wraps record in 'data'; we may have stored either form
        record = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
        return Fixture.model_validate(record)

    def load_latest_snapshot(self, fixture_id: int) -> Fixture | None:
        snaps = self.list_snapshots(fixture_id)
        if not snaps:
            return None
        return self.load_snapshot(snaps[-1])

    def load_all_snapshots(self, fixture_id: int) -> list[Fixture]:
        return [self.load_snapshot(p) for p in self.list_snapshots(fixture_id)]
