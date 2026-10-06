"""Canonical player identity reconciliation across data sources.

For v1 of the spike, the only source is API-Football. The canonical_id is
simply "af-{api_football_id}". This thin layer exists so future augmentation
(FBref, StatsBomb, manual overrides) can be added without changing the
parquet schemas — every cached row already keys off canonical_id.

R-01 mitigation (SPIKE.md §5): start simple, expand only if augmentation
triggers in v2.
"""

from __future__ import annotations

import re

_AF_CANONICAL_PATTERN = re.compile(r"^af-(\d+)$")


def to_canonical(api_football_id: int) -> str:
    """Convert API-Football player ID -> canonical ID."""
    if api_football_id < 1:
        raise ValueError(f"api_football_id must be positive, got {api_football_id}")
    return f"af-{api_football_id}"


def from_canonical(canonical_id: str) -> int:
    """Extract API-Football player ID from canonical ID.

    Raises ValueError if the canonical_id does not start with 'af-'.
    """
    match = _AF_CANONICAL_PATTERN.match(canonical_id)
    if not match:
        raise ValueError(
            f"canonical_id {canonical_id!r} not in 'af-<int>' form. "
            "Multi-source matching not implemented in v1."
        )
    return int(match.group(1))


def is_api_football_canonical(canonical_id: str) -> bool:
    """Predicate: is this canonical_id an API-Football-only ID?"""
    return bool(_AF_CANONICAL_PATTERN.match(canonical_id))
