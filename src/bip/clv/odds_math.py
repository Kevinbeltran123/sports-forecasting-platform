"""Odds math primitives for CLV computation.

Pinnacle h2h prices include a bookmaker margin (the "vig" or overround) of
roughly 2-4%. Comparing raw odds against pick odds inflates CLV systematically
and biases the project's primary success metric upward -- documented as
PITFALLS.md Pitfall 7 (Error #1 of CLV).

`remove_vig` performs proportional (a.k.a. "basic" or "Shin-free") vig removal:
each outcome's implicit probability is rescaled by the overround so the fair
probabilities sum to exactly 1.0. The function is pure -- no I/O, no logging,
no external dependencies beyond stdlib -- so it is safe to call from any layer.
"""

from __future__ import annotations


def remove_vig(odds: dict[str, float]) -> dict[str, float]:
    """Return fair (vig-removed) decimal odds for a market.

    Each input odd is converted to its implicit probability (1/odd), then the
    bookmaker overround is divided out proportionally so that the resulting
    probabilities sum to exactly 1.0. The fair odds returned are the reciprocals
    of those normalised probabilities.

    Args:
        odds: Mapping of outcome key -> raw decimal odd. Keys are
            caller-defined (e.g., "1"/"X"/"2", "home"/"draw"/"away",
            "over"/"under"); the function is key-agnostic.

    Returns:
        Mapping with the same key set, values being the vig-removed decimal
        odds. The implicit probabilities (1/value) sum to 1.0.

    Raises:
        ValueError: If `odds` is empty, or if any value is <= 1.0
            (decimal odds must strictly exceed 1.0 to represent a real market).
    """
    if not odds:
        raise ValueError("remove_vig: empty odds dict")
    for k, v in odds.items():
        if v <= 1.0:
            raise ValueError(
                f"remove_vig: invalid odd for {k!r}: {v} (must be > 1.0)"
            )
    implicit = {k: 1.0 / v for k, v in odds.items()}
    total = sum(implicit.values())  # > 1.0 due to bookmaker overround
    fair_probs = {k: p / total for k, p in implicit.items()}
    return {k: 1.0 / p for k, p in fair_probs.items()}
