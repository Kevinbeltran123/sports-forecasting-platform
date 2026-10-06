"""Grade v3 shadow picks against final fixture outcomes.

**Canonical grading path.** This module is the SINGLE authoritative
grader for v3 picks. All grading that needs fresh Sportmonks data must
go through ``grade_picks_for_date`` (or the callable API
``run_grade_for_date`` which adds write-to-disk). Do NOT build ad-hoc
graders that read gsv_log.parquet to reconstruct outcomes — that creates
circularity between the logging layer and the measurement layer.

Scripts that previously implemented their own grading loops
(grade_day4_real.py, regrade_day4_from_raw.py) are DEPRECATED and
kept only for historical reproducibility. Use this module instead.

Closes the operational gap: cohort_accountant + calibration analyses
consume ``picks_outcomes.parquet``, but nothing populates it. v2 has
``analyze_jornada`` for its own picks, but v3 emits market families
(corners, cards, next_goal) that v2's grader never knew about. This
module is the v3-side grader.

# Design choices

- Reuses ``FinalOutcome`` + ``grade_pick`` from the v2 grading module
  (battle-tested, includes the 2026-05-10 audit fix for ``ou_under <
  line + 1``). For the families v2 didn't grade (next_goal direction,
  props) we add v3-specific logic here.

- Builds ``LiveMatchState`` from the Sportmonks ``Fixture`` via the
  existing ``LiveMatchState.from_fixture`` factory, then derives
  ``FinalOutcome`` via ``derive_final_outcome_from_state``. Single
  source of truth for "final state of a fixture".

  The authoritative fetch uses:
    includes=["participants", "state", "periods", "scores",
              "statistics", "events"]
  This is the canonical include set — do NOT omit any of these.

- Idempotent: writing ``picks_outcomes.parquet`` for the same day twice
  produces the same file. The grader does NOT modify ``picks.parquet``.

- No circularity: ``grade_picks_for_date`` reads ONLY
  ``picks.parquet`` for pick rows, then fetches live fixture data
  from Sportmonks. It never reads ``gsv_log.parquet``.

- Defensive on ungradable picks: when grade_pick returns None (void or
  unknown market), we emit status="void" with profit_units=0. When the
  fixture state isn't yet final, status="pending" with profit_units=0.

# CLI

An offline CLI (not published) is the entrypoint. It reads ``data/cache/v3_shadow/dt=YYYY-MM-DD/picks.parquet``, fetches
each unique fixture's FT state via Sportmonks, grades every pick, and
writes ``picks_outcomes.parquet`` to the same partition.

# Callable API for Wave-2 promotion report

Use ``run_grade_for_date(date_iso, shadow_root=..., client=...)``
to run end-to-end (grade + write parquet) from another script without
going through the CLI. Returns ``(graded_picks, report, out_path)``
where ``out_path`` is the written parquet (or None on dry-run).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bip.evaluation.live.grading import (
    FinalOutcome,
    derive_final_outcome_from_state,
    grade_pick,
)
from bip.evaluation.live.match_state import LiveMatchState
from bip.sports.football.sportmonks.schemas import Fixture

log = logging.getLogger("v3.grader")


# ──────────────────────────────────────────────────────────────────────
# Result records
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradedPick:
    """One graded outcome for a v3 pick. Maps 1:1 to picks_outcomes.parquet
    schema documented in cohort_accountant.load_v3_outcomes."""

    fixture_id: int
    thesis_id: str
    market_id: str
    pick_timestamp_utc: datetime
    status: str  # "won" | "lost" | "void" | "pending"
    profit_units: float
    bookmaker_odd: float | None
    settled_at: datetime


@dataclass(frozen=True)
class GradeReport:
    """Diagnostics emitted by the grader for the operator's daily review."""

    n_picks_input: int
    n_fixtures: int
    n_fixtures_finished: int
    n_won: int
    n_lost: int
    n_void: int
    n_pending: int
    n_ungradable: int  # market_id couldn't be mapped to a grader path


# ──────────────────────────────────────────────────────────────────────
# Market_id → (market, selection) translator
# ──────────────────────────────────────────────────────────────────────


_LINE_RE = re.compile(r"(\d+\.?\d*)")


def _parse_line(market_id: str) -> float | None:
    """Pull the first numeric value out of a market_id like
    ``match_goals_over_2.5`` → 2.5. Returns None if none present.
    """
    m = _LINE_RE.search(market_id)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _format_v2_line_key(prefix: str, line: float) -> str:
    """Produce the v2 grade_pick market key format from a v3 line value.

    v2 uses underscores between int and decimal parts (e.g., 2.5 →
    ``"2_5"``). _parse_total_line in grading.py parses that back.
    """
    int_part = int(line)
    dec_part = int(round((line - int_part) * 10))
    return f"{prefix}{int_part}_{dec_part}"


def grade_v3_pick(
    pick_row: dict[str, Any], outcome: FinalOutcome,
) -> tuple[str, bool | None]:
    """Grade one pick row using the v3 family / market_id taxonomy.

    Returns ``(reason, won)`` where:
        reason: "graded" | "next_goal" | "ungradable"
        won: True | False | None (None = void / unknown)
    """
    family = pick_row.get("family") or ""
    market_id = pick_row.get("market_id") or ""
    direction = (pick_row.get("direction") or "").lower()
    line = pick_row.get("line_value")
    if line is None:
        line = _parse_line(market_id)

    # Goals
    if family == "goals" and line is not None:
        is_1h = "first_half" in market_id
        key = _format_v2_line_key("first_half_ou_" if is_1h else "ou_", line)
        return ("graded", grade_pick(key, direction, outcome))

    # BTTS
    if family == "btts":
        # market_id format examples: btts_yes, both_teams_to_score_no,
        # btts_first_half_yes, btts_second_half_no
        if "first_half" in market_id:
            return ("graded", grade_pick("btts_first_half", direction, outcome))
        if "second_half" in market_id:
            return ("graded", grade_pick("btts_second_half", direction, outcome))
        return ("graded", grade_pick("btts", direction, outcome))

    # Corners (v3 uses match_corners_over_9.5, second_half_corners_over_4.5)
    if family == "corners" and line is not None:
        key = _format_v2_line_key("corners_total_", line)
        return ("graded", grade_pick(key, direction, outcome))

    # Cards
    if family == "cards" and line is not None:
        key = _format_v2_line_key("cards_total_", line)
        return ("graded", grade_pick(key, direction, outcome))

    # 1X2 (fulltime_result)
    if family == "result_1x2":
        return ("graded", grade_pick("fulltime_result", direction, outcome))

    # Next goal — uses goal_events timeline + pick minute (activated_at_minute)
    if family == "next_goal":
        pick_minute = int(pick_row.get("activated_at_minute") or 0)
        won = _grade_next_goal(direction, pick_minute, outcome)
        return ("next_goal", won)

    # Unknown family or insufficient info — caller treats as ungradable
    return ("ungradable", None)


def _grade_next_goal(
    direction: str, pick_minute: int, outcome: FinalOutcome,
) -> bool | None:
    """Grade a next-goal pick.

    direction == 'no'    → win iff no goals after pick_minute
    direction == 'home'  → win iff first goal after pick_minute is by home
    direction == 'away'  → win iff first goal after pick_minute is by away
    other                → None (ungradable)
    """
    after = [(m, t) for m, t in outcome.goal_events if m > pick_minute]
    if direction == "no":
        return len(after) == 0
    if not after:
        # Direction was home/away but no further goals happened — the
        # bet doesn't resolve in the bettor's favor. Most books refund
        # next-goal markets at FT with no goals (void); we conservatively
        # mark as lost so the calibration doesn't get an inflated WR
        # from voids treated as wins.
        return False
    first_team = after[0][1]
    if direction == "home":
        return first_team == outcome.home_team_id
    if direction == "away":
        return first_team == outcome.away_team_id
    return None


# ──────────────────────────────────────────────────────────────────────
# Profit computation
# ──────────────────────────────────────────────────────────────────────


def profit_units_for(won: bool | None, book_odd: float | None) -> float:
    """Stake-unit P/L for one pick.

    won=True   → odd-1   (gross profit per 1u stake)
    won=False  → -1      (lost the stake)
    won=None   → 0       (void / refunded)
    book_odd missing → 0 (can't compute, treat as void for accounting)
    """
    if won is None or book_odd is None:
        return 0.0
    if won:
        return float(book_odd) - 1.0
    return -1.0


def status_for(won: bool | None) -> str:
    if won is None:
        return "void"
    return "won" if won else "lost"


# ──────────────────────────────────────────────────────────────────────
# Outcome-derivation helper for a fixture
# ──────────────────────────────────────────────────────────────────────


def final_outcome_from_fixture(fixture: Fixture) -> FinalOutcome | None:
    """Build a FinalOutcome from a Sportmonks fixture. Returns None if
    the fixture isn't finished yet. The state factory is shared with
    the live-pipeline so we don't reinvent the score-extraction logic
    (and inherit the 2026-05-11 type_id fix automatically)."""
    try:
        state = LiveMatchState.from_fixture(fixture)
    except Exception as exc:  # noqa: BLE001
        log.warning("from_fixture_failed fixture=%s err=%s", fixture.id, exc)
        return None
    return derive_final_outcome_from_state(state, fixture=fixture)


# ──────────────────────────────────────────────────────────────────────
# Batch grading
# ──────────────────────────────────────────────────────────────────────


async def grade_picks_for_date(
    date_iso: str,
    *,
    shadow_root: Path,
    client: Any,
) -> tuple[list[GradedPick], GradeReport]:
    """Grade every pick in ``picks.parquet`` for one daily partition.

    ``client`` is a SportmonksClient (or compatible duck-type for tests
    — anything with ``get_fixture(fixture_id, includes=...)``).
    """
    import polars as pl

    partition = shadow_root / f"dt={date_iso}"
    picks_path = partition / "picks.parquet"
    if not picks_path.exists():
        return [], GradeReport(0, 0, 0, 0, 0, 0, 0, 0)

    df = pl.read_parquet(picks_path)
    rows = list(df.iter_rows(named=True))
    if not rows:
        return [], GradeReport(0, 0, 0, 0, 0, 0, 0, 0)

    fixture_ids = sorted({int(r["fixture_id"]) for r in rows})
    outcomes_by_fid: dict[int, FinalOutcome | None] = {}
    n_finished = 0
    for fid in fixture_ids:
        try:
            fx = await client.get_fixture(
                fid,
                includes=["participants", "state", "periods", "scores",
                          "statistics", "events"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("get_fixture_failed fixture=%d err=%s", fid, exc)
            outcomes_by_fid[fid] = None
            continue
        outcome = final_outcome_from_fixture(fx)
        outcomes_by_fid[fid] = outcome
        if outcome is not None:
            n_finished += 1

    graded: list[GradedPick] = []
    n_won = n_lost = n_void = n_pending = n_ungradable = 0
    now = datetime.now(timezone.utc)

    for r in rows:
        fid = int(r["fixture_id"])
        outcome = outcomes_by_fid.get(fid)
        if outcome is None:
            # Fixture not yet finished (or fetch failed) — mark pending
            graded.append(GradedPick(
                fixture_id=fid,
                thesis_id=str(r["thesis_id"]),
                market_id=str(r["market_id"]),
                pick_timestamp_utc=r["timestamp_utc"],
                status="pending",
                profit_units=0.0,
                bookmaker_odd=r.get("bookmaker_odd"),
                settled_at=now,
            ))
            n_pending += 1
            continue

        reason, won = grade_v3_pick(r, outcome)
        if reason == "ungradable":
            n_ungradable += 1
            graded.append(GradedPick(
                fixture_id=fid,
                thesis_id=str(r["thesis_id"]),
                market_id=str(r["market_id"]),
                pick_timestamp_utc=r["timestamp_utc"],
                status="void",  # unknown market → treat as void for accounting
                profit_units=0.0,
                bookmaker_odd=r.get("bookmaker_odd"),
                settled_at=now,
            ))
            n_void += 1
            continue

        status = status_for(won)
        profit = profit_units_for(won, r.get("bookmaker_odd"))
        graded.append(GradedPick(
            fixture_id=fid,
            thesis_id=str(r["thesis_id"]),
            market_id=str(r["market_id"]),
            pick_timestamp_utc=r["timestamp_utc"],
            status=status,
            profit_units=profit,
            bookmaker_odd=r.get("bookmaker_odd"),
            settled_at=now,
        ))
        if status == "won":
            n_won += 1
        elif status == "lost":
            n_lost += 1
        else:
            n_void += 1

    report = GradeReport(
        n_picks_input=len(rows),
        n_fixtures=len(fixture_ids),
        n_fixtures_finished=n_finished,
        n_won=n_won,
        n_lost=n_lost,
        n_void=n_void,
        n_pending=n_pending,
        n_ungradable=n_ungradable,
    )
    return graded, report


def write_outcomes_parquet(
    graded: list[GradedPick], partition: Path,
) -> Path:
    """Write the GradedPick list to picks_outcomes.parquet in the
    daily partition. Idempotent — overwrites in place."""
    import polars as pl

    partition.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "fixture_id": int(g.fixture_id),
            "thesis_id": g.thesis_id,
            "market_id": g.market_id,
            "pick_timestamp_utc": g.pick_timestamp_utc,
            "status": g.status,
            "profit_units": float(g.profit_units),
            "bookmaker_odd": (
                None if g.bookmaker_odd is None else float(g.bookmaker_odd)
            ),
            "settled_at": g.settled_at,
        }
        for g in graded
    ]
    path = partition / "picks_outcomes.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


async def run_grade_for_date(
    date_iso: str,
    *,
    shadow_root: Path,
    client: Any,
    dry_run: bool = False,
) -> tuple[list[GradedPick], GradeReport, Path | None]:
    """Callable API: grade + optionally write picks_outcomes.parquet.

    Convenience wrapper around ``grade_picks_for_date`` +
    ``write_outcomes_parquet`` for callers that don't want to use the
    CLI (e.g., the Wave-2 shadow-promotion report).

    Returns ``(graded_picks, report, out_path)`` where ``out_path`` is
    the written parquet path (or ``None`` when ``dry_run=True`` or when
    there are no picks to grade).
    """
    graded, report = await grade_picks_for_date(
        date_iso, shadow_root=shadow_root, client=client,
    )
    if not graded or dry_run:
        return graded, report, None
    partition = shadow_root / f"dt={date_iso}"
    out_path = write_outcomes_parquet(graded, partition)
    return graded, report, out_path


__all__ = [
    "GradeReport",
    "GradedPick",
    "_grade_next_goal",
    "final_outcome_from_fixture",
    "grade_picks_for_date",
    "grade_v3_pick",
    "profit_units_for",
    "run_grade_for_date",
    "status_for",
    "write_outcomes_parquet",
]
