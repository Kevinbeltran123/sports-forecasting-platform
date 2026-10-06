"""Tests del log de scoring del análisis (item ② roadmap next-gen)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bip.evaluation.tournaments.team_style_profiler.analysis_log import (
    AnalysisRead,
    append_read,
    grade_read,
    load_reads,
    slice_log,
)


def _read(**kw) -> AnalysisRead:
    base = dict(
        id="r1", date="2026-06-11", fixture="Mexico vs South Africa",
        regime="internacional", coverage="alta", read_level="team",
        market="córners", selection="Mexico +2.0 AH córners", tier="secondary",
        logged_at="2026-06-11T18:00:00+00:00",
    )
    base.update(kw)
    return AnalysisRead(**base)


def test_append_and_load_roundtrip(tmp_path):
    p = tmp_path / "log.jsonl"
    append_read(_read(id="a"), path=p)
    append_read(_read(id="b", market="tarjetas"), path=p)
    reads = load_reads(p)
    assert [r.id for r in reads] == ["a", "b"]
    assert reads[1].market == "tarjetas"


def test_load_missing_file_returns_empty(tmp_path):
    assert load_reads(tmp_path / "nope.jsonl") == []


def test_grade_updates_status_and_roi(tmp_path):
    p = tmp_path / "log.jsonl"
    append_read(_read(id="a", stake=2.0), path=p)
    assert grade_read("a", "HIT", roi=1.8, path=p) is True
    r = load_reads(p)[0]
    assert r.status == "HIT" and r.roi == 1.8


def test_grade_unknown_id_returns_false(tmp_path):
    p = tmp_path / "log.jsonl"
    append_read(_read(id="a"), path=p)
    assert grade_read("zzz", "HIT", path=p) is False


def test_slice_ignores_pending_and_void(tmp_path):
    reads = [
        _read(id="a", status="HIT", roi=1.0),
        _read(id="b", status="MISS", roi=-1.0),
        _read(id="c", status="pending"),
        _read(id="d", status="void"),
    ]
    res = slice_log(reads, by=("read_level",))
    assert res[("team",)]["n"] == 2  # solo HIT/MISS
    assert res[("team",)]["hits"] == 1
    assert res[("team",)]["hit_rate"] == 0.5
    assert res[("team",)]["roi_sum"] == 0.0
    assert res[("team",)]["roi_per_bet"] == 0.0


def test_slice_groups_by_multiple_fields():
    reads = [
        _read(id="a", read_level="team", tier="secondary", status="HIT", roi=2.0),
        _read(id="b", read_level="prop", tier="secondary", status="MISS", roi=-1.0),
        _read(id="c", read_level="team", tier="secondary", status="HIT", roi=1.0),
    ]
    res = slice_log(reads, by=("read_level", "tier"))
    assert res[("team", "secondary")]["n"] == 2
    assert res[("team", "secondary")]["hit_rate"] == 1.0
    assert res[("team", "secondary")]["roi_per_bet"] == 1.5
    assert res[("prop", "secondary")]["hit_rate"] == 0.0


def test_slice_roi_none_when_no_roi_rows():
    reads = [_read(id="a", status="HIT"), _read(id="b", status="MISS")]
    res = slice_log(reads, by=("regime",))
    assert res[("internacional",)]["roi_sum"] is None
    assert res[("internacional",)]["roi_per_bet"] is None


# Boundary / validación de enums : cada campo Literal
# acepta sus valores válidos y rechaza el resto.
@pytest.mark.parametrize("field,good,bad", [
    ("regime", "amistoso", "friendly"),
    ("coverage", "media", "high"),
    ("read_level", "prop", "player"),
    ("tier", "efficient", "main"),
    ("status", "void", "PENDING"),
])
def test_enum_validation(field, good, bad):
    assert getattr(_read(**{field: good}), field) == good
    with pytest.raises(ValidationError):
        _read(**{field: bad})


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        _read(bogus="x")
