"""Tests for the Transfermarkt squad + injury parsers (offline)."""
from __future__ import annotations

from datetime import date

import pytest

from bip.evaluation.tournaments.team_style_profiler.transfermarkt_scrape import (
    InjuryRecord,
    current_injury_status,
    parse_injuries,
    parse_search_for_team,
    parse_squad,
)

_SQUAD_HTML = """
<table class="items"><tbody>
  <tr><td><a href="/erling-haaland/profil/spieler/418560">Erling Haaland</a></td></tr>
  <tr><td><a href="/martin-odegaard/profil/spieler/316264">Martin Ødegaard</a></td></tr>
  <tr><td><a href="/erling-haaland/profil/spieler/418560">Erling Haaland</a></td></tr>
  <tr><td>staff row, no player link</td></tr>
</tbody></table>
"""

_INJURY_HTML = """
<table class="items"><thead><tr><th>Season</th><th>Injury</th><th>from</th>
<th>until</th><th>Days</th><th>Games missed</th></tr></thead><tbody>
  <tr><td>25/26</td><td>Knock</td><td>26/02/2026</td><td>01/03/2026</td><td>4 days</td><td>1</td></tr>
  <tr><td>24/25</td><td>Ankle injury</td><td>31/03/2025</td><td>01/05/2025</td><td>32 days</td><td>6</td></tr>
</tbody></table>
"""


def test_parse_squad_extracts_unique_players():
    squad = parse_squad(_SQUAD_HTML)
    assert len(squad) == 2                      # dedup the repeated Haaland, skip staff row
    assert squad[0].name == "Erling Haaland"
    assert squad[0].tm_id == "418560"
    assert squad[0].profile_path == "/erling-haaland/profil/spieler/418560"


def test_parse_squad_empty_when_no_table():
    assert parse_squad("<html><body>no table</body></html>") == []


def test_parse_injuries_rows():
    recs = parse_injuries(_INJURY_HTML)
    assert len(recs) == 2
    assert recs[0] == InjuryRecord("25/26", "Knock", "26/02/2026", "01/03/2026", "4 days", "1")
    assert recs[1].injury == "Ankle injury"


# ── current_injury_status — boundary-parametrized on the `until` date ──


@pytest.mark.parametrize("until,today,expected_injured", [
    ("01/06/2026", date(2026, 5, 30), True),    # returns in the future → out
    ("30/05/2026", date(2026, 5, 30), True),    # until == today (inclusive) → out
    ("29/05/2026", date(2026, 5, 30), False),   # returned yesterday → fit
    ("01/03/2026", date(2026, 5, 30), False),   # long healed → fit
])
def test_current_injury_status_until_boundary(until, today, expected_injured):
    recs = [InjuryRecord("25/26", "Knock", "01/01/2026", until, "x", "1")]
    st = current_injury_status(recs, today)
    assert st.injured is expected_injured


@pytest.mark.parametrize("until_raw", ["-", "?", "", "Unknown"])
def test_ongoing_injury_no_return_date(until_raw):
    recs = [InjuryRecord("25/26", "Cruciate ligament tear", "01/05/2026", until_raw, "x", "x")]
    st = current_injury_status(recs, date(2026, 5, 30))
    assert st.injured is True
    assert st.ongoing is True
    assert st.until is None
    assert st.injury == "Cruciate ligament tear"


def test_no_records_means_fit():
    st = current_injury_status([], date(2026, 5, 30))
    assert st.injured is False and st.last_injury is None


# ── Search resolver: senior national team over youth sides ──

_SEARCH_HTML = """
<div>
  <a href="/usbekistan-u23/startseite/verein/33815">Uzbekistan U23</a>
  <a href="/usbekistan/startseite/verein/3563">Uzbekistan</a>
  <a href="/usbekistan-u20/startseite/verein/22990">Uzbekistan U20</a>
</div>
"""


def test_resolver_picks_senior_team_not_youth():
    path = parse_search_for_team(_SEARCH_HTML, "Uzbekistan")
    assert path == "/usbekistan/kader/verein/3563"   # senior, kader path


def test_resolver_handles_and_and_accents():
    html = '<a href="/bosnien-herzegowina/startseite/verein/3457">Bosnia-Herzegovina</a>'
    assert parse_search_for_team(html, "Bosnia and Herzegovina") == "/bosnien-herzegowina/kader/verein/3457"


def test_resolver_none_when_no_match():
    html = '<a href="/some-club/startseite/verein/999">Random Club</a>'
    assert parse_search_for_team(html, "Norway") is None


def test_uses_most_recent_record_only():
    # healed latest injury, older one irrelevant → fit
    recs = [
        InjuryRecord("25/26", "Knock", "01/02/2026", "10/02/2026", "x", "1"),
        InjuryRecord("24/25", "ACL", "01/01/2025", "-", "x", "x"),
    ]
    assert current_injury_status(recs, date(2026, 5, 30)).injured is False
