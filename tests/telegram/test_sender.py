"""GREEN tests for PICK-04 (D-12 HTML, D-14 FLAG marker, D-15 bullet cap, T-3-XSS escape)."""
from __future__ import annotations

from bip.core.storage.models import Pick


def _make_pick(claude_validation: str | None = "CONFIRM",
               claude_summary: str | None = "good form * weak away defense",
               home: str = "Manchester United",
               away: str = "Chelsea") -> Pick:
    return Pick(
        fixture_id=12345,
        league="premier_league",
        market="1X2",
        selection="1",
        model_probability=0.55,
        implied_probability=0.45,
        edge=0.08,
        best_odds=2.10,
        bookmaker="betano",
        suggested_stake=1.5,
        claude_validation=claude_validation,
        claude_summary=claude_summary,
    )


class TestRender:
    def test_render_escapes_special_chars(self):
        """T-3-XSS: <script> in team name HTML-escaped via Jinja2 autoescape."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick()
        out = render_pick(pick, home_team="<script>alert(1)</script>", away_team="Arsenal",
                          model_version="2026-05-02-1")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_flag_warning_marker(self):
        """D-14: claude_validation='FLAG' → WARNING (U+26A0) entity + 'Claude flagged' line."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick(claude_validation="FLAG")
        out = render_pick(pick, home_team="Arsenal", away_team="Liverpool",
                          model_version="v1", reason_code="aggregate_stat_seduction")
        assert "&#x26A0;" in out
        assert "Claude flagged:" in out
        assert "aggregate_stat_seduction" in out

    def test_confirm_clean(self):
        """D-14: CONFIRM → no WARNING marker."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick(claude_validation="CONFIRM")
        out = render_pick(pick, home_team="Arsenal", away_team="Liverpool", model_version="v1")
        assert "&#x26A0;" not in out
        assert "Claude flagged:" not in out

    def test_summary_bullets_max_3(self):
        """D-15: split on ' * '; cap at 3."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick(claude_summary="b1 * b2 * b3 * b4 * b5")
        out = render_pick(pick, home_team="A", away_team="B", model_version="v1")
        assert out.count("• ") == 3

    def test_summary_split_separator(self):
        """D-15: separator is literal ' * '."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick(claude_summary="bullet one * bullet two")
        out = render_pick(pick, home_team="A", away_team="B", model_version="v1")
        assert out.count("• ") == 2

    def test_empty_summary_no_bullets(self):
        from bip.core.telegram.sender import render_pick
        pick = _make_pick(claude_summary=None)
        out = render_pick(pick, home_team="A", away_team="B", model_version="v1")
        assert "• " not in out

    def test_message_length_under_600_chars(self):
        """specifics §194: total under ~600 chars for mobile rendering (700 soft cap)."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick(claude_summary="b1 * b2 * b3")
        out = render_pick(pick, home_team="Manchester United", away_team="Chelsea FC",
                          model_version="2026-05-02-PL-v3")
        assert len(out) <= 700

    def test_html_parse_mode_allowed_tags_only(self):
        """D-12: Telegram HTML parse_mode supports <b>,<i>,<u>,<s>,<a>,<code>,<pre> only."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick()
        out = render_pick(pick, home_team="A", away_team="B", model_version="v1")
        for forbidden in ("<div", "<span", "<p>", "</p>"):
            assert forbidden not in out.lower()

    def test_stake_none_renders_zero(self):
        """Filtered/rejected picks have suggested_stake=None — must render as 0.0u."""
        from bip.core.telegram.sender import render_pick
        pick = _make_pick()
        pick.suggested_stake = None
        out = render_pick(pick, home_team="A", away_team="B", model_version="v1")
        assert "0.0u" in out
