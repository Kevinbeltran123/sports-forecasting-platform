"""Tests for telegram_alerts formatters."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from bip.evaluation.live.telegram_alerts import (
    classify_tier,
    format_burst_digest,
    format_jornada_summary,
    format_outcome_reply,
    format_pick_alert,
    format_pick_alert_with_tier,
    format_pre_jornada_brief,
    format_scoreboard,
    send_jornada_summary,
    send_pick_alert,
)
from bip.evaluation.live.value_detector import LivePick


def _make_pick(
    *,
    home="Real Madrid", away="Barcelona", market="ou_3_5", selection="under",
    minute=25, odd=1.85, our_prob=0.78, raw_prob=0.71,
    edge_pct=18.5, stake=1.05, kelly_full=0.42,
    logical_score=0.85, components=None, flagged=None,
) -> LivePick:
    return LivePick(
        fixture_id=1, minute=minute,
        home_team=home, away_team=away,
        market=market, selection=selection,
        bookmaker_id=2, bookmaker_odd=odd,
        our_probability=our_prob,
        fair_odd=1.0 / our_prob,
        edge_pct=edge_pct,
        kelly_fraction_full=kelly_full,
        suggested_stake_pct=stake,
        snapshot_kind="live",
        flagged_reason=flagged,
        logical_score=logical_score,
        logical_components=components or {},
        confidence_half_width=0.05,
        model_probability_raw=raw_prob,
    )


class TestFormatPickAlert:
    def test_basic_fields_in_output(self):
        pick = _make_pick()
        out = format_pick_alert(pick)
        assert "Real Madrid" in out
        assert "Barcelona" in out
        assert "ou_3_5" in out
        assert "under" in out
        assert "1.85" in out
        assert "18.5" in out  # edge

    def test_score_included_when_provided(self):
        pick = _make_pick(minute=42)
        out = format_pick_alert(pick, home_score=1, away_score=0)
        assert "1-0" in out
        assert "42'" in out

    def test_score_omitted_when_not_provided(self):
        pick = _make_pick()
        out = format_pick_alert(pick)
        # No score block when not passed
        assert "Score" not in out

    def test_league_included(self):
        pick = _make_pick()
        out = format_pick_alert(pick, league_name="La Liga")
        assert "La Liga" in out

    def test_calibrated_vs_raw_shown_when_differ(self):
        pick = _make_pick(our_prob=0.78, raw_prob=0.71)
        out = format_pick_alert(pick)
        assert "Cal prob" in out
        assert "0.780" in out
        assert "0.710" in out

    def test_single_prob_when_equal(self):
        pick = _make_pick(our_prob=0.71, raw_prob=0.71)
        out = format_pick_alert(pick)
        assert "Cal prob" not in out
        assert "Prob" in out

    def test_flagged_reason_shown(self):
        pick = _make_pick(flagged="extreme_edge_no_sm_confirmation")
        out = format_pick_alert(pick)
        assert "FLAGGED" in out
        assert "extreme_edge_no_sm_confirmation" in out

    def test_no_flag_block_when_clean(self):
        pick = _make_pick(flagged=None)
        out = format_pick_alert(pick)
        assert "FLAGGED" not in out

    def test_logical_components_top2_shown(self):
        pick = _make_pick(
            components={"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3},
        )
        out = format_pick_alert(pick)
        # Top 2 by abs value should appear
        assert "a" in out
        assert "b" in out

    def test_html_escapes_special_chars_in_names(self):
        # Defensive: team names should be escaped to prevent injection
        pick = _make_pick(home="<script>alert(1)</script>", away="B")
        out = format_pick_alert(pick)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_kelly_diagnostic_field_shown(self):
        pick = _make_pick(kelly_full=0.42)
        out = format_pick_alert(pick)
        assert "0.42" in out


class TestFormatJornadaSummary:
    def test_basic_summary(self):
        out = format_jornada_summary(
            jornada_date="2026-05-13",
            n_emit_total=487,
            n_emit_settled=425,
            n_won=270,
            profit_units=123.5,
            stake_pct_total=420.3,
        )
        assert "2026-05-13" in out
        assert "487" in out
        assert "270" in out
        # Win rate 270/425 = 63.5%
        assert "63.5" in out
        # ROI 123.5/420.3 = 29.4%
        assert "29.38" in out or "29.4" in out

    def test_top_n_block_included_when_provided(self):
        out = format_jornada_summary(
            jornada_date="2026-05-13",
            n_emit_total=487, n_emit_settled=425, n_won=270,
            profit_units=123.5, stake_pct_total=420.3,
            top_n_stats={"n": 25, "won": 20, "profit": 15.8, "roi": 63.2},
        )
        assert "Top-N" in out
        assert "63.2" in out
        assert "15.8" in out

    def test_drops_by_reason_excludes_below_min_edge(self):
        out = format_jornada_summary(
            jornada_date="2026-05-13",
            n_emit_total=487, n_emit_settled=425, n_won=270,
            profit_units=123.5, stake_pct_total=420.3,
            drops_by_reason={
                "below_min_edge": 200,  # excluded (too noisy)
                "market_blacklist": 18,
                "positive_side_binary_ban": 23,
                "commentary_cooloff": 12,
            },
        )
        assert "market_blacklist" in out
        assert "positive_side_binary_ban" in out
        assert "commentary_cooloff" in out
        # below_min_edge is excluded from the displayed list (count 200
        # should NOT appear). The string itself may appear in the header
        # "(excl. below_min_edge)".
        assert "200" not in out

    def test_best_market_included(self):
        out = format_jornada_summary(
            jornada_date="2026-05-13",
            n_emit_total=487, n_emit_settled=425, n_won=270,
            profit_units=123.5, stake_pct_total=420.3,
            best_market=("ou_3_5", 10, 9, 91.0),
        )
        assert "ou_3_5" in out
        assert "9/10" in out

    def test_calibrator_block_optional(self):
        # Without calibrator metadata
        out_no_cal = format_jornada_summary(
            jornada_date="2026-05-13",
            n_emit_total=10, n_emit_settled=8, n_won=5,
            profit_units=2.0, stake_pct_total=10.0,
        )
        assert "Calibrator" not in out_no_cal

        # With calibrator metadata
        out_cal = format_jornada_summary(
            jornada_date="2026-05-13",
            n_emit_total=10, n_emit_settled=8, n_won=5,
            profit_units=2.0, stake_pct_total=10.0,
            calibrator_ece_cv=0.052,
            calibrator_n_markets_own_fit=11,
        )
        assert "Calibrator" in out_cal
        assert "0.052" in out_cal
        assert "11" in out_cal


class TestPreJornadaBrief:
    def test_basic(self):
        out = format_pre_jornada_brief(
            jornada_date="2026-05-13", n_fixtures=42,
        )
        assert "2026-05-13" in out
        assert "42" in out

    def test_with_profile_and_band(self):
        out = format_pre_jornada_brief(
            jornada_date="2026-05-13", n_fixtures=42,
            profile_name="aggressive",
            expected_roi_band="+40% to +60%",
        )
        assert "aggressive" in out
        assert "+40% to +60%" in out


class TestSendHelpers:
    @pytest.mark.asyncio
    async def test_send_pick_alert_invokes_bot(self):
        bot = AsyncMock()
        bot.send_html = AsyncMock()
        pick = _make_pick()
        await send_pick_alert(bot, pick, home_score=1, away_score=0)
        bot.send_html.assert_awaited_once()
        sent = bot.send_html.await_args.args[0]
        assert "Real Madrid" in sent
        assert "1-0" in sent

    @pytest.mark.asyncio
    async def test_send_jornada_summary_invokes_bot(self):
        bot = AsyncMock()
        bot.send_html = AsyncMock()
        await send_jornada_summary(
            bot,
            jornada_date="2026-05-13",
            n_emit_total=487, n_emit_settled=425, n_won=270,
            profit_units=123.5, stake_pct_total=420.3,
        )
        bot.send_html.assert_awaited_once()


# ── V2: Tier classification boundary cases ──────────────────────────────────


class TestClassifyTier:
    """Tier rule (per design memo §B):
    - Tier 1: not flagged AND L >= 0.85 AND edge_pct >= 8.0
    - Tier 2: not flagged (and not Tier 1), OR flagged with L >= 0.80
    - Tier 3: flagged AND L < 0.80
    """

    @pytest.mark.parametrize("L,edge,expected", [
        (0.85, 8.0, 1),    # exact boundary — included
        (0.84, 12.0, 2),   # just below logical floor
        (0.85, 7.9, 2),    # just below edge floor
        (0.90, 50.0, 1),   # solidly inside
        (1.00, 0.0, 2),    # high L, no edge → drops to Tier 2
    ])
    def test_clean_pick_tier_boundary(self, L, edge, expected):
        p = _make_pick(logical_score=L, edge_pct=edge, flagged=None)
        assert classify_tier(p) == expected

    @pytest.mark.parametrize("L,expected", [
        (0.80, 2),         # exact boundary — flagged Tier 2
        (0.79, 3),         # just below flagged floor → Tier 3
        (0.95, 2),         # flagged but strong logical → still Tier 2
        (0.50, 3),         # flagged, weak logical → Tier 3
    ])
    def test_flagged_pick_tier_boundary(self, L, expected):
        p = _make_pick(logical_score=L, flagged="extreme_edge_no_sm_confirmation")
        assert classify_tier(p) == expected


class TestTierTemplates:
    def test_tier1_label_present(self):
        p = _make_pick(logical_score=0.90, edge_pct=12.0, flagged=None)
        out = format_pick_alert_with_tier(p, tier=1)
        assert "TIER 1" in out
        assert "HIGH-CONVICTION" in out
        # No "FLAGGED" word for clean picks
        assert "FLAGGED" not in out

    def test_tier2_label_present(self):
        p = _make_pick(logical_score=0.75, edge_pct=5.0, flagged=None)
        out = format_pick_alert_with_tier(p, tier=2)
        assert out.startswith("<b>PICK</b>")
        assert "FLAGGED" not in out  # clean

    def test_tier2_flagged_shows_reason(self):
        p = _make_pick(logical_score=0.82, flagged="stale_odd")
        out = format_pick_alert_with_tier(p, tier=2)
        assert "FLAGGED" in out
        assert "stale_odd" in out

    def test_tier3_uses_blockquote(self):
        p = _make_pick(logical_score=0.65, flagged="extreme_edge_no_sm_confirmation")
        out = format_pick_alert_with_tier(p, tier=3)
        assert "<blockquote>" in out
        assert "</blockquote>" in out
        assert "INFO" in out

    def test_tier3_includes_flag_reason(self):
        p = _make_pick(logical_score=0.65, flagged="positive_side_binary_ban")
        out = format_pick_alert_with_tier(p, tier=3)
        assert "positive_side_binary_ban" in out

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            format_pick_alert_with_tier(_make_pick(), tier=4)

    def test_dispatcher_picks_correct_tier(self):
        """Default _make_pick has L=0.85, edge=18.5 → Tier 1."""
        p = _make_pick()
        out = format_pick_alert(p)
        assert "TIER 1" in out

    def test_no_emojis_in_any_tier(self):
        """Operator directive: zero emojis in v2 output."""
        # Check standard unicode emoji ranges aren't present.
        import re
        emoji_re = re.compile(
            r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
        )
        for L, edge, flagged in [
            (0.90, 12.0, None),    # Tier 1
            (0.75, 5.0, None),     # Tier 2 clean
            (0.82, 5.0, "stale"),  # Tier 2 flagged
            (0.65, 5.0, "stale"),  # Tier 3
        ]:
            p = _make_pick(logical_score=L, edge_pct=edge, flagged=flagged)
            out = format_pick_alert(p)
            assert not emoji_re.search(out), \
                f"emoji leaked into tier output for L={L}, flagged={flagged}: {out!r}"


# ── V2: Outcome reply (Tier 4) ──────────────────────────────────────────────


class TestFormatOutcomeReply:
    @pytest.mark.parametrize("status,expected_verdict", [
        ("won", "WON"),
        ("lost", "LOST"),
        ("void", "VOID"),
    ])
    def test_verdict_word(self, status, expected_verdict):
        out = format_outcome_reply(
            status=status, market="ou_2_5", selection="over",
            bookmaker_odd=1.85, profit_units=0.85,
        )
        assert expected_verdict in out

    def test_includes_emit_pl(self):
        out = format_outcome_reply(
            status="won", market="btts", selection="yes",
            bookmaker_odd=2.10, profit_units=1.10,
        )
        assert "+1.10u" in out

    def test_includes_loss_with_sign(self):
        out = format_outcome_reply(
            status="lost", market="btts", selection="yes",
            bookmaker_odd=2.10, profit_units=-1.00,
        )
        assert "-1.00u" in out

    def test_no_placement_block_when_not_placed(self):
        out = format_outcome_reply(
            status="won", market="ou_2_5", selection="over",
            bookmaker_odd=1.85, profit_units=0.85,
        )
        assert "Placed" not in out

    def test_placement_block_when_placed(self):
        out = format_outcome_reply(
            status="won", market="ou_2_5", selection="over",
            bookmaker_odd=1.85, profit_units=0.85,
            placed_stake_pct=1.5, placed_odd=1.90,
            actual_profit_units=1.35,
        )
        assert "Placed" in out
        assert "1.50%" in out
        assert "1.90" in out
        assert "+1.35u" in out


# ── V2: Burst digest (§E) ───────────────────────────────────────────────────


class TestBurstDigest:
    def test_empty_returns_empty_string(self):
        assert format_burst_digest([], window_seconds=60) == ""

    def test_header_shows_count_and_window(self):
        picks = [_make_pick() for _ in range(3)]
        out = format_burst_digest(picks, window_seconds=60)
        assert "BURST" in out
        assert "3 picks" in out
        assert "60s" in out

    def test_one_line_per_pick(self):
        picks = [
            _make_pick(home="A", away="B", market="m1"),
            _make_pick(home="C", away="D", market="m2"),
            _make_pick(home="E", away="F", market="m3"),
        ]
        out = format_burst_digest(picks, window_seconds=30)
        for team in ("A", "B", "C", "D", "E", "F"):
            assert team in out
        for market in ("m1", "m2", "m3"):
            assert market in out


# ── V2: Scoreboard ──────────────────────────────────────────────────────────


class TestScoreboard:
    def test_basic_fields(self):
        out = format_scoreboard(
            date="2026-05-10",
            n_emit=12, n_placed=8, n_pending=2,
            n_settled=10, n_won=6,
            pl_emit_units=2.45,
            updated_at="2026-05-10T17:30:00Z",
        )
        assert "2026-05-10" in out
        assert "12" in out  # emit
        assert "8" in out   # placed
        # Win rate 6/10 = 60.0%
        assert "60.0" in out
        assert "+2.45u" in out

    def test_placed_block_when_provided(self):
        out = format_scoreboard(
            date="2026-05-10",
            n_emit=12, n_placed=8, n_pending=2,
            n_settled=10, n_won=6,
            pl_emit_units=2.45, pl_placed_units=1.80,
            updated_at="now",
        )
        assert "placed only" in out
        assert "+1.80u" in out

    def test_drawdown_block_only_when_nonzero(self):
        out_no_dd = format_scoreboard(
            date="2026-05-10", n_emit=1, n_placed=0, n_pending=0,
            n_settled=1, n_won=1, pl_emit_units=1.0,
            drawdown_pct=0.0, updated_at="now",
        )
        assert "Drawdown" not in out_no_dd

        out_dd = format_scoreboard(
            date="2026-05-10", n_emit=1, n_placed=0, n_pending=0,
            n_settled=1, n_won=0, pl_emit_units=-1.0,
            drawdown_pct=3.5, updated_at="now",
        )
        assert "Drawdown" in out_dd
        assert "3.5" in out_dd

    def test_handles_zero_settled(self):
        # Pre-jornada state: nothing settled yet — must not /0
        out = format_scoreboard(
            date="2026-05-10", n_emit=5, n_placed=3, n_pending=5,
            n_settled=0, n_won=0, pl_emit_units=0.0,
            updated_at="now",
        )
        assert "0.0%" in out
