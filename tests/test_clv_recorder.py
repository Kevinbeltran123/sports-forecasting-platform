"""CLV recorder tests — CLV-02 (calculation), CLV-03 (rolling average)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock


class TestClvCalculation:
    """CLV-02: clv_percentage = (odds_at_pick / closing_odds - 1) * 100."""

    def test_clv_percentage_formula(self):
        """CLV formula collapses to (2.10 / 2.00 - 1) * 100 = +5.0 on a vig-free dict."""
        from bip.clv.recorder import calculate_clv_percentage
        # {"home": 2.00, "away": 2.00} is vig-free (1/2 + 1/2 = 1.0), so fair == raw.
        result = calculate_clv_percentage(
            odds_at_pick=2.10,
            closing_odds_dict={"home": 2.00, "away": 2.00},
            selection="home",
        )
        assert result == pytest.approx(5.0, abs=0.01)

    def test_clv_negative_when_odds_worse_than_closing(self):
        """CLV is negative when we got worse odds than closing line."""
        from bip.clv.recorder import calculate_clv_percentage
        result = calculate_clv_percentage(
            odds_at_pick=1.90,
            closing_odds_dict={"home": 2.00, "away": 2.00},
            selection="home",
        )
        assert result == pytest.approx(-5.0, abs=0.01)

    def test_clv_zero_when_odds_equal_closing(self):
        """CLV is 0.0 when odds at pick == fair closing odds."""
        from bip.clv.recorder import calculate_clv_percentage
        result = calculate_clv_percentage(
            odds_at_pick=2.00,
            closing_odds_dict={"home": 2.00, "away": 2.00},
            selection="home",
        )
        assert result == pytest.approx(0.0, abs=0.001)

    def test_clv_record_stored_with_odds_fetched_at(self, mock_client):
        """ClvRecord.odds_fetched_at is set when recording CLV -- D-04c.

        Also locks the persistence invariant: pinnacle_closing_odds = RAW odd.
        """
        from bip.clv.recorder import ClvRecorder
        from bip.core.storage.models import ClvRecord
        from tests.conftest import setup_mock_chain

        setup_mock_chain(mock_client, data=[{"id": 1}])
        recorder = ClvRecorder(client=mock_client)
        clv = recorder.record(
            pick_id=1,
            fixture_id=12345,
            sport="football",
            market="onextwo",
            odds_at_pick=2.10,
            closing_odds_dict={"home": 2.00, "away": 2.00},
            selection="home",
            odds_fetched_at=datetime(2026, 4, 22, 17, 45, 0, tzinfo=timezone.utc),
        )
        assert clv.clv_percentage == pytest.approx(5.0, abs=0.01)
        assert clv.odds_fetched_at is not None
        # Persistence-decision invariant: raw closing odd is stored, not fair.
        assert clv.pinnacle_closing_odds == pytest.approx(2.00, abs=1e-6)


class TestRollingAverage:
    """CLV-03: Rolling 50-pick average CLV drops below +1% triggers alert flag."""

    def test_rolling_average_above_threshold(self):
        """No alert when rolling 50-pick average CLV >= +1%."""
        from bip.clv.recorder import compute_rolling_clv_average
        clv_values = [3.0] * 50  # all +3% — healthy
        avg = compute_rolling_clv_average(clv_values)
        assert avg == pytest.approx(3.0)
        assert avg >= 1.0

    def test_rolling_average_below_threshold(self):
        """Rolling average below +1% must return value < 1.0 (caller decides alert)."""
        from bip.clv.recorder import compute_rolling_clv_average
        clv_values = [-2.0] * 30 + [1.0] * 20  # average negative
        avg = compute_rolling_clv_average(clv_values)
        assert avg < 1.0

    def test_rolling_average_uses_last_50(self):
        """compute_rolling_clv_average uses only the last 50 values."""
        from bip.clv.recorder import compute_rolling_clv_average
        # 100 values: first 50 bad, last 50 good
        clv_values = [-5.0] * 50 + [4.0] * 50
        avg = compute_rolling_clv_average(clv_values)
        assert avg == pytest.approx(4.0, abs=0.01)

    def test_fewer_than_50_picks_uses_all(self):
        """With fewer than 50 picks, average all available values."""
        from bip.clv.recorder import compute_rolling_clv_average
        clv_values = [2.0, 3.0, 4.0]
        avg = compute_rolling_clv_average(clv_values)
        assert avg == pytest.approx(3.0, abs=0.01)


class TestRemoveVig:
    """Proportional vig removal -- PITFALLS.md Pitfall 7."""

    def test_remove_vig_proportional(self):
        """3-way market: vig-removed implicit probabilities sum to 1.0; fair odds > raw odds."""
        from bip.clv.odds_math import remove_vig
        raw = {"1": 1.95, "X": 3.40, "2": 4.20}
        fair = remove_vig(raw)
        prob_sum = sum(1.0 / fair[k] for k in raw)
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
        assert fair["1"] > 1.95
        assert fair["X"] > 3.40
        assert fair["2"] > 4.20

    def test_remove_vig_two_way(self):
        """Symmetric 2-way market at 1.91/1.91 -> fair 2.00/2.00 (Pinnacle ~2.5% vig)."""
        from bip.clv.odds_math import remove_vig
        fair = remove_vig({"over": 1.91, "under": 1.91})
        assert fair["over"] == pytest.approx(2.0, abs=1e-6)
        assert fair["under"] == pytest.approx(2.0, abs=1e-6)

    def test_remove_vig_rejects_empty_dict(self):
        """Empty dict -> ValueError mentioning 'empty'."""
        from bip.clv.odds_math import remove_vig
        with pytest.raises(ValueError, match="empty"):
            remove_vig({})

    def test_remove_vig_rejects_invalid_odds(self):
        """Any odd <= 1.0 -> ValueError mentioning the offending key/value."""
        from bip.clv.odds_math import remove_vig
        with pytest.raises(ValueError, match="1"):
            remove_vig({"1": 1.0, "X": 3.0, "2": 4.0})
        with pytest.raises(ValueError, match="0.5|1"):
            remove_vig({"1": 0.5, "X": 3.0})

    def test_remove_vig_preserves_keys(self):
        """Output dict has the same key set as the input."""
        from bip.clv.odds_math import remove_vig
        fair = remove_vig({"a": 2.0, "b": 2.1, "c": 2.2})
        assert set(fair.keys()) == {"a", "b", "c"}


class TestClvVigRemoval:
    """calculate_clv_percentage + ClvRecorder.record() under the dict+selection signature."""

    def test_calculate_clv_uses_vig_removed_price(self):
        """Vig-removed CLV is materially lower than raw CLV (~+5.5% vs ~+7.7%)."""
        from bip.clv.recorder import calculate_clv_percentage
        result = calculate_clv_percentage(
            odds_at_pick=2.10,
            closing_odds_dict={"1": 1.95, "X": 3.40, "2": 4.20},
            selection="1",
        )
        # Raw CLV would be (2.10/1.95 - 1) * 100 ~= 7.69%.
        # Vig-removed fair odd for "1" is ~2.038, so CLV ~= +3.05%.
        # Spec wants ~+5.5% with abs=0.5 OR strictly < 7.0 (proves vig was removed).
        assert result < 7.0
        assert result == pytest.approx(3.05, abs=0.5)

    def test_calculate_clv_raises_on_missing_selection(self):
        """Selection not present in the dict -> ClvError."""
        from bip.clv.recorder import calculate_clv_percentage
        from bip.core.errors import ClvError
        with pytest.raises(ClvError, match="Z|selection"):
            calculate_clv_percentage(
                odds_at_pick=2.00,
                closing_odds_dict={"1": 1.95, "X": 3.40, "2": 4.20},
                selection="Z",
            )

    def test_calculate_clv_raises_on_invalid_dict(self):
        """Empty dict or any odd <= 1.0 -> ClvError (wrapping ValueError)."""
        from bip.clv.recorder import calculate_clv_percentage
        from bip.core.errors import ClvError
        with pytest.raises(ClvError):
            calculate_clv_percentage(
                odds_at_pick=2.00,
                closing_odds_dict={},
                selection="home",
            )
        with pytest.raises(ClvError):
            calculate_clv_percentage(
                odds_at_pick=2.00,
                closing_odds_dict={"home": 1.0, "away": 2.0},
                selection="home",
            )

    def test_record_persists_raw_closing_odd_for_selection(self, mock_client):
        """Persistence invariant: raw odd in pinnacle_closing_odds, vig-removed CLV."""
        from bip.clv.recorder import ClvRecorder, calculate_clv_percentage
        from tests.conftest import setup_mock_chain

        setup_mock_chain(mock_client, data=[{"id": 1}])
        recorder = ClvRecorder(client=mock_client)
        closing = {"1": 1.95, "X": 3.40, "2": 4.20}
        clv = recorder.record(
            pick_id=1,
            fixture_id=12345,
            sport="football",
            market="onextwo",
            odds_at_pick=2.10,
            closing_odds_dict=closing,
            selection="1",
        )
        # Raw odd for the selection is persisted (audit trail).
        assert clv.pinnacle_closing_odds == pytest.approx(1.95, abs=1e-6)
        # implied_prob_closing == 1 / raw odd (audit consistency).
        assert clv.implied_prob_closing == pytest.approx(1.0 / 1.95, abs=1e-6)
        # clv_percentage matches the vig-removed math from calculate_clv_percentage.
        expected = calculate_clv_percentage(
            odds_at_pick=2.10, closing_odds_dict=closing, selection="1"
        )
        assert clv.clv_percentage == pytest.approx(expected, abs=1e-6)
        assert clv.clv_percentage < 7.0  # proves vig was removed
