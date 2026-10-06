"""Tests for Phase 3 core type extensions: PickStatus, Pick model, domain errors.

Plan 03-02 Task 1 (TDD RED phase):
  - PickStatus gains filtered + rejected (D-03)
  - Pick gains 4 claude_* fields with ISO datetime serialization (D-08)
  - 3 new domain errors in core/errors.py (PATTERNS.md discretion #3)
"""

from datetime import UTC, datetime

import pytest


class TestPickStatusExtensions:
    """Test 1 + Test 2: PickStatus has 7 members with filtered and rejected."""

    def test_filtered_value(self):
        from bip.core.types import PickStatus
        assert PickStatus.filtered.value == "filtered"

    def test_rejected_value(self):
        from bip.core.types import PickStatus
        assert PickStatus.rejected.value == "rejected"

    def test_enum_has_seven_members(self):
        from bip.core.types import PickStatus
        assert len(list(PickStatus)) == 7


class TestPickModelClaudeFields:
    """Test 3 + Test 4: Pick has 4 claude_* fields; to_supabase_dict serializes correctly."""

    def _base_pick_kwargs(self, **overrides):
        defaults = dict(
            fixture_id=1,
            league="premier_league",
            market="1X2",
            selection="1",
            model_probability=0.5,
            implied_probability=0.45,
            edge=0.05,
            best_odds=2.0,
            bookmaker="betano",
        )
        defaults.update(overrides)
        return defaults

    def test_claude_fields_default_to_none_backcompat(self):
        """Phase 1+2 picks construct without claude_* args — all default to None."""
        from bip.core.storage.models import Pick
        p = Pick(**self._base_pick_kwargs())
        assert p.claude_validation is None
        assert p.claude_reasoning is None
        assert p.claude_summary is None
        assert p.claude_validated_at is None

    def test_to_supabase_dict_claude_validated_at_none_when_not_set(self):
        """to_supabase_dict emits None for claude_validated_at when not provided."""
        from bip.core.storage.models import Pick
        p = Pick(**self._base_pick_kwargs())
        d = p.to_supabase_dict()
        assert d.get("claude_validated_at") is None

    def test_to_supabase_dict_claude_validated_at_iso_string(self):
        """When claude_validated_at is set, to_supabase_dict emits an ISO string."""
        from bip.core.storage.models import Pick
        now = datetime.now(UTC)
        p = Pick(
            **self._base_pick_kwargs(
                fixture_id=2,
                league="la_liga",
                selection="X",
                model_probability=0.3,
                implied_probability=0.3,
                edge=0.0,
                best_odds=3.5,
                claude_validation="CONFIRM",
                claude_validated_at=now,
            )
        )
        d = p.to_supabase_dict()
        assert d["claude_validation"] == "CONFIRM"
        assert isinstance(d["claude_validated_at"], str)
        assert "T" in d["claude_validated_at"]  # ISO format marker

    def test_to_supabase_dict_with_all_claude_fields(self):
        """All 4 claude_* fields roundtrip through to_supabase_dict."""
        from bip.core.storage.models import Pick
        now = datetime.now(UTC)
        p = Pick(
            **self._base_pick_kwargs(
                claude_validation="FLAG",
                claude_reasoning="Injury concern for key player",
                claude_summary="Mild concern — play with reduced stake",
                claude_validated_at=now,
            )
        )
        d = p.to_supabase_dict()
        assert d["claude_validation"] == "FLAG"
        assert d["claude_reasoning"] == "Injury concern for key player"
        assert d["claude_summary"] == "Mild concern — play with reduced stake"
        assert isinstance(d["claude_validated_at"], str)


class TestDomainErrors:
    """Test 5: PickError, ClaudeError, TelegramError are exported and are Exception subclasses."""

    def test_import_succeeds(self):
        from bip.core.errors import ClaudeError, PickError, TelegramError
        assert PickError
        assert ClaudeError
        assert TelegramError

    def test_pick_error_is_exception(self):
        from bip.core.errors import PickError
        assert issubclass(PickError, Exception)

    def test_claude_error_is_exception(self):
        from bip.core.errors import ClaudeError
        assert issubclass(ClaudeError, Exception)

    def test_telegram_error_is_exception(self):
        from bip.core.errors import TelegramError
        assert issubclass(TelegramError, Exception)

    def test_errors_are_raiseable(self):
        from bip.core.errors import ClaudeError, PickError, TelegramError
        with pytest.raises(PickError):
            raise PickError("pick failed")
        with pytest.raises(ClaudeError):
            raise ClaudeError("malformed tool_use response")
        with pytest.raises(TelegramError):
            raise TelegramError("bot send failed")
