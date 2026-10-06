"""GREEN tests for src/bip/core/claude/learnings_loader.py."""
from __future__ import annotations

from pathlib import Path

import pytest


class TestLearningsLoader:
    def test_sha_stamp_and_cache_block(self):
        from bip.core.claude.learnings_loader import load_learnings, _LEARNINGS_PATH

        assert _LEARNINGS_PATH.exists(), f"port football-learnings.md to {_LEARNINGS_PATH}"
        text, sha = load_learnings(force_reload=True)
        assert isinstance(text, str) and len(text) > 0
        assert isinstance(sha, str)
        # SHA is either 40-char git hex OR 16-char sha256 fallback
        assert len(sha) in (40, 16), f"unexpected sha length {len(sha)}: {sha!r}"
        assert all(c in "0123456789abcdef" for c in sha), f"sha must be hex: {sha!r}"

    def test_module_cache_avoids_disk_reread(self, monkeypatch):
        # Reset cache, ensure first call reads disk, second call hits cache
        import bip.core.claude.learnings_loader as ll
        monkeypatch.setattr(ll, "_CACHE", None)

        read_count = {"n": 0}
        original_read = Path.read_text

        def counting_read(self, *a, **kw):
            read_count["n"] += 1
            return original_read(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", counting_read)
        ll.load_learnings()  # first call — reads disk
        ll.load_learnings()  # second call — cache hit
        assert read_count["n"] == 1, f"expected 1 disk read, got {read_count['n']}"

    def test_falls_back_to_sha256_when_uncommitted(self, monkeypatch):
        import bip.core.claude.learnings_loader as ll
        monkeypatch.setattr(ll, "_CACHE", None)
        # Force git_sha to return None
        monkeypatch.setattr(ll, "_git_sha", lambda path: None)
        text, sha = ll.load_learnings(force_reload=True)
        assert len(sha) == 16, f"expected 16-char sha256 fallback, got {len(sha)}: {sha!r}"

    def test_size_above_2048_tokens(self):
        """Pitfall 4: file must be ≥2048 tokens for Sonnet 4.6 prompt cache."""
        from bip.core.claude.learnings_loader import load_learnings
        text, _ = load_learnings(force_reload=True)
        approx_tokens = len(text) // 4  # rough estimate
        assert approx_tokens >= 2048, (
            f"learnings.md is {approx_tokens} approx tokens — below Sonnet 4.6 cache "
            "minimum (2048). Cache will silently miss, costing 1.25x input price per call."
        )

    def test_missing_file_raises_file_not_found(self, monkeypatch, tmp_path):
        import bip.core.claude.learnings_loader as ll
        monkeypatch.setattr(ll, "_CACHE", None)
        monkeypatch.setattr(ll, "_LEARNINGS_PATH", tmp_path / "nonexistent.md")
        with pytest.raises(FileNotFoundError):
            ll.load_learnings(force_reload=True)
