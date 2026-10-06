"""Loader for football-learnings.md with git SHA stamping (Risks 5).

D-05: learnings is loaded verbatim into the Role C system prompt. The git SHA is
stamped into both the prompt (forms part of the cache breakpoint) and persisted
to claude_reasoning so audit traces pin which version of the file was used.

PATTERNS.md §13: package-data path via Path(__file__).parent / "prompts" / ...
PATTERNS.md drift risk #5: do NOT chunk the file — caching requires ONE block ≥2048 tokens.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_LEARNINGS_PATH: Path = Path(__file__).parent / "prompts" / "football-learnings.md"
_CACHE: tuple[str, str] | None = None


def _git_sha(path: Path) -> str | None:
    """Return git commit SHA of `path` (40-char hex). None on failure / uncommitted."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=path.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if len(sha) == 40 else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _sha256_short(text: str) -> str:
    """Fallback when git is unavailable or file is uncommitted (Risks 5)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_learnings(force_reload: bool = False) -> tuple[str, str]:
    """Return (text, sha). Memoized at module level after first call.

    Args:
        force_reload: bypass cache (useful in tests).

    Returns:
        text: full content of football-learnings.md (UTF-8).
        sha:  40-char git hex if committed, else 16-char sha256 hex fallback.

    Raises:
        FileNotFoundError: if learnings.md is missing.
    """
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    if not _LEARNINGS_PATH.exists():
        raise FileNotFoundError(
            f"learnings file not found at {_LEARNINGS_PATH} — port via plan 03-04 Task 1"
        )

    text = _LEARNINGS_PATH.read_text(encoding="utf-8")
    sha = _git_sha(_LEARNINGS_PATH) or _sha256_short(text)

    logger.info(
        "learnings_loaded",
        path=str(_LEARNINGS_PATH),
        sha=sha,
        sha_source=("git" if len(sha) == 40 else "sha256_fallback"),
        char_count=len(text),
        approx_token_count=len(text) // 4,
    )

    _CACHE = (text, sha)
    return _CACHE
