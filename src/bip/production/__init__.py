"""bip.production - production deployment package (D-05).

Public API:
- build_orchestrator(settings): factory returning a fully-wired PipelineOrchestrator
  + the two TelegramBots + the systemd notifier.
- DriftChecker: weekly drift wrapper (D-14) consumed by the orchestrator's _check_drift.
"""
from bip.production.builder import DriftChecker, build_orchestrator

__all__ = ["build_orchestrator", "DriftChecker"]
