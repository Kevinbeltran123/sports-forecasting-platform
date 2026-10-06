"""ModelRegistry — central pluggable registry for v4 prediction models.

Producers call `register(model)` to make a BaseModel discoverable. The
orchestrator (Sprint 1+) iterates `iter_applicable(fixture)` to find
every model that can handle a given fixture and calls `.predict()` on
each, collecting the union of PredictionRecord rows.

The registry is dict-keyed by model.name to allow duplicate registration
to overwrite (useful in tests and hot-reload). Order of iteration is
insertion order (Python 3.7+ dict guarantee), which gives deterministic
behavior in tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import structlog

from bip.models.base import BaseModel

log = structlog.get_logger(__name__)


class ModelRegistry:
    """In-memory registry of BaseModel instances keyed by name.

    Typical usage:
        registry = ModelRegistry()
        registry.register(MundialModel(lock_path=...))
        # Sprint 1: registry.register(LigasModel(...))

        for model in registry.iter_applicable(fixture):
            preds = model.predict(fixture)
            ...
    """

    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}

    def register(self, model: BaseModel) -> None:
        """Add (or replace) a model in the registry, keyed by model.name."""
        if not isinstance(model, BaseModel):
            raise TypeError(f"Expected BaseModel instance, got {type(model).__name__}")
        if model.name in self._models:
            log.info("model_registry_replaced", name=model.name, version=model.version)
        else:
            log.info("model_registry_registered", name=model.name, version=model.version)
        self._models[model.name] = model

    def get(self, name: str) -> BaseModel:
        """Retrieve a registered model by name; KeyError if not present."""
        return self._models[name]

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)

    def names(self) -> list[str]:
        """Return registered model names in insertion order."""
        return list(self._models.keys())

    def iter_applicable(self, fixture: dict[str, Any]) -> Iterator[BaseModel]:
        """Yield every registered model whose `can_handle(fixture)` returns True.

        Errors raised inside `can_handle` are caught + logged so a single
        misbehaving model does not block the rest. Producers should keep
        `can_handle` pure and side-effect-free.
        """
        for name, model in self._models.items():
            try:
                if model.can_handle(fixture):
                    yield model
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "model_can_handle_error",
                    name=name,
                    error=str(exc),
                    fixture_id=fixture.get("fixture_id"),
                )


__all__ = ["ModelRegistry"]
