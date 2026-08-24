from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any

from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability


def load_selected_backend(
    name: str,
    import_name: str,
    capabilities: Iterable[BackendCapability],
    *,
    registry: BackendRegistry = BACKEND_REGISTRY,
) -> ModuleType:
    """Select every required capability before importing an optional module."""

    for capability in capabilities:
        registry.select(
            capability,
            name=name,
            device="cpu",
            differentiable=False,
        )

    try:
        return importlib.import_module(import_name)
    except ImportError as error:
        raise RuntimeError(
            f"Backend {name!r} passed registry discovery but importing {import_name!r} failed: {error}"
        ) from error


def load_explicit_backend(
    name: str,
    import_name: str,
    capabilities: Iterable[BackendCapability],
    *,
    device: Any,
    dtype: Any,
    differentiable: bool,
    registry: BackendRegistry = BACKEND_REGISTRY,
) -> ModuleType:
    """Select an accelerated backend for an explicit runtime before importing it."""

    for capability in capabilities:
        registry.select(
            capability,
            name=name,
            device=device,
            dtype=dtype,
            differentiable=differentiable,
        )

    try:
        return importlib.import_module(import_name)
    except ImportError as error:
        raise RuntimeError(
            f"Backend {name!r} passed registry discovery but importing {import_name!r} failed: {error}"
        ) from error


__all__ = ["load_explicit_backend", "load_selected_backend"]
