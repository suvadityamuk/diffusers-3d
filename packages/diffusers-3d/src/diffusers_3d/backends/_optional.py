from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any

import torch
from packaging.version import InvalidVersion, Version

from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability


def diagnostic_build_identity(module: ModuleType) -> str | None:
    """Return optional upstream version/build text without treating it as trust evidence."""

    for name in ("__build_id__", "__version__", "version"):
        value = getattr(module, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def validate_accelerated_runtime(
    backend_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    require_triton: bool = False,
) -> None:
    """Validate the active Torch/device/toolchain surface after registry selection."""

    try:
        torch_version = Version(torch.__version__.split("+", maxsplit=1)[0])
    except InvalidVersion as error:
        raise RuntimeError(f"{backend_name} could not validate torch version {torch.__version__!r}") from error
    if torch_version < Version("2.4"):
        raise RuntimeError(f"{backend_name} requires torch>=2.4, found torch=={torch.__version__}")
    try:
        floating = torch.empty((), dtype=dtype).is_floating_point()
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(f"{backend_name} received an unsupported torch dtype {dtype}") from error
    if not floating:
        raise RuntimeError(f"{backend_name} requires a floating-point torch dtype")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{backend_name} requires an available CUDA/ROCm torch runtime")
        if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
            raise RuntimeError(f"{backend_name} device index {device.index} is unavailable")
        if torch.version.cuda is None and torch.version.hip is None:
            raise RuntimeError(f"{backend_name} requires a CUDA- or ROCm-enabled torch build")
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{backend_name} requires bfloat16 support on the selected CUDA/ROCm device")
        if require_triton:
            try:
                triton = importlib.import_module("triton")
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"{backend_name} requires Triton compatible with the active torch runtime"
                ) from error
            if not callable(getattr(triton, "jit", None)):
                raise RuntimeError(f"{backend_name} requires the public triton.jit runtime API")


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


__all__ = [
    "diagnostic_build_identity",
    "load_explicit_backend",
    "load_selected_backend",
    "validate_accelerated_runtime",
]
