from __future__ import annotations

import importlib
from types import ModuleType

_INSTALL_HINTS = {
    "portable": 'pip install "diffusers-3d[portable]"',
    "meshlab": 'pip install "diffusers-3d[meshlab]" (pymeshlab, GPL)',
    "gaussian": 'pip install "diffusers-3d[gaussian]"',
    "cumesh": "pip install -r requirements/backends/cumesh.txt (see docs/backends.md)",
}


def require(module: str, install: str) -> ModuleType:
    """Import an optional dependency, or raise an ``ImportError`` that says how to install it."""

    try:
        return importlib.import_module(module)
    except ImportError as error:
        hint = _INSTALL_HINTS.get(install, install)
        raise ImportError(f"{module!r} is not installed. Install it with: {hint}") from error


__all__ = ["require"]
