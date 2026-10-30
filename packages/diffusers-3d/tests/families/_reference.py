"""Locate a pinned upstream checkout for the reference-parity tests."""

from __future__ import annotations

import importlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType


class ReferenceCheckoutError(RuntimeError):
    """The upstream checkout a parity test needs is missing or at the wrong revision."""


def import_reference_dependency(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError, RuntimeError) as error:
        raise ReferenceCheckoutError(f"reference dependency {module_name!r} is unavailable: {error}") from error


def reference_unavailable(error: ReferenceCheckoutError) -> str:
    """Return a skip reason, or re-raise when ``DIFFUSERS_3D_REQUIRE_REFERENCE=1``."""

    if os.environ.get("DIFFUSERS_3D_REQUIRE_REFERENCE") == "1":
        raise error
    return str(error)


@lru_cache(maxsize=None)
def validate_reference_checkout(
    root: Path, *, expected_revision: str, expected_repository: str, expected_paths: tuple[str, ...]
) -> None:
    """Check that ``root`` is a checkout of ``expected_repository`` at ``expected_revision`` with the given files."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ReferenceCheckoutError(f"reference checkout is unavailable: {root} (clone {expected_repository})")
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    revision = result.stdout.strip().lower()
    if result.returncode or revision != expected_revision:
        raise ReferenceCheckoutError(f"reference checkout {root} is at {revision!r}; expected {expected_revision!r}")
    missing = [path for path in expected_paths if not (root / path).is_file()]
    if missing:
        raise ReferenceCheckoutError(f"reference checkout {root} is missing {missing}")
