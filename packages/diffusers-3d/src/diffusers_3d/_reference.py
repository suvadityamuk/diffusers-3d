from __future__ import annotations

import importlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit


class ReferenceCheckoutError(RuntimeError):
    """Raised when a parity checkout is not the exact trusted source tree."""


def import_reference_dependency(module_name: str) -> ModuleType:
    """Import a parity-only dependency through the required-reference policy."""

    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError, RuntimeError) as error:
        raise ReferenceCheckoutError(f"reference dependency {module_name!r} is unavailable: {error}") from error


def reference_unavailable(error: ReferenceCheckoutError) -> str:
    """Return an optional-skip reason or fail required-reference execution."""

    if os.environ.get("DIFFUSERS_3D_REQUIRE_REFERENCE") == "1":
        raise error
    return str(error)


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ReferenceCheckoutError("git is required to validate reference parity checkouts") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise ReferenceCheckoutError(f"could not validate reference checkout {root}: {detail}")
    return result.stdout.strip()


def _normalized_repository_url(url: str) -> str:
    normalized = url.strip().lower().rstrip("/")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    elif normalized.startswith("ssh://git@github.com/"):
        normalized = "https://github.com/" + normalized.removeprefix("ssh://git@github.com/")
    parsed = urlsplit(normalized)
    if parsed.hostname:
        port = f":{parsed.port}" if parsed.port is not None else ""
        normalized = f"{parsed.hostname}{port}/{parsed.path.lstrip('/')}"
    return normalized.removesuffix(".git")


@lru_cache(maxsize=None)
def validate_reference_checkout(
    root: Path,
    *,
    expected_revision: str,
    expected_repository: str,
    expected_paths: tuple[str, ...],
) -> None:
    """Validate identity, cleanliness, origin, and tracked source files before import."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ReferenceCheckoutError(f"reference checkout is unavailable: {root}")

    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ReferenceCheckoutError(f"reference root {root} is not the checkout top level {top_level}")

    revision = _git(root, "rev-parse", "HEAD").lower()
    if revision != expected_revision:
        raise ReferenceCheckoutError(f"reference checkout {root} is at {revision!r}; expected {expected_revision!r}")

    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ReferenceCheckoutError(f"reference checkout {root} is not clean:\n{status}")

    repository = _git(root, "remote", "get-url", "origin")
    repository_identity = _normalized_repository_url(repository)
    expected_identity = _normalized_repository_url(expected_repository)
    if repository_identity != expected_identity:
        raise ReferenceCheckoutError(
            f"reference checkout origin {repository_identity!r} does not match {expected_identity!r}"
        )

    for relative_path in expected_paths:
        source_path = root / relative_path
        if not source_path.is_file():
            raise ReferenceCheckoutError(f"reference checkout is missing required source file: {relative_path}")
        _git(root, "cat-file", "-e", f"{expected_revision}:{relative_path}")


__all__ = [
    "ReferenceCheckoutError",
    "import_reference_dependency",
    "reference_unavailable",
    "validate_reference_checkout",
]
