from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections.abc import Sequence
from typing import Any

from ._version import __version__
from .backends import BACKEND_REGISTRY, BackendRegistry


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def compatibility_report(*, registry: BackendRegistry = BACKEND_REGISTRY) -> dict[str, Any]:
    """Return core versions and side-effect-free optional backend discovery."""

    if not isinstance(registry, BackendRegistry):
        raise TypeError("registry must be a BackendRegistry")
    statuses = registry.report().statuses
    return {
        "backends": [
            {
                "available": status.available,
                "devices": sorted(status.spec.devices),
                "distribution": status.distribution_name,
                "dtypes": sorted(status.spec.dtypes),
                "importable": status.importable,
                "installed": status.installed,
                "license_class": status.spec.license_class.value,
                "name": status.name,
                "provenance_verified": status.provenance_verified,
                "reason": status.reason,
                "support_level": status.spec.support_level.value,
                "version": status.version,
            }
            for status in statuses
        ],
        "package": {
            "accelerate": _distribution_version("accelerate"),
            "diffusers": _distribution_version("diffusers"),
            "diffusers_3d": __version__,
            "python": platform.python_version(),
            "torch": _distribution_version("torch"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diffusers-3d-report",
        description="Report core versions and discover optional backends without importing them.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact rather than indented JSON.")
    arguments = parser.parse_args(argv)
    print(json.dumps(compatibility_report(), indent=None if arguments.compact else 2, sort_keys=True))
    return 0


__all__ = ["compatibility_report", "main"]
