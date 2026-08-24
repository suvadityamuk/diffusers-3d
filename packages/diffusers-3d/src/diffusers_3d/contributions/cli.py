from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .validation import validate_manifest_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diffusers-3d-validate",
        description="Validate a local diffusers-3d integration manifest without network access.",
    )
    parser.add_argument("manifest", help="Path to a versioned integration manifest JSON file.")
    arguments = parser.parse_args(argv)

    report = validate_manifest_file(arguments.manifest)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.is_valid else 1


__all__ = ["main"]
