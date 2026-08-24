from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath

FAMILIES = {
    "hunyuan3d": "LICENSE-TENCENT-HUNYUAN-3D-2.1",
    "trellis": "LICENSE-MIT",
    "trellis2": "LICENSE-MIT",
}


def verify_wheel(path: Path) -> None:
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"expected one wheel file, got {path}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

    for family, license_name in FAMILIES.items():
        family_root = f"diffusers_3d/families/{family}"
        for filename in ("README.md", "NOTICE", license_name, "diffusers_3d_integration.json"):
            required = f"{family_root}/{filename}"
            if required not in names:
                raise RuntimeError(f"wheel is missing required family artifact: {required}")

    manifests = {name for name in names if name.endswith("/diffusers_3d_integration.json")}
    expected_manifests = {
        f"diffusers_3d/families/{family}/diffusers_3d_integration.json" for family in FAMILIES
    }
    if manifests != expected_manifests:
        raise RuntimeError(
            f"wheel manifest set does not match shipped families: expected {sorted(expected_manifests)}, "
            f"got {sorted(manifests)}"
        )

    forbidden_parts = {"__pycache__", ".pytest_cache", "build", "tests", "tools"}
    forbidden_suffixes = (".pyc", ".pyo", ".tmp")
    forbidden_names = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            forbidden_parts.intersection(parts)
            or name.endswith(forbidden_suffixes)
            or any(part.endswith(".egg-info") for part in parts)
            or name.endswith(".dist-info/SOURCES.txt")
            or name in {"pyproject.toml", "setup.cfg", "setup.py"}
        ):
            forbidden_names.append(name)
    if forbidden_names:
        raise RuntimeError(f"wheel contains source-only or temporary artifacts: {sorted(forbidden_names)}")

    top_levels = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if "diffusers_3d" not in top_levels or not any(name.endswith(".dist-info") for name in top_levels):
        raise RuntimeError(f"wheel has unexpected top-level layout: {sorted(top_levels)}")
    unexpected = {
        name for name in top_levels if name != "diffusers_3d" and not name.endswith(".dist-info")
    }
    if unexpected:
        raise RuntimeError(f"wheel contains unexpected top-level paths: {sorted(unexpected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify diffusers-3d wheel contents.")
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    verify_wheel(arguments.wheel)
    print(f"verified wheel contents: {arguments.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
