from __future__ import annotations

import argparse
import configparser
import email
import zipfile
from pathlib import Path, PurePosixPath

FAMILIES = {
    "trellis": "LICENSE-MIT",
    "trellis2": "LICENSE-MIT",
}
LICENSE_EXPRESSION = "Apache-2.0 AND MIT"
LICENSE_FILES = {
    "LICENSE-APACHE-2.0",
    "src/diffusers_3d/families/trellis/LICENSE-MIT",
    "src/diffusers_3d/families/trellis/NOTICE",
    "src/diffusers_3d/families/trellis2/LICENSE-MIT",
    "src/diffusers_3d/families/trellis2/NOTICE",
}
ENTRY_POINTS = {
    "diffusers-3d-check-release": "diffusers_3d.contributions.removal:release_check_main",
    "diffusers-3d-convert-trellis": "diffusers_3d.families.trellis.conversion:main",
    "diffusers-3d-convert-trellis2": "diffusers_3d.families.trellis2.conversion:main",
    "diffusers-3d-report": "diffusers_3d.compatibility:main",
    "diffusers-3d-validate": "diffusers_3d.contributions.cli:main",
}


def _distribution_metadata(contents: bytes, *, source: str):
    metadata = email.message_from_bytes(contents)
    if metadata.get("Name") != "diffusers-3d":
        raise RuntimeError(f"{source} has unexpected project name: {metadata.get('Name')!r}")
    if metadata.get("License-Expression") != LICENSE_EXPRESSION:
        raise RuntimeError(f"{source} has unexpected License-Expression: {metadata.get('License-Expression')!r}")
    license_files = set(metadata.get_all("License-File", ()))
    if license_files != LICENSE_FILES:
        raise RuntimeError(
            f"{source} License-File headers do not match the aggregate license set: "
            f"expected {sorted(LICENSE_FILES)}, got {sorted(license_files)}"
        )
    return metadata


def _entry_points(contents: str, *, source: str) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(contents)
    scripts = dict(parser.items("console_scripts")) if parser.has_section("console_scripts") else {}
    if scripts != ENTRY_POINTS:
        raise RuntimeError(
            f"{source} console scripts do not match pyproject.toml: expected {ENTRY_POINTS}, got {scripts}"
        )


def _expected_python_files(source_root: Path) -> set[str]:
    if not source_root.is_dir():
        raise ValueError(f"source package root is unavailable: {source_root}")
    return {f"diffusers_3d/{source.relative_to(source_root).as_posix()}" for source in source_root.rglob("*.py")}


def verify_wheel(path: Path, *, source_root: Path | None = None) -> None:
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"expected one wheel file, got {path}")
    source_root = source_root or Path(__file__).parents[1] / "src" / "diffusers_3d"
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        dist_info_directories = {
            PurePosixPath(name).parts[0]
            for name in names
            if PurePosixPath(name).parts and PurePosixPath(name).parts[0].endswith(".dist-info")
        }
        if len(dist_info_directories) != 1:
            raise RuntimeError(f"wheel must contain exactly one .dist-info directory: {sorted(dist_info_directories)}")
        dist_info = next(iter(dist_info_directories))
        metadata_path = f"{dist_info}/METADATA"
        entry_points_path = f"{dist_info}/entry_points.txt"
        for required in (metadata_path, entry_points_path):
            if required not in names:
                raise RuntimeError(f"wheel is missing required distribution metadata: {required}")
        _distribution_metadata(archive.read(metadata_path), source=metadata_path)
        _entry_points(archive.read(entry_points_path).decode("utf-8"), source=entry_points_path)

        expected_license_paths = {f"{dist_info}/licenses/{license_file}" for license_file in LICENSE_FILES}
        actual_license_paths = {
            name for name in names if name.startswith(f"{dist_info}/licenses/") and not name.endswith("/")
        }
        if actual_license_paths != expected_license_paths:
            raise RuntimeError(
                "wheel .dist-info license files do not match METADATA: "
                f"expected {sorted(expected_license_paths)}, got {sorted(actual_license_paths)}"
            )

    for family, license_name in FAMILIES.items():
        family_root = f"diffusers_3d/families/{family}"
        for filename in ("README.md", "NOTICE", license_name, "diffusers_3d_integration.json"):
            required = f"{family_root}/{filename}"
            if required not in names:
                raise RuntimeError(f"wheel is missing required family artifact: {required}")

    manifests = {name for name in names if name.endswith("/diffusers_3d_integration.json")}
    expected_manifests = {f"diffusers_3d/families/{family}/diffusers_3d_integration.json" for family in FAMILIES}
    if manifests != expected_manifests:
        raise RuntimeError(
            f"wheel manifest set does not match shipped families: expected {sorted(expected_manifests)}, "
            f"got {sorted(manifests)}"
        )

    expected_python_files = _expected_python_files(source_root)
    actual_python_files = {name for name in names if name.startswith("diffusers_3d/") and name.endswith(".py")}
    if actual_python_files != expected_python_files:
        raise RuntimeError(
            "wheel Python module set does not match src/diffusers_3d: "
            f"missing {sorted(expected_python_files - actual_python_files)}, "
            f"unexpected {sorted(actual_python_files - expected_python_files)}"
        )

    forbidden_parts = {"__pycache__", ".pytest_cache", "build", "dist", "tests", "tools"}
    forbidden_suffixes = (".pyc", ".pyo", ".tmp")
    forbidden_names = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            forbidden_parts.intersection(parts)
            or name.endswith(forbidden_suffixes)
            or any(part.endswith(".egg-info") for part in parts)
            or name.endswith(".dist-info/SOURCES.txt")
            or name in {".coverage", "MANIFEST.in", "pyproject.toml", "setup.cfg", "setup.py"}
        ):
            forbidden_names.append(name)
    if forbidden_names:
        raise RuntimeError(f"wheel contains source-only or temporary artifacts: {sorted(forbidden_names)}")

    top_levels = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if "diffusers_3d" not in top_levels or not any(name.endswith(".dist-info") for name in top_levels):
        raise RuntimeError(f"wheel has unexpected top-level layout: {sorted(top_levels)}")
    unexpected = {name for name in top_levels if name != "diffusers_3d" and not name.endswith(".dist-info")}
    if unexpected:
        raise RuntimeError(f"wheel contains unexpected top-level paths: {sorted(unexpected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify diffusers-3d wheel contents.")
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).parents[1] / "src" / "diffusers_3d",
        help="Source package used to assert that every Python module is present.",
    )
    arguments = parser.parse_args()
    verify_wheel(arguments.wheel, source_root=arguments.source_root)
    print(f"verified wheel contents: {arguments.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
