from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath

from verify_wheel import LICENSE_FILES, _distribution_metadata, _entry_points


def _required_source_files(package_root: Path) -> set[str]:
    required = {
        "COMPATIBILITY.md",
        "CONTRIBUTING.md",
        "LICENSE-APACHE-2.0",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
    }
    patterns = {
        "docs": ("*.md",),
        "requirements": ("*.md", "*.txt"),
        "src/diffusers_3d": ("*.json", "*.md", "*.py", "LICENSE*", "NOTICE", "py.typed"),
        "templates": ("*.json", "*.md", "*.py"),
        "tests": ("*.py",),
        "tools": ("*.py",),
    }
    for directory, globs in patterns.items():
        root = package_root / directory
        if not root.is_dir():
            raise ValueError(f"required source directory is unavailable: {root}")
        for pattern in globs:
            required.update(
                path.relative_to(package_root).as_posix() for path in root.rglob(pattern) if path.is_file()
            )
    return required


def verify_sdist(path: Path, *, package_root: Path | None = None) -> None:
    if not path.is_file() or not path.name.endswith(".tar.gz"):
        raise ValueError(f"expected one .tar.gz source distribution, got {path}")
    package_root = package_root or Path(__file__).parents[1]

    with tarfile.open(path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise RuntimeError(f"sdist must contain exactly one top-level directory: {sorted(roots)}")
        archive_root = next(iter(roots))
        names = {PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix() for member in members}

        required_source_files = _required_source_files(package_root)
        missing = required_source_files - names
        if missing:
            raise RuntimeError(f"sdist is missing required source files: {sorted(missing)}")

        metadata_path = "PKG-INFO"
        entry_points_path = "src/diffusers_3d.egg-info/entry_points.txt"
        archive_names = {member.name: member for member in members}
        for required in (metadata_path, entry_points_path):
            full_path = f"{archive_root}/{required}"
            if full_path not in archive_names:
                raise RuntimeError(f"sdist is missing generated metadata: {required}")
        metadata_member = archive.extractfile(archive_names[f"{archive_root}/{metadata_path}"])
        entry_points_member = archive.extractfile(archive_names[f"{archive_root}/{entry_points_path}"])
        if metadata_member is None or entry_points_member is None:
            raise RuntimeError("sdist metadata members could not be read")
        _distribution_metadata(metadata_member.read(), source=metadata_path)
        _entry_points(entry_points_member.read().decode("utf-8"), source=entry_points_path)

    missing_license_files = LICENSE_FILES - names
    if missing_license_files:
        raise RuntimeError(f"sdist is missing declared license files: {sorted(missing_license_files)}")

    forbidden_parts = {"__pycache__", ".pytest_cache", "build", "dist", "wheelhouse", "wheel-unpacked"}
    forbidden_suffixes = (".pyc", ".pyo", ".tmp")
    forbidden_names = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            forbidden_parts.intersection(parts)
            or name.endswith(forbidden_suffixes)
            or name in {".coverage", ".DS_Store"}
            or "htmlcov" in parts
        ):
            forbidden_names.append(name)
    if forbidden_names:
        raise RuntimeError(f"sdist contains temporary or generated artifacts: {sorted(forbidden_names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify diffusers-3d source distribution contents.")
    parser.add_argument("sdist", type=Path)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).parents[1],
        help="Source tree used to assert intentional sdist completeness.",
    )
    arguments = parser.parse_args()
    verify_sdist(arguments.sdist, package_root=arguments.package_root)
    print(f"verified sdist contents: {arguments.sdist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
