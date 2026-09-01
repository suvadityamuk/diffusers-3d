from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FORBIDDEN_MARKER = "".join(("OBJECT3D_CONTRACT_", "VALIDATION_ONLY"))


@dataclass(frozen=True, slots=True)
class ForbiddenMarkerMatch3D:
    """One forbidden marker occurrence in a release input."""

    path: str
    line: int
    column: int

    def to_dict(self) -> dict[str, object]:
        return {"column": self.column, "line": self.line, "path": self.path}


@dataclass(frozen=True, slots=True)
class ReleaseScanFailure3D:
    """A source or build path that could not be scanned."""

    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"message": self.message, "path": self.path}


@dataclass(frozen=True, slots=True)
class ReleaseRemovalReport3D:
    """Deterministic forbidden-marker scan results."""

    marker: str
    scanned_files: int
    matches: tuple[ForbiddenMarkerMatch3D, ...]
    failures: tuple[ReleaseScanFailure3D, ...]

    @property
    def is_clean(self) -> bool:
        return not self.matches and not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "clean": self.is_clean,
            "failures": [failure.to_dict() for failure in self.failures],
            "marker": self.marker,
            "matches": [match.to_dict() for match in self.matches],
            "scanned_files": self.scanned_files,
        }


def scan_forbidden_marker(
    paths: Sequence[str | Path],
    *,
    marker: str = DEFAULT_FORBIDDEN_MARKER,
) -> ReleaseRemovalReport3D:
    """Scan caller-selected source and build paths for a release-blocking marker."""

    if isinstance(paths, (str, Path)) or not isinstance(paths, Sequence) or not paths:
        raise TypeError("paths must be a non-empty sequence of paths")
    if not isinstance(marker, str) or not marker:
        raise ValueError("marker must be a non-empty string")
    try:
        marker_bytes = marker.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("marker must be UTF-8 encodable") from error

    files: set[Path] = set()
    failures: list[ReleaseScanFailure3D] = []
    for raw_path in paths:
        try:
            path = Path(raw_path)
        except TypeError:
            failures.append(ReleaseScanFailure3D(path=repr(raw_path), message="path is not path-like"))
            continue
        if not path.exists():
            failures.append(ReleaseScanFailure3D(path=str(path), message="path does not exist"))
        elif path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            failures.append(ReleaseScanFailure3D(path=str(path), message="path is not a regular file or directory"))

    matches: list[ForbiddenMarkerMatch3D] = []
    scanned_files = 0
    for path in sorted(files, key=lambda item: str(item)):
        try:
            contents = path.read_bytes()
        except OSError as error:
            failures.append(ReleaseScanFailure3D(path=str(path), message=str(error)))
            continue
        scanned_files += 1
        offset = contents.find(marker_bytes)
        while offset >= 0:
            line = contents.count(b"\n", 0, offset) + 1
            previous_newline = contents.rfind(b"\n", 0, offset)
            column = offset - previous_newline
            matches.append(ForbiddenMarkerMatch3D(path=str(path), line=line, column=column))
            offset = contents.find(marker_bytes, offset + len(marker_bytes))

    return ReleaseRemovalReport3D(
        marker=marker,
        scanned_files=scanned_files,
        matches=tuple(matches),
        failures=tuple(sorted(failures, key=lambda item: (item.path, item.message))),
    )


def release_check_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diffusers-3d-check-release",
        description="Scan selected source/build paths for a reserved release-blocking marker.",
    )
    parser.add_argument("paths", nargs="+", help="Source or build files/directories to scan.")
    parser.add_argument(
        "--marker",
        default=DEFAULT_FORBIDDEN_MARKER,
        help="Forbidden marker text (defaults to the package's reserved release marker).",
    )
    arguments = parser.parse_args(argv)
    report = scan_forbidden_marker(arguments.paths, marker=arguments.marker)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.is_clean else 1


__all__ = [
    "DEFAULT_FORBIDDEN_MARKER",
    "ForbiddenMarkerMatch3D",
    "ReleaseRemovalReport3D",
    "ReleaseScanFailure3D",
    "release_check_main",
    "scan_forbidden_marker",
]
