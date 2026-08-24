from __future__ import annotations

from diffusers_3d import DEFAULT_FORBIDDEN_MARKER, scan_forbidden_marker


def test_forbidden_marker_scan_covers_source_and_build_paths(tmp_path):
    source = tmp_path / "src"
    build = tmp_path / "build"
    source.mkdir()
    build.mkdir()
    (source / "clean.py").write_text("clean = True\n", encoding="utf-8")
    (build / "generated.py").write_text(
        f"first = 1\n# {DEFAULT_FORBIDDEN_MARKER}\n",
        encoding="utf-8",
    )

    report = scan_forbidden_marker((source, build))

    assert not report.is_clean
    assert report.scanned_files == 2
    assert not report.failures
    assert len(report.matches) == 1
    assert report.matches[0].path.endswith("generated.py")
    assert report.matches[0].line == 2
    assert report.matches[0].column == 3


def test_forbidden_marker_scan_supports_custom_marker_and_reports_missing_paths(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("CUSTOM_RELEASE_BLOCKER\n", encoding="utf-8")

    custom_report = scan_forbidden_marker((source,), marker="CUSTOM_RELEASE_BLOCKER")
    missing_report = scan_forbidden_marker((tmp_path / "missing",))

    assert len(custom_report.matches) == 1
    assert not custom_report.failures
    assert not missing_report.is_clean
    assert missing_report.failures[0].message == "path does not exist"
