from __future__ import annotations

import json
import sys

import diffusers_3d
from diffusers_3d import BACKEND_REGISTRY, DEFAULT_BACKEND_SPECS, compatibility_report
from diffusers_3d.compatibility import main


def test_compatibility_report_lists_core_versions_and_registry_without_optional_imports():
    optional_roots = {name.split(".", maxsplit=1)[0] for spec in DEFAULT_BACKEND_SPECS for name in spec.import_names}
    imported_before = {name: name in sys.modules for name in optional_roots}

    report = compatibility_report()

    assert report["package"]["diffusers_3d"] == diffusers_3d.__version__
    assert report["package"]["diffusers"]
    assert report["package"]["python"]
    assert report["package"]["torch"]
    assert [backend["name"] for backend in report["backends"]] == [spec.name for spec in BACKEND_REGISTRY]
    assert {name: name in sys.modules for name in optional_roots} == imported_before
    assert all(
        {
            "available",
            "devices",
            "distribution",
            "dtypes",
            "importable",
            "installed",
            "license_class",
            "name",
            "provenance_verified",
            "reason",
            "support_level",
            "version",
        }
        == set(backend)
        for backend in report["backends"]
    )


def test_compatibility_report_cli_is_json_and_console_script_is_wired(capsys):
    assert main(["--compact"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["package"]["diffusers_3d"] == diffusers_3d.__version__
