from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffusers_3d.contributions.cli import main
from diffusers_3d.contributions.removal import release_check_main

pytestmark = pytest.mark.release

PACKAGE_ROOT = Path(__file__).parents[2]


def test_validate_cli_exit_codes_and_structured_output(tmp_path, manifest_factory, capsys):
    valid_path = manifest_factory().save(tmp_path / "valid.json")
    assert main([str(valid_path)]) == 0
    valid_output = json.loads(capsys.readouterr().out)
    assert valid_output == {"errors": [], "valid": True, "warnings": []}

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"schema": "broken"}', encoding="utf-8")
    assert main([str(invalid_path)]) == 1
    invalid_output = json.loads(capsys.readouterr().out)
    assert invalid_output["valid"] is False
    assert invalid_output["errors"][0]["code"] == "manifest.invalid"


def test_release_cli_exit_codes(tmp_path, capsys):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "module.py").write_text("value = 1\n", encoding="utf-8")
    assert release_check_main([str(clean), "--marker", "FORBIDDEN_RELEASE_VALUE"]) == 0
    assert json.loads(capsys.readouterr().out)["clean"] is True

    (clean / "module.py").write_text("FORBIDDEN_RELEASE_VALUE\n", encoding="utf-8")
    assert release_check_main([str(clean), "--marker", "FORBIDDEN_RELEASE_VALUE"]) == 1
    assert json.loads(capsys.readouterr().out)["clean"] is False


def test_console_scripts_are_wired_in_pyproject():
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'diffusers-3d-validate = "diffusers_3d.contributions.cli:main"' in pyproject
    assert 'diffusers-3d-check-release = "diffusers_3d.contributions.removal:release_check_main"' in pyproject
    assert 'diffusers-3d-report = "diffusers_3d.compatibility:main"' in pyproject
