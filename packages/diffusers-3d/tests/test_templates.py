from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1]
TEMPLATES = PACKAGE_ROOT / "templates"


def test_template_python_skeletons_compile():
    python_paths = sorted(TEMPLATES.rglob("*.py"))

    assert python_paths
    for path in python_paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_reviewed_family_template_imports_and_builds_exact_registrations():
    template = TEMPLATES / "reviewed-model-family"
    source = (
        "from reviewed_family.registrations import "
        "REVIEWED_MODEL_REGISTRATION, REVIEWED_PIPELINE_REGISTRATION; "
        "metadata = REVIEWED_PIPELINE_REGISTRATION.metadata; "
        "assert metadata.components[0].name == 'denoiser'; "
        "assert REVIEWED_MODEL_REGISTRATION.metadata.family_id == metadata.family_id"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(template),
            str(PACKAGE_ROOT / "src"),
            environment.get("PYTHONPATH"),
        )
        if value
    )

    subprocess.run([sys.executable, "-c", source], check=True, env=environment)
