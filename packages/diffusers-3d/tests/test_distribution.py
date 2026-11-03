from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
CUMESH_REQUIREMENT = "cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f"


@pytest.mark.release
def test_cumesh_requirement_is_pinned_to_a_commit():
    text = (PACKAGE_ROOT / "requirements" / "backends" / "cumesh.txt").read_text(encoding="utf-8")
    assert [line for line in text.splitlines() if line and not line.startswith("#")] == [CUMESH_REQUIREMENT]


@pytest.mark.release
def test_wheel_and_sdist_are_complete_and_policy_clean(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(PACKAGE_ROOT),
        ],
        check=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "tools" / "verify_wheel.py"), str(wheels[0])],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "tools" / "verify_sdist.py"), str(sdists[0])],
        check=True,
    )
