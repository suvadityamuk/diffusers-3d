from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[2]
PINNED_SOURCE_REQUIREMENTS = {
    "cumesh.txt": ("cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f"),
    "diffoctreerast.txt": (
        "diffoctreerast @ git+https://github.com/JeffreyXiang/diffoctreerast.git"
        "@b09c20b84ec3aace4729e6e18a613112320eca3a"
    ),
    "flex-gemm.txt": (
        "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e"
    ),
    "mip-splatting.txt": (
        "diff_gaussian_rasterization @ git+https://github.com/autonomousvision/mip-splatting.git"
        "@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
        "#subdirectory=submodules/diff-gaussian-rasterization"
    ),
    "nvdiffrast.txt": (
        "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
    ),
    "o-voxel.txt": (
        "o_voxel @ git+https://github.com/microsoft/TRELLIS.2.git"
        "@75fbf0183001ed9876c8dbb35de6b68552ee08bd#subdirectory=o-voxel"
    ),
}


@pytest.mark.release
def test_distribution_does_not_require_opencv():
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"opencv-python-headless"' not in pyproject
    assert '"opencv-python"' not in pyproject


@pytest.mark.release
def test_source_backend_requirement_records_are_immutable():
    requirements_root = PACKAGE_ROOT / "requirements" / "backends"
    for filename, expected in PINNED_SOURCE_REQUIREMENTS.items():
        requirement_lines = [
            line
            for line in (requirements_root / filename).read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        assert requirement_lines == [expected]
    nvdiffrec_record = (requirements_root / "nvdiffrec.txt").read_text(encoding="utf-8")
    assert "git+https://github.com/JeffreyXiang/nvdiffrec.git@a3e73909a01887c8a135235ff860dd23a045cc1b" in (
        nvdiffrec_record
    )
    assert not [line for line in nvdiffrec_record.splitlines() if line and not line.startswith("#")]


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
