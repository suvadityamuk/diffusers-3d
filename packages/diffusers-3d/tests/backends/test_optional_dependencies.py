import subprocess
import sys

import pytest

from diffusers_3d.backends import _optional


def test_missing_dependency_raises_an_install_hint():
    with pytest.raises(ImportError, match=r'pip install "diffusers-3d\[portable\]"'):
        _optional.require("diffusers_3d_missing_module", "portable")


def test_importing_the_package_imports_no_optional_library():
    optional = ("trimesh", "skimage", "xatlas", "gsplat", "cumesh", "fast_simplification")
    code = f"import sys, diffusers_3d; print(sorted(m for m in {optional!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
