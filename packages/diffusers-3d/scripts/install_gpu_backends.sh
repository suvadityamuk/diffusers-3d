#!/usr/bin/env bash
# Build the accelerated backends at the revisions the package registry pins.
#
# Needs a CUDA toolchain (nvcc) matching the installed torch and an already installed diffusers-3d, whose
# registry supplies the URLs and revisions so this script cannot drift from the package's provenance checks.
#
#   ACCEPT_NVDIFFRAST_RESEARCH_LICENSE=1   also build nvdiffrast and the O-Voxel runtime (which imports it).
#                                          nvdiffrast is research/evaluation only; read its license first.
#   TORCH_CUDA_ARCH_LIST                   defaults to 8.0 (A100).
#   MAX_JOBS                               parallel nvcc jobs, defaults to 8.
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

pin() {
  python - "$1" <<'PY'
import sys
from diffusers_3d.backends import BACKEND_REGISTRY

spec = BACKEND_REGISTRY.get(sys.argv[1])
print(f"git+{spec.source_url}@{spec.source_revision}")
PY
}

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step "build tools"
python -m pip install -q ninja

step "utils3d (pinned source, not the PyPI project of the same name)"
python -m pip install -q "$(pin utils3d)"

step "FlexGEMM"
python -m pip install -q --no-build-isolation "$(pin flex_gemm)"

step "CuMesh"
python -m pip install -q --no-build-isolation "$(pin cumesh)"

step "gsplat"
python -m pip install -q gsplat

if [[ "${ACCEPT_NVDIFFRAST_RESEARCH_LICENSE:-0}" == "1" ]]; then
  step "nvdiffrast (research license acknowledged)"
  python -m pip install -q --no-build-isolation "$(pin nvdiffrast)"
  step "O-Voxel runtime"
  # --no-deps keeps pip from re-resolving CuMesh/FlexGEMM from unpinned git URLs; list its pure-Python needs by hand.
  # cv2 is imported by its postprocess module even though the upstream metadata omits it.
  python -m pip install -q easydict opencv-python-headless plyfile tqdm trimesh zstandard
  python -m pip install -q --no-deps --no-build-isolation "$(pin o_voxel)#subdirectory=o-voxel"
else
  echo; echo "=== skipping nvdiffrast and O-Voxel: set ACCEPT_NVDIFFRAST_RESEARCH_LICENSE=1 to build them"
fi

step "installed backends"
python - <<'PY'
from diffusers_3d.backends import BACKEND_REGISTRY, discover_backend

for name in ("utils3d", "flex_gemm", "cumesh", "gsplat", "nvdiffrast", "o_voxel"):
    status = discover_backend(BACKEND_REGISTRY.get(name))
    print(f"{name:12s} installed={status.installed} importable={status.importable} provenance={status.provenance_verified}")
PY
