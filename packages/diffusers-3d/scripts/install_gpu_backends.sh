#!/usr/bin/env bash
# Build CuMesh (used by the textured-GLB exports of TRELLIS and TRELLIS.2) and install gsplat.
#
# Needs a CUDA toolchain (nvcc) matching the installed torch.
#   TORCH_CUDA_ARCH_LIST   defaults to 8.0 (A100).
#   MAX_JOBS               parallel nvcc jobs, defaults to 8.
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
here="$(cd "$(dirname "$0")/.." && pwd)"

python -m pip install -q ninja gsplat
python -m pip install -q --no-build-isolation -r "$here/requirements/backends/cumesh.txt"
python -c "import cumesh, gsplat; print('cumesh and gsplat import')"
