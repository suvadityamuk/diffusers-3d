# Backend source records

Each `.txt` file pins one optional backend that has no usable PyPI wheel to an immutable upstream commit. Its
comment header names the package and import, the license, and how to build it. No externally hosted wheel is
approved, and mutable branches are not allowed.

These records fix source identity only. They are not a tested build matrix: record the Python, PyTorch, CUDA/HIP,
compiler, operating system, and GPU architecture of any build you rely on.

## TRELLIS.2 native backends

`o_voxel` names FlexGEMM and CuMesh as unpinned Git dependencies. Install the two pinned records first, then
O-Voxel without dependency resolution so those pins are not replaced:

```bash
python -m pip install \
  -r packages/diffusers-3d/requirements/backends/flex-gemm.txt \
  -r packages/diffusers-3d/requirements/backends/cumesh.txt
python -m pip install --no-deps --no-build-isolation \
  -r packages/diffusers-3d/requirements/backends/o-voxel.txt
```

CuMesh bundles cubvh under NVIDIA's non-commercial Source Code License.

## Research-only backends

nvdiffrast, diffoctreerast, and the mip-splatting Gaussian rasterizer carry non-commercial research licenses.
Review those licenses before installing them:

```bash
python -m pip install --no-build-isolation \
  -r packages/diffusers-3d/requirements/backends/nvdiffrast.txt \
  -r packages/diffusers-3d/requirements/backends/diffoctreerast.txt \
  -r packages/diffusers-3d/requirements/backends/mip-splatting.txt
```

`nvdiffrec.txt` is a source-only record: the pinned tree has no Python build metadata, so it cannot be installed
with pip. The registry reports `nvdiffrec_render` as unavailable unless an audited package records PEP 610
direct-URL provenance for that exact commit.
