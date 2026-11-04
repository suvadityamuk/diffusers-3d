# CuMesh

CuMesh has no PyPI wheel, so `cumesh.txt` pins it to an upstream commit. Only the textured-mesh exports of TRELLIS
and TRELLIS.2 (`textured_mesh_from_splats`, `textured_mesh_from_ovoxel`) use it. Build it against your PyTorch
and CUDA toolchain:

```bash
python -m pip install --no-build-isolation -r packages/diffusers-3d/requirements/backends/cumesh.txt
```

CuMesh bundles cubvh under NVIDIA's non-commercial Source Code License. Read it before you install.

# FlexGEMM

FlexGEMM (MIT) has no PyPI wheel either, so `flex_gemm.txt` pins the commit the TRELLIS.2 release uses. It provides
Triton sparse-convolution kernels, which TRELLIS.2 and Pixal3D use on CUDA once it is installed (other sparse families
opt in with `set_sparse_conv_backend("flex_gemm")`), and the 3D grid sampling of the TRELLIS.2 PBR bake. It compiles a small CUDA extension, so it needs `nvcc`:

```bash
python -m pip install --no-build-isolation -r packages/diffusers-3d/requirements/backends/flex_gemm.txt
```
