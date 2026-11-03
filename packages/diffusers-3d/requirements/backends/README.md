# CuMesh

CuMesh has no PyPI wheel, so `cumesh.txt` pins it to an upstream commit. Only the textured-mesh exports of TRELLIS
and TRELLIS.2 (`textured_mesh_from_splats`, `textured_mesh_from_ovoxel`) use it. Build it against your PyTorch
and CUDA toolchain:

```bash
python -m pip install --no-build-isolation -r packages/diffusers-3d/requirements/backends/cumesh.txt
```

CuMesh bundles cubvh under NVIDIA's non-commercial Source Code License. Read it before you install.
