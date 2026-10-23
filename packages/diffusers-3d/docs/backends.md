# Backends

Optional libraries sit behind backends in `diffusers_3d.backends`. None is imported when the package is imported, and
none is chosen for you when different choices would change the output. Each backend is registered with a
`BackendSpec` in `BACKEND_REGISTRY` and falls into one of three classes:

- **portable**: CPU, pip-installable, permissive license.
- **accelerated**: compiled or JIT GPU code with a tested compatibility range.
- **research_only**: restricted licenses. Never selected automatically, and loading requires an explicit
  acknowledgement argument.

`diffusers-3d-report` prints which backends are available in the current environment.

## Built-in plain-PyTorch paths

These need no optional dependency and run on any device:

- `mesh_rasterizer`: rasterizes triangles to face index, barycentrics, and depth. With `align_corners=True` it follows
  the `custom_rasterizer` convention used by the Hunyuan3D-2.1 and Step1X-3D texture bakers.
- `multiview_baking`: the multi-view texture baker shared by Hunyuan3D-2.1 Paint and Step1X-3D (camera setup, texel
  rasterization, back-projection, view merging, and inpainting). It replaces upstream's `MeshRender`. The final gap
  fill uses `texture_baking.fill_texture` instead of `cv2.inpaint`.
- `texture_baking`: UV-atlas rasterization and per-texel averaging of rendered views.
- `radiance_field.render_radiance_field`: volume-renders a TRELLIS `RadianceFieldAsset`.
- `ovoxel_to_mesh`: TRELLIS.2 dual-grid mesh extraction, bit-identical to upstream.
- O-Voxel schema conversion and `.npz` read/write.

## Portable

Install with the `portable` extra.

- `TrimeshBackend`: OBJ, PLY, GLB, and STL I/O, plus quadric decimation through `fast-simplification`. It refuses to
  write channels or transforms that the target format cannot hold.
- `ScikitImageBackend`: marching cubes.
- `XAtlasBackend`: UV unwrapping.

## Accelerated

Install these yourself. Each source build is pinned to a commit in `requirements/backends/`, and the backend refuses
to load an install that does not record that commit.

| Backend | Used for | Install |
|---|---|---|
| `CuMeshBackend` | mesh repair, simplification, and UV unwrap for textured GLBs from TRELLIS and TRELLIS.2 | `cumesh.txt` |
| gsplat | rasterizing Gaussian splats, and baking TRELLIS textured GLBs | `gaussian` extra |
| `OVoxelBackend` (native) | `.vxz` files and voxel rendering | `o-voxel.txt`, see below |
| `FlexGemmBackend` | sparse convolution; no pipeline needs it any more | `flex-gemm.txt` |
| spconv | sparse convolution; no pipeline needs it any more | the wheel for your CUDA version |
| `KaolinFlexiCubesBackend` | FlexiCubes surface extraction from the Apache-2.0 `kaolin.ops` module only | a Kaolin build matching your CUDA |

```bash
pip install --no-build-isolation -r packages/diffusers-3d/requirements/backends/cumesh.txt
```

CuMesh bundles cubvh, which is under an NVIDIA non-commercial license. Loading it warns.

To install the native O-Voxel runtime, install FlexGEMM and CuMesh first. Then install O-Voxel without dependency
resolution, so its unpinned Git dependencies cannot replace those pins:

```bash
pip install -r packages/diffusers-3d/requirements/backends/flex-gemm.txt \
            -r packages/diffusers-3d/requirements/backends/cumesh.txt
pip install --no-deps --no-build-isolation -r packages/diffusers-3d/requirements/backends/o-voxel.txt
```

The `o_voxel` package imports nvdiffrast at import time, so native O-Voxel loading also needs the nvdiffrast license
acknowledgement. `.vxz` does not store the grid resolution, so pass it when reading. VXZ v0 only holds uint8
attributes, so assets with float split weights are rejected; use `.npz` for lossless files.

## Textured mesh export

- `Trellis2PBRPostprocessFacade.to_textured_mesh` (TRELLIS.2 and Pixal3D) follows upstream `to_glb` with CuMesh. Texel
  rasterization and PBR baking run in plain PyTorch instead of nvdiffrast and FlexGEMM.
- `TrellisGlbPostprocessFacade.to_textured_mesh(mesh, gaussians)` (TRELLIS) uses CuMesh, xatlas, and gsplat renders
  baked with `texture_baking`.

Export the result with `TrimeshBackend` after converting it to `right_handed_y_up`.

## Research only

nvdiffrast, nvdiffrec, diffoctreerast, mip-splatting, and utils3d (the EasternJournalist repository, not the PyPI
package of the same name) are registered for reference comparisons only. No pipeline uses them. Review their licenses
before installing:

```bash
pip install --no-build-isolation \
  -r packages/diffusers-3d/requirements/backends/nvdiffrast.txt \
  -r packages/diffusers-3d/requirements/backends/diffoctreerast.txt \
  -r packages/diffusers-3d/requirements/backends/mip-splatting.txt
```
