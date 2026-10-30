# Backends

`diffusers_3d.backends` holds plain-PyTorch rasterizers and bakers, plus thin adapters over optional libraries.
Importing the package imports none of those libraries. Each adapter imports its library when you construct it, and
raises an `ImportError` that names the install command when the library is missing.

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
- O-Voxel conversion to and from upstream's tensors, and lossless `.npz` read/write (`OVoxelAsset.save`/`load`).
- `splat_ply`: Gaussian splats in the 3D Gaussian Splatting PLY layout (`GaussianSplatAsset.save`/`load`).

## Portable

Install with the `portable` extra.

- `TrimeshBackend`: OBJ, PLY, GLB, and STL I/O, plus quadric decimation through `fast-simplification`. It refuses to
  write channels or transforms that the target format cannot hold.
- `ScikitImageBackend`: marching cubes.
- `XAtlasBackend`: UV unwrapping.

`MeshAsset.save`, `load`, `simplify`, `repair`, and `unwrap_uv` call these for you. The adapters expose the same
operations as named methods (`TrimeshBackend().simplify(mesh, face_count)`, `XAtlasBackend().unwrap_uv(mesh)`).

## Background removal

Each pipeline declares its upstream remover in `background_removal` (a `BackgroundRemoval`: remover, trigger, the
resize upstream applies first, and whether it runs automatically); see the table in [inference.md](inference.md).

- `RembgBackgroundRemover` wraps rembg (MIT; `u2net` weights Apache-2.0, downloaded on first use), always with an
  explicit session name, because newer rembg defaults to a non-commercial model.
- `BiRefNetBackgroundRemover` is a plain-PyTorch port of BiRefNet with `ZhengPeng7/BiRefNet`'s MIT weights, bit-identical
  to upstream's code. Its `resize` argument picks the TRELLIS.2/Step1X wrapper (`"pil"`) or the IDOL/TripoSplat one
  (`"tensor"`).

Both need the `background` extra (rembg and torchvision).

## GPU

| Adapter | Used for | Install |
|---|---|---|
| `GsplatBackend` | rasterizing Gaussian splats, and baking TRELLIS textured GLBs | `gaussian` extra |
| `CuMeshBackend` | mesh repair, simplification, and UV unwrap for textured GLBs from TRELLIS and TRELLIS.2 | `requirements/backends/cumesh.txt` |

```bash
pip install --no-build-isolation -r packages/diffusers-3d/requirements/backends/cumesh.txt
```

CuMesh bundles cubvh, which is under an NVIDIA non-commercial license. Loading it warns.

## Textured mesh export

- `textured_mesh_from_ovoxel(asset)` (TRELLIS.2 and Pixal3D) follows upstream `to_glb` with CuMesh. Texel
  rasterization and PBR baking run in plain PyTorch instead of nvdiffrast and FlexGEMM.
- `textured_mesh_from_splats(mesh, gaussians)` (TRELLIS) uses CuMesh, xatlas, and gsplat renders
  baked with `texture_baking`.

Write the result with `mesh.save("textured.glb")`.
