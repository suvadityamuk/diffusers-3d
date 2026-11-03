# TripoSR image-to-3D

Stability AI and Tripo AI's TripoSR, ported from upstream revision `107cefdc244c39106fa830359024f6a2f1c78871` with
weights from `stabilityai/TripoSR` at `5b521936b01fbe1890f6f9baed0254ab6351c04a`. It is a feed-forward LRM-style model
with no sampling loop. A ViT encodes the image, a transformer turns learned tokens into triplanes, and a NeRF MLP
decodes density and colour. Marching cubes then extracts a vertex-coloured mesh. MIT license.

```python
from diffusers_3d import AutoPipelineForImageTo3D

pipeline = AutoPipelineForImageTo3D.from_pretrained("path/to/converted").to("cuda")
output = pipeline("chair.png")
mesh = output.objects[0]
frames = pipeline.render_views(output.latents.latents)  # (1, 30, 256, 256, 3), as run.py --render
```

- RGBA images, or RGB with a `mask`, are cropped to the foreground, padded so it fills `foreground_ratio` (0.85), and
  composited on grey, as upstream `run.py` does. Inputs without usable alpha are first cut out at full size with
  rembg (u2net, the `background` extra), as upstream does by default; `pipeline.disable_background_removal()` matches
  upstream's `--no-remove-bg`.
- Defaults match `run.py`: a 256³ density grid at threshold 25 and 8192-point decoder chunks.
- Meshes come from a plain-PyTorch port of torchmcubes' marching cubes
  ([`backends/marching_cubes.py`](../../backends/marching_cubes.py)), following its CUDA kernel on CUDA and its CPU
  kernel on CPU, as upstream does. The CUDA kernel keeps at most four of the five triangles that 32 of the 256 cube
  cases produce, so those meshes have small holes, as upstream's do. Vertices that agree to 8 decimals are then
  merged, as upstream's `trimesh.Trimesh` does by default. Meshes are returned on the CPU in TripoSR's right-handed
  Z-up frame, with per-vertex colours in `[0, 1]`; trimesh and the package exporter both round them to 8 bits.
  `mesh.save` writes the raw Z-up coordinates, as `run.py` does. The triplanes are returned as latents.
- `render_views` is upstream's NeRF volume renderer (`TSR.render`), returning float frames instead of PIL images or
  a video.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.ViTModel` | DINO ViT-B/16 fine-tuned by TripoSR, at 512 px |
| `transformer` | `TripoSRTransformerModel` | learned triplane tokens plus a stock Diffusers `Transformer2DModel` (16 blocks), then a 2x transposed-conv upsample |
| `decoder` | `TripoSRTriplaneDecoder` | 10-layer SiLU MLP on concatenated triplane features |

Upstream's `Transformer1D` is Diffusers' `BasicTransformerBlock` stack between a group norm, linear projections, and a
residual. That is exactly `Transformer2DModel(use_linear_projection=True)` applied to a `(3·32, 32)` view of the
planes, so the backbone tensors load unchanged. Images are resized with antialiased bilinear `F.interpolate` and
normalized with the ImageNet statistics, as upstream does.

## Differences from upstream

With upstream's CPU defaults, triplanes and rendered frames match upstream bit for bit. Meshes match bit for bit
when upstream runs NumPy transcriptions of torchmcubes' CPU and CUDA sources, and the marching cubes matches the
compiled torchmcubes on both devices (see Verification). What still differs:

- torchmcubes' CPU kernel built with OpenMP numbers vertices in thread completion order; this port uses the
  single-threaded order. The triangles are the same either way.
- The converter bakes the position embeddings at 512 px with the interpolation of transformers 4.35, which upstream
  pins. Transformers 5 interpolates differently: on the release checkpoint it moves the triplanes by up to 182 (out of
  3521) and changes the mesh.
- `render_views` also renders cameras whose rays partly miss the bounding box; upstream raises there. With exactly
  three views, upstream takes its camera cross products over the view axis and gets wrong cameras; here they always
  use the vector axis.
- If the density never crosses the threshold, upstream returns an empty mesh on CPU and a single all-zero triangle on
  CUDA; here `extract_mesh` raises, because a `MeshAsset` needs at least one vertex.
- Not ported: texture baking (`--bake-texture`). It rasterizes the xatlas atlas with OpenGL through moderngl, and
  that output depends on the GPU. On this machine, a PyTorch rasterizer missed or added a few dozen texels, and its
  positions differed by up to 1.5e-5. Training is not ported either.
- Background removal is opt-in; see above.

## Conversion

```bash
diffusers-3d-convert-triposr stabilityai/TripoSR output/
```

The source is a Hub id or a folder with `config.yaml` and `model.ckpt`. `--revision` defaults to the pinned weights.
The ViT config comes from `facebook/dino-vitb16` at a pinned revision. All other tensors keep their values; only key
prefixes change. Transformers maps the legacy ViT tensor names.

## Verification

Tiny-weight CPU tests check preprocessing, the baked position embeddings, a full forward plus mesh extraction, and
`render_views` against the pinned upstream code. Upstream runs with the real `trimesh` and with line-by-line NumPy
transcriptions of torchmcubes' CPU and CUDA kernels in place of the compiled extension (the port is forced to the
matching kernel). The triplanes, density grid, vertices, faces, colours, and frames are compared exactly. The
marching cubes is also compared with both transcriptions on all 256 single-cell cases and on fields with exact zeros
and values in the snapping distance, and its triangle tables with the pinned torchmcubes sources. Set `DIFFUSERS_3D_TRIPOSR_REFERENCE_ROOT` to the TripoSR checkout and
`DIFFUSERS_3D_TORCHMCUBES_REFERENCE_ROOT` to a torchmcubes checkout at `3381600ddc3d2e4d74222f8495866be5fafbace4`
(its head when TripoSR was pinned; TripoSR installs it unpinned). These tests need `omegaconf` and `trimesh`.

Outside the test suite, the marching cubes was compared with torchmcubes built at that revision on a T4 (torch
2.4.1, CUDA 12.1). On 16 volumes up to 256³ (random, smooth, spheres, snapping fields, a TripoSR-like density, and
tiled single cells), which hit all 256 cube cases and 48,423 cells of the 32 five-triangle cases, vertices and faces
were bit-identical on CUDA and, with one OpenMP thread, on CPU. It also matched a prebuilt PyMCubes wheel (the same
Bourke tables) as oriented triangle sets on all 254 non-empty single-cell cases and on random fields. PyMCubes snaps no vertices and merges
vertices by edge, so only the triangles and their winding were compared. The released checkpoint was also compared
on CPU on upstream's `chair.png`, again with the torchmcubes transcription. At the default 256³, the 42,161 vertices
(42,174 before trimesh merges them), the 84,160 faces, and the colours were bit-identical. Rendered frames were
bit-identical with 2 views at 32 px, 4 views at 64 px, and `run.py`'s default 30 views at 256 px.

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
