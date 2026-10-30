# Hunyuan3D-2.1 shape and Paint

Tencent Hunyuan3D-2.1, ported from upstream revision `82920d643c0dc2f7bfd7255f45f62d386edfe60c` with weights from
`tencent/Hunyuan3D-2.1` at `0b94677654c57bb9a6b6845cd7b704ccf551d327`. The shape stage turns one image into an
untextured mesh. The Paint stage gives a mesh PBR textures from a reference image. Everything runs in plain PyTorch on
any device.

**License.** Code and weights are under the [Tencent Hunyuan 3D 2.1 Community License](LICENSE-TENCENT-HUNYUAN-3D-2.1),
not Apache-2.0. It does not apply in the EU, the UK, or South Korea. It also requires a separate license above
1 million monthly active users and forbids using outputs to improve other AI models. Constructing either pipeline
warns about it.

```python
import torch
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition

shape = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = shape(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

paint = AutoPipelineFor3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d", dtype=torch.float16).to("cuda")
textured = paint(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
textured.save("out.glb")
```

- The foreground comes from alpha or a `mask`.
- Shape defaults: 50 steps, guidance 5.0, a 385³ occupancy grid (`octree_resolution=384`), iso-level 0. Meshes are
  right-handed Y-up inside `[-1.01, 1.01]³`. Marching cubes needs `pip install "diffusers-3d[portable]"`.
- Paint defaults follow `demo.py`: the mesh is read as its GLB export holds it, remeshed as `remesh_mesh` does
  (pymeshlab OBJ round trip, then open3d quadric decimation to `max_faces=40000`; `None` skips it), unwrapped with
  `xatlas.parametrize`, and rendered from `max_num_views=6` fixed views. Raising `max_num_views` adds ±20° views that
  uncover more than 1% of the surface. It then runs 15 UniPC steps at guidance 3.0, upscales views 4x with
  Real-ESRGAN, bakes a 4096 px atlas, inpaints it, and writes it as upstream's OBJ export does: halved to 2048 px,
  truncated to 8 bits, and JPEG-compressed.
- Paint returns one `PBRMaterial` with the decoded base-colour, metallic, and roughness JPEGs, on the vertices and
  UVs of that OBJ (six decimals), in the input's right-handed Y-up frame. Remeshing needs
  `pip install "diffusers-3d[meshlab]" open3d`.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2Model` | `facebook/dinov2-large` at 518 px, loaded from there; all-zero unconditional context |
| `transformer` | `Hunyuan3DDiTModel` | 21-block DiT with MoE in the last 6 blocks; LoRA-loadable |
| `vae` | `Hunyuan3DShapeVAE` | ShapeVAE decoder to an occupancy field |
| `scheduler` | `Hunyuan3DFlowMatchEulerDiscreteScheduler` | stock flow-match Euler with Hunyuan's sigma schedule |

Paint:

| Component | Class | Notes |
| --- | --- | --- |
| `feature_extractor`, `image_encoder` | `transformers.BitImageProcessorPil`, `Dinov2Model` | DINOv2-giant; loads from `facebook/dinov2-giant` |
| `unet` | `Hunyuan3DPaintUNet` | two stock `UNet2DConditionModel`s plus material, reference, and multiview attention |
| `vae` | `diffusers.AutoencoderKL` | the Stable Diffusion 2.1 VAE; loads from `sd2-community/stable-diffusion-2-1` |
| `scheduler` | `diffusers.UniPCMultistepScheduler` | v-prediction, zero terminal SNR, trailing timesteps |
| `upscaler` (optional) | `Hunyuan3DPaintRRDBNet` | Real-ESRGAN x4plus; loads from its GitHub release (SHA-256 checked) |

The converted Paint folder holds only the UNet and the scheduler. The other components are third-party and load from
their original sources at pinned revisions (`paint_components.py`): the release's VAE equals Stable Diffusion 2.1's
tensor for tensor (Stability AI's own repository is gone, so the community copy is used), and upstream itself loads
DINOv2 and Real-ESRGAN. Pass any of them to `from_pretrained` to use your own, or `upscaler=None` to skip the
upscaler.

## Differences from upstream

Shape:

- Meshes are merged by trimesh exactly as upstream's export. A field that never crosses `mc_level` raises instead of
  returning `None`.
- Not ported, and not on the default path of `demo.py` (which runs neither `enable_flashvdm` nor a post-processor):
  FlashVDM (adaptive key/value selection), DISO dual marching cubes, the Gradio app's pymeshlab clean-up, the
  ShapeVAE encoder, multi-view and text conditioning, and training. Upstream's own `HierarchicalVolumeDecoding`
  (FlashVDM without key/value selection) truncates its refined query points to integers, so every refined query
  lands on `int(bbox_min)`.

Paint:

- Rasterization: on CUDA the package runs `custom_rasterizer` itself (optional; install it as
  `diffusers_3d.backends.mesh_rasterizer.CUSTOM_RASTERIZER_INSTALL` says). The plain-PyTorch rasterizer it falls back
  to elsewhere is not equal: over the 18 candidate views and two UV rasters of a 27k-face mesh it picks a different
  face on 14 of 43 million covered pixels (no coverage differences), and its barycentrics differ by up to 5e-4.
- Exact against the pinned code: the mesh remesh (pymeshlab, trimesh, open3d calls; checked with upstream's pins on
  a 76k-face mesh without decimation), `meshVerticeInpaint` (C++ arithmetic), `cv2.inpaint`, and the JPEG export.
  Version- and platform-dependent: pymeshlab newer than 2022.2.post3 writes 16 decimals where upstream's writes six
  (the package rewrites the OBJ to six); xatlas 0.0.9 (upstream's pin, no wheel for recent Pythons) and newer or
  arm64 builds give different atlases; OpenCV's float `cv2.resize` differs by up to 1.5 ulp between OpenCV 4.10 on
  x86-64 and 5.0 on arm64, which shifts a few 8-bit texels and then the JPEG.
- The GLB that upstream also writes goes through Blender (`bpy`); the package exports the OBJ's textures instead.
- Kept on purpose to match the released pipeline: every view uses the same guidance, coarse position grids average in
  a zero background, the boundary kernel set during view selection (4 px) is used for baking, only the first two
  upscaled views are resized to the render size, and Real-ESRGAN gets swapped channels (upstream hands its BGR
  interface RGB images).
- Noise: the latents come from `generator` and the VAE posteriors from a copy of its starting state, which is what
  upstream's seeding (generator and global RNG both seeded 0) gives.

## Conversion

```bash
diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 hunyuan3d-2.1
diffusers-3d-convert-hunyuan3d-paint tencent/Hunyuan3D-2.1 hunyuan3d-2.1-paint
```

The source is a Hub id or a local release folder; `--revision` defaults to the pinned weights. The shape converter
remaps `model.fp16.ckpt` and permutes the fused q/k/v rows so per-projection head splitting gives the same attention.
It does not write the conditioner's DINOv2, which is `facebook/dinov2-large` cast to float16 tensor for tensor; the
pipeline loads it from Meta's repository in the pipeline's dtype.
The Paint converter maps the UNets and writes them with the scheduler; the third-party components stay out.

## Verification

Tiny-weight CPU tests compare the transformer, VAE, scheduler, image preprocessing, vertex merging, and upstream's own
shape pipeline end to end; another checks the shared hierarchical grid decoder against FlashVDM's refinement. Paint
tests compare the UNets, rotary embeddings, view selection, and baking. Set `DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT` to
the upstream checkout to run them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-TENCENT-HUNYUAN-3D-2.1](LICENSE-TENCENT-HUNYUAN-3D-2.1) and [NOTICE](NOTICE). Paint also carries the
CreativeML Open RAIL++-M terms of Stable Diffusion 2.1 and the BSD-3-Clause terms of Real-ESRGAN.
