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
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition, TrimeshBackend

shape = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = shape(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

paint = AutoPipelineFor3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d", dtype=torch.float16).to("cuda")
textured = paint(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
TrimeshBackend().export_mesh(textured, "out.glb")
```

- The foreground comes from alpha or a `mask`.
- Shape defaults: 50 steps, guidance 5.0, a 385³ occupancy grid (`octree_resolution=384`), iso-level 0. Meshes are
  right-handed Y-up inside `[-1.01, 1.01]³`. Marching cubes needs `pip install "diffusers-3d[portable]"`.
- Paint defaults: the mesh is simplified to `max_faces=40000` (`None` keeps it), unwrapped with xatlas, and rendered
  from `max_num_views=6` fixed views. Raising `max_num_views` adds ±20° views that uncover more than 1% of the surface.
  It then runs 15 UniPC steps at guidance 3.0, upscales views 4x with Real-ESRGAN, and bakes a 2048 px texture.
- Paint returns one `PBRMaterial` with base-colour, metallic, and roughness textures.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2Model` | DINOv2-L/14 at 518 px; all-zero unconditional context |
| `transformer` | `Hunyuan3DDiTModel` | 21-block DiT with MoE in the last 6 blocks; LoRA-loadable |
| `vae` | `Hunyuan3DShapeVAE` | ShapeVAE decoder to an occupancy field |
| `scheduler` | `Hunyuan3DFlowMatchEulerDiscreteScheduler` | stock flow-match Euler with Hunyuan's sigma schedule |

Paint:

| Component | Class | Notes |
| --- | --- | --- |
| `feature_extractor`, `image_encoder` | `transformers.BitImageProcessorPil`, `Dinov2Model` | DINOv2-giant |
| `unet` | `Hunyuan3DPaintUNet` | two stock `UNet2DConditionModel`s plus material, reference, and multiview attention |
| `vae` | `diffusers.AutoencoderKL` | Stable Diffusion 2.1 VAE |
| `scheduler` | `diffusers.UniPCMultistepScheduler` | v-prediction, zero terminal SNR, trailing timesteps |
| `upscaler` (optional) | `Hunyuan3DPaintRRDBNet` | Real-ESRGAN x4plus |

To skip the upscaler, convert with `--no-upscaler`, or load with
`Hunyuan3DPaintPipeline.from_pretrained(..., upscaler=None)` (from `diffusers_3d.families.hunyuan3d`).

## Differences from upstream

Shape:

- OpenCV is not a dependency. `resize_like_opencv` reproduces its resize in NumPy, within one level on a few pixels.
- The mesh is returned unmerged; upstream's trimesh merges duplicate vertices. A field that never crosses `mc_level`
  raises instead of returning `None`.
- Not ported: FlashVDM and DISO (opt-in accelerations), the ShapeVAE encoder, multi-view and text conditioning, mesh
  post-processing, and training.

Paint:

- Rasterization, baking, and inpainting use plain PyTorch instead of compiled extensions. Uncovered texels are filled
  ring by ring instead of with `cv2.inpaint`, so those texels differ from upstream.
- The mesh is simplified with fast-simplification instead of pymeshlab and trimesh, so meshes and atlases differ.
- Textures stay float and export as PNG; upstream writes 8-bit JPEG.
- Kept on purpose to match the released pipeline: every view uses the same guidance, coarse position grids average in
  a zero background, the bake margin is computed at 1024 px, and Real-ESRGAN gets swapped channels.
- Noise comes from `generator`. Upstream samples VAE posteriors from the global RNG.

## Conversion

```bash
diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 hunyuan3d-2.1
diffusers-3d-convert-hunyuan3d-paint tencent/Hunyuan3D-2.1 hunyuan3d-2.1-paint
```

The source is a Hub id or a local release folder; `--revision` defaults to the pinned weights. The shape converter
remaps `model.fp16.ckpt` and permutes the fused q/k/v rows so per-projection head splitting gives the same attention.
The Paint converter maps the UNets, fetches DINOv2-giant, and fetches Real-ESRGAN with a SHA-256 check.

## Verification

Tiny-weight CPU tests compare the transformer, VAE, scheduler, and upstream's own shape pipeline end to end. Paint
tests compare the UNets, rotary embeddings, view selection, and baking. Set `DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT` to
the upstream checkout to run them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-TENCENT-HUNYUAN-3D-2.1](LICENSE-TENCENT-HUNYUAN-3D-2.1) and [NOTICE](NOTICE). Paint also carries the
CreativeML Open RAIL++-M terms of Stable Diffusion 2.1 and the BSD-3-Clause terms of Real-ESRGAN.
