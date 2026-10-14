# Hunyuan3D-2.1: image-to-shape and Paint

This family ports the shape stage and the PBR texture stage (Hunyuan3D-Paint 2.1) of Tencent Hunyuan3D-2.1 at
upstream revision `82920d643c0dc2f7bfd7255f45f62d386edfe60c`, and converts the released weights from the Hub
repository `tencent/Hunyuan3D-2.1` (revision `0b94677654c57bb9a6b6845cd7b704ccf551d327`, folders `hunyuan3d-dit-v2-1`
and `hunyuan3d-paintpbr-v2-1`). Everything runs in plain PyTorch on any device. Mesh extraction, UV unwrapping, and
simplification need the portable extra (`pip install "diffusers-3d[portable]"`).

> **License.** The code in this directory and the converted weights are governed by the
> [Tencent Hunyuan 3D 2.1 Community License](LICENSE-TENCENT-HUNYUAN-3D-2.1), not by Apache-2.0. The license **does
> not apply in the European Union, the United Kingdom, or South Korea**, requires a separate license from Tencent for
> products above 1 million monthly active users, forbids using outputs to improve other AI models, and incorporates
> an Acceptable Use Policy. Constructing `Hunyuan3DImageTo3DPipeline` or `Hunyuan3DPaintPipeline` emits a
> `UserWarning` saying so. The Paint UNets and VAE also carry the CreativeML Open RAIL++-M terms of Stable Diffusion
> 2.1, and the upscaler Real-ESRGAN's BSD-3-Clause terms. See [NOTICE](NOTICE).

## Usage

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

`rgba` is a `(4, H, W)` tensor in `[0, 1]` whose alpha channel masks the object. To convert the release yourself,
see Conversion below.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2Model` | DINOv2-L/14 at 518 px; CLS and patch tokens after the final layer norm condition the DiT; all-zero unconditional context |
| `transformer` | `Hunyuan3DDiTModel` | 21-block U-skip DiT, hidden size 2048, MoE (8 experts, top 2, one shared expert) in the last 6 blocks; Diffusers `Timesteps`, `TimestepEmbedding`, `FeedForward` |
| `vae` | `Hunyuan3DShapeVAE` | ShapeVAE decoder: 16 self-attention blocks over the 4096 x 64 latent set, then a cross-attention occupancy query |
| `scheduler` | `Hunyuan3DFlowMatchEulerDiscreteScheduler` | Diffusers `FlowMatchEulerDiscreteScheduler` with Hunyuan's schedule: sigmas `linspace(0, 1, steps)` then a terminal 1 |

`Hunyuan3DImageTo3DPipeline` returns one `MeshAsset` (right-handed, Y up, inside `[-1.01, 1.01]^3`) per image with
the released defaults: 50 steps, classifier-free guidance 5.0 (negative disables it), a 385^3 occupancy grid
(`octree_resolution=384`), iso-level 0, and 8000 grid points per decoder chunk. It also supports
`Object3DLoraLoaderMixin` LoRAs on the transformer.

## Conversion

```bash
diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 hunyuan3d-2.1-diffusers   # or a local release folder
```

The converter reads `config.yaml` and the aggregate `model.fp16.ckpt`, rejects any option the release does not use,
remaps keys onto the Diffusers modules (fp16 is kept), re-saves DINOv2 as a stock Transformers component (in
float32), drops the ShapeVAE point-cloud encoder, and writes the
Diffusers folder, the `object3d_model_index.json` sidecar, and `hunyuan3d_conversion.json`. The released attention
layers split heads after concatenating the q/k/v projections; the converter permutes those projection rows once so
that ordinary per-projection head splitting computes the same attention. Every mapped key and shape of the published
checkpoint (752 transformer, 439 DINOv2, and 266 decoder tensors) was checked against these modules.

## Differences from upstream

- **Inputs.** The pipeline takes `ImageCondition` values whose alpha channel or mask already isolates the object; it
  never removes backgrounds. Upstream recenters with OpenCV; `resize_like_opencv` reproduces OpenCV's `INTER_AREA`
  and `INTER_CUBIC` uint8 arithmetic in NumPy (OpenCV is not a dependency). Measured against OpenCV 5.0, the
  preprocessed image differs by one level on 14 of 786,432 values for a shrunk crop and on 1 for an enlarged crop.
- **Surface extraction.** This is the released default path: a dense occupancy grid and scikit-image Lewiner
  marching cubes, with upstream's vertex scaling (divided by `octree_resolution + 1`) and its exported face winding.
  Upstream returns a `trimesh.Trimesh` built with `process=True`, which merges vertices equal to 8 decimals; this
  pipeline returns the marching-cubes mesh unmerged. If the occupancy field never crosses `mc_level`, upstream
  returns `None` for that object while this pipeline raises a `ValueError`. FlashVDM hierarchical decoding and the DISO extractor are
  opt-in upstream accelerations and are not ported.
- **Not ported.** The ShapeVAE encoder (it needs `torch_cluster`), multi-view and text conditioning, training code,
  and upstream's mesh post-processing (floater and degenerate-face removal, face reduction).

## Parity

`tests/families/hunyuan3d/test_reference_parity.py` loads the pinned checkout from
`DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT` and compares tiny random-weight models on CPU in float32: the transformer
output is bit-identical, the VAE decoder within 1e-6, the scheduler bit-identical, and upstream's own
`Hunyuan3DDiTFlowMatchingPipeline.__call__` against this pipeline gives identical conditioning, final latents, and
mesh. The released checkpoint has not been compared with upstream on a GPU. By hand on an A100 (2026-10-13), the
published `suvadityamuk/Hunyuan3D-2.1-diffusers-3d` turned the TRELLIS `T.png` example into a 1.10M-face mesh in
32 s (float16, 7.3 GB peak).

# Hunyuan3D-Paint 2.1

`Hunyuan3DPaintPipeline` takes a `MeshAsset` (for example the shape pipeline's output) and an `ImageCondition` whose
alpha or mask isolates the object, and returns a UV-textured `MeshAsset` with one `PBRMaterial`: a base-colour
texture and metallic and roughness textures. Export it as GLB with `TrimeshBackend().export_mesh(mesh, "out.glb")`.

```python
# A folder written by diffusers-3d-convert-hunyuan3d-paint
pipeline = Hunyuan3DPaintPipeline.from_pretrained("/path/to/hunyuan3d-paint", dtype=torch.float16).to("cuda")
textured = pipeline(mesh, image, generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

The pipeline runs these steps:

1. It merges the mesh's duplicate vertices and simplifies it to `max_faces=40000` faces with quadric decimation
   (`TrimeshBackend`, which needs `fast-simplification`). Pass `max_faces=None` to keep the mesh as it is.
2. It UV-unwraps the mesh with xatlas.
3. It selects views as upstream does: six fixed views, then the candidates at ±20° elevation that each uncover more
   than 1% of the surface, up to `max_num_views`.
4. It renders flat-normal and position maps with the plain-PyTorch rasterizer.
5. It generates albedo and metallic-roughness views with 15 UniPC steps and three-way guidance (3.0).
6. It upscales every view 4x with Real-ESRGAN.
7. It bakes and inpaints the views with `backends/multiview_baking.py`, then box-filters the 4096² atlas to
   `texture_size` (2048).

## Paint components

| Component | Class | Notes |
| --- | --- | --- |
| `feature_extractor` | `transformers.BitImageProcessorPil` | the `facebook/dinov2-giant` preprocessing: Pillow bicubic resize to 256, center crop 224, ImageNet normalization |
| `image_encoder` | `transformers.Dinov2Model` | DINOv2-giant; its final tokens condition the DINO cross-attention |
| `unet` | `Hunyuan3DPaintUNet` | two stock `UNet2DConditionModel`s (Stable Diffusion 2.1 layout) plus learned text tokens and a stock `ImageProjection`; see below |
| `vae` | `diffusers.AutoencoderKL` | Stable Diffusion 2.1 VAE |
| `scheduler` | `diffusers.UniPCMultistepScheduler` | built from the released DDIM config (v-prediction, zero terminal SNR) with trailing timesteps, as upstream does at load time |
| `upscaler` (optional) | `Hunyuan3DPaintRRDBNet` | Real-ESRGAN x4plus. To skip it, pass `upscaler=None` to `Hunyuan3DPaintPipeline.from_pretrained`, or convert with `--no-upscaler`. |

`Hunyuan3DPaintUNet.unet` takes the noisy latents concatenated with the normal and position latents. Its transformer
blocks are `Hunyuan3DPaintTransformerBlock`, a `BasicTransformerBlock` subclass that adds three attentions:

- **Material-aware self-attention.** Self-attention within each view, with separate projections for
  metallic-roughness, held by an attention processor.
- **Reference attention.** Cross-attention to the self-attention inputs of `reference_unet`, which runs once on the
  reference latents at timestep 0.
- **Multiview attention.** Attention across all views with 3D rotary embeddings of the voxelized surface position.
  The frequencies come from Diffusers' `get_1d_rotary_pos_embed` and are applied with `apply_rotary_emb`. Both are
  bit for bit equal to upstream's.

The block also adds a cross-attention to the projected DINOv2 tokens. The released prompt is replaced by learned
token sequences, so the CLIP text encoder, tokenizer, and CLIP image encoder that the release ships are not used and
are not converted.

## Paint conversion

```bash
diffusers-3d-convert-hunyuan3d-paint tencent/Hunyuan3D-2.1 hunyuan3d-paint-2.1-diffusers   # or a local release folder
```

The converter does the following:

- It reads `unet/` and `vae/` with `torch.load(weights_only=True)` and keeps their stored dtypes: float16 for the
  UNets, float32 for the VAE.
- It rejects UNet config options the port does not implement.
- It maps the released `UNet2p5DConditionModel` keys onto `Hunyuan3DPaintUNet`. All 1,747 published tensors map
  with matching shapes.
- It builds the `UniPCMultistepScheduler`.
- It fetches DINOv2-giant from `facebook/dinov2-giant` at revision `611a9d42f2335e0f921f1e313ad3c1b7178d206d`.
- It fetches `RealESRGAN_x4plus.pth` from `lllyasviel/Annotators` and checks it against the SHA-256 of Real-ESRGAN's
  release asset. Pass `--no-upscaler` to omit it.
- It writes safetensors, the `object3d_model_index.json` sidecar, and `hunyuan3d_paint_conversion.json`.

## Paint differences from upstream

- **Rasterization, baking, and inpainting** use `backends/multiview_baking.py` (plain PyTorch, no compiled
  extension). The rasterizer reproduces `custom_rasterizer`, and the depth-edge mask reproduces `cv2.Canny`
  exactly. Upstream fills the texels no view covers in two steps:
  - It colours uncovered vertices from their neighbours with a C++ extension that sweeps vertices in order. The port
    propagates ring by ring instead, which colours the same vertices from the same neighbours but averages in a
    different order.
  - It fills the rest with `cv2.inpaint(INPAINT_NS)` on 8-bit colour. The port fills ring by ring from known
    neighbours (`fill_texture`) on float colour instead. On the trellis test mesh, where 62% of the covered texels
    were filled this way, the two fills differ by 5.4/255 on average on those texels (23.9 dB PSNR). Texels a view
    covers match upstream within 0.6/255 on average (48.9 dB).
- **Inputs and outputs.** The mesh is read in its own coordinate system and world transform. The textured mesh keeps
  the simplified, unwrapped geometry in right-handed Y-up world coordinates. Upstream normalizes the mesh and
  restores it with a floating-point round trip. Upstream also saves 8-bit JPEG textures, which the port does not
  use: the base colour and metallic-roughness textures stay float and are exported as PNG in the GLB. Metallic and
  roughness are the red and green channels of the generated metallic-roughness views, as in upstream's `save_mesh`.
- **Mesh preparation.** Upstream loads and re-saves the mesh with pymeshlab and simplifies it with trimesh. The port
  merges duplicate vertices and simplifies through `TrimeshBackend` (fast-simplification). Different
  simplifications give different meshes and atlases.
- **Released behaviour kept on purpose.** These follow the released inference pipeline, not its training code:
  - The per-view guidance ramp in upstream's denoising loop is never reached, because the released pipeline does not
    pass camera azimuths, so every view uses the same guidance.
  - The released pipeline modifies its float16 position maps in place while voxelizing the finest grid, so the
    coarser grids average whole cells with a zero background. `position_voxel_indices` reproduces this.
  - `MeshRender` recomputes the silhouette and depth-edge margin at the 1024-pixel view-selection resolution and
    keeps it for the bake, so the bake distrusts `render_size // 512` pixels: 4 at 2048, not 8.
  - Real-ESRGAN receives RGB images through its BGR interface, so the network sees swapped channels; the port swaps
    them the same way.
- **Upscaled views.** Upstream resizes only the first two upscaled views to the render size, because it loops over
  the number of keys in a dictionary. With the default 512-pixel views every view is already 2048, so this changes
  nothing. With larger views, the port resizes every view.
- **Randomness.** The initial noise is drawn first from `generator`, then the VAE posterior samples. A fresh seeded
  generator therefore reproduces upstream's initial latents; upstream samples the posteriors from the global RNG.

## Paint parity

`tests/families/hunyuan3d/test_paint_reference_parity.py` loads the pinned checkout. The checkout's CUDA rasterizer
kernel is replaced by the package rasterizer, and its material attention runs on the CPU with its hard-coded
`.to("cuda:0")` removed. On CPU in float32:

- A tiny random-weight `UNet2p5DConditionModel` (both UNets and every added attention) and `Hunyuan3DPaintUNet`
  give bit-identical outputs.
- Voxel indices and 3D rotary embeddings are bit-identical.
- View selection picks the same nine views on a mesh that needs them all. Mesh normalization and cameras agree
  within 1e-7.
- `back_project` and `fast_bake_texture` are compared in two modes:
  - the released `back_sample` mode with flat normals, whose textures, weights, and trust mask are bit-identical;
  - the `linear` mode with smooth normals that Step1X-3D uses, which agrees within 1e-6 (upstream computes vertex
    normals in float64).

The released checkpoint was compared with upstream Paint on one A100 in float16, on the same simplified mesh and
atlas (see `docs/compatibility.md`):

- Upstream's own pipeline has a CUDA rasterizer, the C++ inpainter, and Real-ESRGAN.
- The generated views match upstream at 61 dB PSNR.
- Texels a view covers match within 0.6/255 on average. The inpainted texels are where the two inpainting methods
  differ.
