# Step1X-3D

This family integrates the geometry stage of the Apache-2.0 StepFun Step1X-3D
implementation at revision `cb5ac944709c6c913109070c7b90c3447f57f3d4`, with
weights from `stepfun-ai/Step1X-3D` (`Step1X-3D-Geometry-1300m`). An image
conditions a rectified-flow transformer over the shape VAE's latent set, the
VAE decoder evaluates occupancy on a grid, and marching cubes extracts a
watertight mesh. Everything runs in plain PyTorch on any device; mesh
extraction needs the portable `scikit-image` backend
(`pip install "diffusers-3d[portable]"`). The [texture stage](#texture-stage)
paints a mesh from a reference image.

## Usage

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Step1X-3D-Geometry-1300m-diffusers-3d").to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

`rgba` is a `(4, H, W)` tensor in `[0, 1]` whose alpha channel masks the object. To convert the release yourself,
see "Conversion and parity" below.

## Components

- `image_encoder` is the stock Transformers `Dinov2WithRegistersModel`
  (ViT-L/14, 4 registers, 518 px). The pipeline conditions on its final
  layer-normed class and patch tokens with the registers dropped
  (1 + 37² = 1370 x 1024); the null condition is all zeros, as released.
  TRELLIS uses the same encoder differently: it takes pre-norm tokens with an
  unparameterized norm and keeps the registers.
- `Step1X3DTransformer1DModel` is the released Flux-style transformer (12
  dual-stream + 24 single-stream blocks, width 1280, no positional embedding,
  condition tokens first). Dual-stream blocks are the stock Diffusers
  `JointTransformerBlock` with RMS Q/K norms; single-stream blocks and the
  joint-attention processor are package code because this Diffusers build
  ships no Flux model. `time_proj`/`time_embedding` are the stock `Timesteps`
  and `TimestepEmbedding`. The label-conditioned checkpoint's transformer
  (`Step1X-3D-Geometry-Label-1300m`) also loads into this class.
- `Step1X3DShapeDecoder` is the decoder half of the shape VAE: `post_kl`,
  16 stock `BasicTransformerBlock`s, and one cross-attention query block
  (stock `Attention` and `FeedForward`) over a Fourier embedding of the query
  point. The point encoder needs farthest-point sampling (`torch_cluster`) and
  is unused at inference, so conversion drops its 124 tensors.
- `Step1X3DFlowMatchEulerScheduler` is the stock
  `FlowMatchEulerDiscreteScheduler`, subclassed only so the exact-class
  registry can name it.
- `Step1X3DImageTo3DPipeline` follows the released `inference.py`: 50 steps,
  guidance 7.5 with `[negative, conditional]` ordering, the first step's
  velocity zeroed, bounds 1.05, octree resolution 384, level 0.

## Differences from the released pipeline

- **Occupancy grid.** Upstream's hierarchical volume decoder (from Hunyuan3D,
  under a non-commercial license header, so not ported) evaluates a 97³ grid
  and then only a band near the surface at 193³ and 385³, leaving other cells
  NaN; scikit-image then emits NaN vertices there. This family evaluates the
  full dense grid. Values agree with upstream wherever it evaluates (within
  float32 rounding of its `index * step` fine-level coordinates). Parts that
  the coarse pass misses are recovered rather than dropped. The cost is all
  385³ (57M) decoder queries at the default resolution, instead of a
  near-surface band.
- **Half precision.** Query points and their Fourier embedding stay in
  float32 even when the decoder runs in bfloat16 or float16. Half-precision
  coordinates cannot resolve a 384³ grid. Upstream instead switches the VAE to
  float16 for bfloat16 latents, so low-precision runs are more accurate here,
  not identical.
- **Marching cubes.** It uses the same Lewiner extraction through
  `ScikitImageBackend`, with upstream's winding and its
  `index / (resolution + 1) * 2 * bounds - bounds` vertex mapping. Vertices
  and faces match upstream bit for bit on the same grid.
- **Post-processing.** Upstream removes floaters, decimates to 200k faces with
  pymeshlab (GPL), and smooth-shades with trimesh. Meshes here are returned
  raw, as upstream does for `output_type="np"`.
- **Image framing.** It reproduces the released crop to the alpha bounding
  box, white compositing, 0.95 fit, square padding, and Pillow bicubic resize.
  The pipeline never runs background removal. Opaque or RGB images are framed
  as a full-frame crop; upstream would run `rembg` on them first.
- **fp16 clipping.** The stock `JointTransformerBlock` does not clip the
  context stream in float16 as upstream's Flux block does. The released
  checkpoints run in float32.

## Not ported

- **The label-conditioned variant.** Its symmetry/edge-type labels need
  `Dinov2CLIPEncoder` (DINOv2 and CLIP tokens concatenated) and
  `LabelEncoder`. The converter rejects it; its transformer already loads into
  `Step1X3DTransformer1DModel`.
- **The point encoder and DMC extraction.** The point encoder is training-only.
  DMC extraction would need `diso`, a CUDA build.

## Conversion and parity

`diffusers-3d-convert-step1x3d <Step1X-3D-Geometry-1300m> <output>` rejects
missing or unexpected weights in every component. It splits the VAE's per-head fused
`c_qkv`/`c_kv` projections into `to_q`/`to_k`/`to_v`, renames the timestep
MLP to `time_embedding`, and writes the Diffusers layout,
`object3d_model_index.json`, and a `step1x3d_conversion.json` report.
Converted key sets and shapes match the published safetensors headers for
the transformer (732 tensors), VAE decoder (231 of 322), and visual encoder
(440).

Reference parity runs tiny random weights through the pinned upstream code on
CPU. The following are bit for bit:

- the transformer, VAE tokens, and point queries
- the dense grid against upstream's vanilla decoder
- marching-cubes output
- image-encoder tokens and preprocessing
- a full 4-step guided image-to-mesh run, covering latents, vertices, and
  faces

By hand on an A100 (2026-10-13), the published
`suvadityamuk/Step1X-3D-Geometry-1300m-diffusers-3d` turned the TRELLIS
`T.png` example into a 0.93M-face mesh in 95 s (float32, 9.4 GB peak).

## Texture stage

`Step1X3DMeshTexturePipeline` takes a `MeshAsset` and an `ImageCondition` and
returns the mesh UV-unwrapped with a base-colour texture, following the
released `Step1X-3D-Texture` pipeline (MV-Adapter image-guided SDXL):

```python
from diffusers_3d import AutoPipelineFor3D

texture = AutoPipelineFor3D.from_pretrained("suvadityamuk/Step1X-3D-Texture-diffusers-3d", dtype=torch.float16).to("cuda")
textured = texture(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

Upstream decimates the geometry stage's mesh to 200k faces before texturing; do the same with
`TrimeshBackend().process_geometry(mesh, operation="simplify_quadric_decimation", parameters={"face_count": 200_000})`.
The pipeline runs these steps:

1. The mesh is cleaned with trimesh (optional), unwrapped with xatlas, and
   scaled to `max |x| = 0.5` in a Z-up frame.
2. Six orthographic renders (front, right, back, left, top, bottom) of
   position and normal condition a T2I-Adapter.
3. The reference image is cropped to its alpha, fitted to 90% of the frame,
   and composited on grey. One UNet pass over its latent caches every
   self-attention input.
4. The SDXL UNet denoises the six views with the MV-Adapter processors
   installed for the call: attention along image rows across the front,
   right, back, and left views, along columns across the front, back, top,
   and bottom views, and to the cached reference.
5. For the first half of the schedule the six latent views step together
   through a shared latent texture (SyncMVD): the predicted clean views are
   baked into the texture, the DDPM posterior is taken on the texture, and
   the views are rendered back over solid-colour backgrounds.
6. The decoded views are upscaled to the texture size with Lanczos and
   baked onto the mesh UVs with the shared `multiview_baking` backend, as
   upstream's `MeshRender` does. Texels no view covers are inpainted.

Components:

- `vae`, `text_encoder`, `text_encoder_2`, `tokenizer`, `tokenizer_2`, and
  `unet` are the stock SDXL components, `adapter` is the stock `T2IAdapter`
  (`full_adapter_xl`, six input channels), and `scheduler` is the stock
  `DDPMScheduler` with the released shifted-SNR betas (log-SNR moving towards
  SNR / 8) stored as `trained_betas`.
- `Step1X3DMultiViewAdapter` (`mv_adapter`) holds one
  `Step1X3DMultiViewAttnProcessor` per UNet self-attention layer at that
  layer's path, so the released tensors load without renaming.

Differences from the released pipeline:

- **Rendering.** The package rasterizer replaces nvdiffrast (geometry
  renders) and pytorch3d (texture synchronisation). Baking averages pixels
  into texels directly instead of back-propagating through a render. Empty
  texels take the value of a nearest filled texel found by jump flooding,
  instead of the CuPy Voronoi kernel, whose source has no license.
- **UVs.** One xatlas atlas serves the latent texture and the output;
  upstream unwraps three times. The synchronisation's cosine weights use
  normals of the welded mesh, and its top and bottom cameras sit at the
  geometry renders' ±89.99° instead of ±90°.
- **Randomness and precision.** All noise, background colours, and VAE
  samples come from the pipeline generator; upstream also draws from the
  global torch and Python generators. The VAE runs in its own dtype.
- **Inputs and outputs.** No background removal runs: the reference image
  needs an alpha channel or a mask, and opaque images are framed whole.
  Upstream decimates the mesh with pymeshlab. The output keeps the input
  mesh's scale and frame.
- **Final bake.** The views are upscaled to `texture_size` for baking;
  upstream sets that resolution separately (both are 2048). Uncovered
  texels are filled from their neighbours instead of with `cv2.inpaint`.

`diffusers-3d-convert-step1x3d-texture <Step1X-3D-Texture> <output>` loads
the SDXL base and the fp16-fix VAE at pinned revisions, loads the
T2I-Adapter (38 tensors) and MV-Adapter (700 tensors) from
`step1x-3d-ig2v.safetensors`, rejecting unexpected keys, and precomputes the
betas. The output contains the SDXL base weights, which are under the
CreativeML Open RAIL++-M License and its use restrictions, so the converter
copies that license next to them as `LICENSE-SDXL.md`. The VAE is MIT.

Reference parity on CPU, with the package rasterizer and texel fill on both
sides where upstream needs CUDA: the attention processors on a tiny UNet,
the scheduler betas, and the reference framing are bit for bit; the
geometry renders agree to 1.2e-6, the synchronised texture step to 2.4e-7,
and the final bake (`back_project` and `fast_bake_texture`) to 3e-7 on
covered texels.

See `LICENSE-APACHE-2.0`, `LICENSE-MIT`, and `NOTICE` for source and license
terms.
