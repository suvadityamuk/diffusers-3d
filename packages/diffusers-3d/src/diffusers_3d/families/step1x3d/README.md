# Step1X-3D image-to-geometry

This family integrates the geometry stage of the Apache-2.0 StepFun Step1X-3D
implementation at revision `cb5ac944709c6c913109070c7b90c3447f57f3d4`, with
weights from `stepfun-ai/Step1X-3D` (`Step1X-3D-Geometry-1300m`). An image
conditions a rectified-flow transformer over the shape VAE's latent set, the
VAE decoder evaluates occupancy on a grid, and marching cubes extracts a
watertight mesh. Everything runs in plain PyTorch on any device; mesh
extraction needs the portable `scikit-image` backend
(`pip install "diffusers-3d[portable]"`).

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
- **The texture stage.** Multi-view SDXL with geometry conditioning and
  baking is not ported yet.

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

See `LICENSE-APACHE-2.0` and `NOTICE` for source and license terms.
