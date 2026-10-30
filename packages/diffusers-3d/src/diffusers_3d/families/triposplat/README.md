# TripoSplat image-to-Gaussians

VAST-AI-Research TripoSplat (SIGGRAPH 2026), ported from upstream revision `d8db9e018b413dd9c4a9fe22463781bf98e8e68d`
with weights from `VAST-AI/TripoSplat` at `56a96e603204ec410c4da60c13ea4fa09a2169a9`. A rectified-flow transformer
samples 8192 latent tokens from DINOv3 and Flux.2 VAE image tokens. An octree decoder then places a chosen number of
3D Gaussians. MIT license.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("path/to/converted").to("cuda")
splat = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(42)).objects[0]
```

- Inputs without usable alpha are cut out by default with upstream's BiRefNet network and weights
  (`ZhengPeng7/BiRefNet`, MIT), after the shorter side is resized to 1024 px in the image's own PIL mode, as upstream
  does (palettes resize with NEAREST; LA and transparent palette alpha is ignored). With
  `pipeline.disable_background_removal()` they must carry RGBA alpha or a `mask`.
- Defaults match upstream: 20 steps, guidance 3.0, shift 3.0, 262144 Gaussians, a 1024-pixel canvas, and a 1-pixel
  alpha erosion.
- Without `dtype`, `from_pretrained` loads each component in upstream's precision: DINOv3 and the VAE in bfloat16,
  the transformer and decoder in float16. Pass `dtype` (for example `dtype=torch.float32` on CPU) to load every
  component in one precision instead.
- `num_gaussians` must lie in 32768 to 262144, as upstream asserts, and is rounded to a multiple of 32.
- Guidance runs the conditional and unconditional passes one after the other, as upstream does.
  `batch_guidance=True` runs them as one batch, which is faster but not bit-identical.
- Each image gives one `GaussianSplatAsset`. It uses SH degree 0, sits inside `[-0.5, 0.5]^3`, and is in the
  right-handed Z-up frame (TRELLIS's). Upstream's `.ply` export rotates to `(x, -z, y)`.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.DINOv3ViTModel` | ViT-H/16+; all-zero tokens are the unconditional input |
| `vae` | `TripoSplatImageVAE` | Flux.2 VAE encoder on the stock Diffusers `Encoder`, patchified and batch-norm standardized |
| `transformer` | `TripoSplatTransformerModel` | 2+2 refiner and 24 joint blocks, width 1024, learned 3D rotary positions |
| `decoder` | `TripoSplatGaussianDecoder` | octree point sampler (8 levels) and a 16-block Gaussian decoder, 32 Gaussians per point |
| `scheduler` | `diffusers.FlowMatchEulerDiscreteScheduler` | stock, `shift=3.0`, given upstream's uniform sigmas |

The flow state packs the latent tokens and the camera token into one tensor, so the stock scheduler steps both at
once.

## Differences from upstream

- The stock scheduler steps float32 sigmas where upstream steps float64 ones. One Euler step is bit-identical to
  upstream; after three steps the latents of a random tiny model differ by about 6e-5.
- The VAE posterior, the initial noise, and the octree resampling all draw from `generator`, as upstream's first two
  do. Upstream's octree resampling uses the global torch RNG, which `run` never seeds, so its final points are not
  reproducible; with the same stream (`torch.manual_seed(s)` against `torch.Generator().manual_seed(s)`) the octree
  points are identical.
- Not ported: the TripoSplat VAE encoder, `.ply`/`.splat` writers, and training.

## Conversion

```bash
diffusers-3d-convert-triposplat VAST-AI/TripoSplat output/
```

The source is a Hub id or a local snapshot; `--revision` defaults to the pinned weights. Only the two files that
inference needs are downloaded. The flow transformer and decoder keep their released float16. The release also
bundles DINOv3 and the Flux.2 VAE; those copies equal `facebook/dinov3-vith16plus-pretrain-lvd1689m` (cast to
bfloat16) and FLUX.2-dev's `vae/` tensor for tensor, so they are gated third-party weights and are never read or
written. `from_pretrained` loads them from those repositories at pinned revisions (accept both licenses on the Hub
and log in first), with TripoSplat's own VAE settings (batch-norm epsilon `1e-5`, no log-variance clamp).

## Verification

Tiny-weight CPU tests check that DINOv3, the Flux.2 VAE encoder, the flow transformer, the octree sampler, and the
Gaussian decoder are bit-identical to the pinned upstream `model.py`, including a camera MLP sized by a
non-default `num_refiner_layers`. They also check the Gaussian activations, the Euler schedule, and the guided sampler
against upstream's `FlowEulerCfgSampler`. Set `DIFFUSERS_3D_TRIPOSPLAT_REFERENCE_ROOT`
to the upstream checkout to run them.

A real-weight CPU float32 run (2 steps, 32768 Gaussians, seed 42, `building_stone_house.webp`) against upstream's
stage functions gave identical preprocessed images and image tokens, and the same octree points. The final latents
differed by at most 2.4e-4 (max |latent| 6.0), from the batched guidance passes. Gaussian means differed by at most
1.4e-5.

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
