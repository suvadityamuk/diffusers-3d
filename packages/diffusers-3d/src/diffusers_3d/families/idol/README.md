# IDOL image-to-Gaussians

Speridlabs IDOL (2026), ported from upstream revision `4702a7f00850e34b6e83a2170cc123c3e9570744` with weights from
`speridlabs/idol` at `2731c2291fc65a797c5cb627a5e5526fc1490902`. One image in, 32,768 to 1,048,576 3D Gaussians out.
Code and weights are Apache-2.0 and not gated.

IDOL keeps TripoSplat's architecture. Its released denoiser and decoders load without changes into this package's
`TripoSplatTransformerModel` and `TripoSplatGaussianDecoder`, so this family adds only a pipeline and a converter.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("path/to/converted").to("cuda")
splat = pipeline(ImageCondition(image=rgba), mode="real", generator=torch.Generator("cuda").manual_seed(42)).objects[0]
splat.save("object.ply")
```

- Inputs without usable alpha are cut out by default with upstream's BiRefNet network and weights
  (`ZhengPeng7/BiRefNet`, MIT), after the shorter side is resized to 1024 px in the image's own PIL mode, as upstream
  does (palettes resize with NEAREST; LA and transparent palette alpha is ignored). With
  `pipeline.disable_background_removal()` they must carry RGBA alpha or a `mask`.
- `mode="synthetic"` (the default, for renders) crops 1.2 times the object; `mode="real"` (photos) crops 1.7 times.
  `crop_margin` overrides both.
- Defaults match upstream: 50 steps, shift 3.0, 262144 Gaussians, and a 1024-pixel canvas. Guidance is
  `guidance_scale=4.0`: upstream's `cfg_strength=3.0` in the Diffusers convention `s * cond - (s - 1) * uncond`.
- Without `dtype`, `from_pretrained` loads each component in upstream's precision: DINOv3 and the VAE in bfloat16, the
  transformer and decoder in float32 with float16 linear layers inside their blocks. Pass `dtype` to load every
  component in one precision instead.
- `num_gaussians` is rounded down to a multiple of 32, but to at least 32, as upstream does.
- Each image gives one `GaussianSplatAsset`. It uses SH degree 0, sits inside `[-0.5, 0.5]^3`, and is in the
  right-handed Z-up frame. `metadata["camera"]` holds the predicted input camera: yaw, pitch, radius, and FOV.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.DINOv3ViTModel` | ViT-H/16+ (`facebook/dinov3-vith16plus-pretrain-lvd1689m`, gated) |
| `vae` | `TripoSplatImageVAE` | encoder half of the FLUX.2-dev VAE (bfloat16, as upstream loads it), patchified and batch-norm standardized |
| `transformer` | `TripoSplatTransformerModel` | 2+2 refiner and 24 joint blocks, width 1024, learned 3D rotary positions |
| `decoder` | `TripoSplatGaussianDecoder` | octree point sampler (8 levels) and a 16-block Gaussian decoder, 32 Gaussians per point |
| `scheduler` | `diffusers.FlowMatchEulerDiscreteScheduler` | stock, `shift=3.0`, given upstream's uniform sigmas |

## Differences from upstream

- Classifier-free guidance runs both passes as one batch, as upstream's default GPU path (`optimize=True`) does.
  Upstream's eager path (`optimize=False`, and every CPU run) runs them one after the other; the two differ by float
  rounding (6.8e-4 on real weights after 2 steps). The GPU path's cuDNN attention and TF32 matmuls are not ported.
- The VAE posterior, the initial noise, and the octree sampler draw from `generator` in upstream's order. Upstream
  draws them from the global torch RNG after `torch.manual_seed(seed)`, so a CPU `torch.Generator().manual_seed(seed)`
  reproduces its CPU stream.
- `GaussianSplatAsset.save` writes upright Y-up `(x, z, -y)`. Upstream's `save_ply` writes `(x, -z, y)`, which is
  the same splat turned half a turn about x.
- Not ported: BiRefNet, the `torch.compile`/cuDNN fast path, the Gradio demo, and training.

## Conversion

```bash
diffusers-3d-convert-idol speridlabs/idol output/
```

The converter downloads `config.json`, `denoiser`, `decoder`, and `decoder_gs` (about 2.6 GB, float32). It maps
them through the TripoSplat key renames and merges the two decoders. It also checks that the stored latent positions
equal the Sobol points the transformer recomputes.

DINOv3 and the Flux.2 VAE are third-party and gated: the converter never downloads or writes them, and
`model_index.json` only names them. `from_pretrained` loads DINOv3 from `facebook/dinov3-vith16plus-pretrain-lvd1689m`
and the VAE encoder from FLUX.2-dev's `vae/` folder, each at a pinned revision; accept both licenses on the Hub and log
in first. `from_pretrained` applies upstream's precisions on load (see above); pass `dtype=torch.float32` on CPU, where
bfloat16 matmuls are slow.

## Verification

Tiny-weight CPU tests check the converted modules against the pinned upstream. Run them with
`DIFFUSERS_3D_IDOL_REFERENCE_ROOT` pointing at the upstream checkout. They cover:

- the denoiser, octree sampler, Gaussian decoder, and Gaussian activations, which are bit-identical both in float32
  and with upstream's own float16 block linears;
- DINOv3 and Flux.2 image tokens, also bit-identical;
- image preparation in both modes, which is pixel-identical;
- guided sampling, within 2e-4;
- the PLY frame.

A real-weight CPU float32 run used 2 steps, 32768 Gaussians, seed 42, and a rembg cut-out of
`demo/examples/fig-explorer.webp`. It was compared with upstream's stage functions on the same inputs:

- The preprocessed images and DINOv3 tokens were identical.
- Flux.2 tokens were identical when given upstream's bfloat16-rounded VAE input.
- Final latents equalled upstream's batched-guidance sampler exactly. They differed from its sequential one by at most
  6.8e-4 (max |latent| 7.3), the same gap as between upstream's two modes.
- On identical latents and seed, the octree points, means, quaternions, opacities, and SH were identical, and scales
  differed by at most 1.5e-8.

The DINOv3 weights for that run were TripoSplat's bundled bfloat16 copy, because the Meta repository is gated.

See [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and [NOTICE](NOTICE).
