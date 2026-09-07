# TRELLIS image-to-3D

This family integrates the MIT-licensed Microsoft TRELLIS implementation at
revision `442aa1e1afb9014e80681d3bf604e8d728a86ee7`. The reviewed package path
is the portable first-stage image-to-sparse-structure pipeline. Sparse SLAT and
representation decoders are separately marked experimental and capability
gated.

## Reviewed portable path

- `TrellisSparseStructureFlowModel` preserves released parameter names and
  dense transformer math while using Diffusers attention dispatch. Tiny CPU
  float32 forward output and selected backward gradients were measured against
  the pinned upstream implementation with identical weights and inputs.
- `TrellisSparseStructureDecoder` preserves the released dense Conv3D decoder
  and thresholds occupancy logits at zero. It returns package-owned
  `SparseVoxelAsset` objects with TRELLIS `[x, y, z]` coordinates, a native
  right-handed Z-up transform, and cell centers in `[-0.5, 0.5]`.
- `TrellisDinov2Conditioner` uses Transformers DINOv2 blocks with ImageNet
  normalization, class/register/patch token order, upstream `x_prenorm`
  semantics, an unparameterized final layer norm, and all-zero unconditional
  tokens.
- `TrellisFlowEulerScheduler` implements the released `t=1` noise to `t=0`
  data direction, `sigma_min=1e-5`, rational `rescale_t`, `t*1000` model
  timesteps, guidance intervals, and `(1+w)*cond-w*uncond`.
- `TrellisImageTo3DPipeline` always supports
  `formats=("sparse_structure",)` without CUDA extensions or renderer
  dependencies. Typed RGBA alpha and separate masks are quantized to uint8
  before the pinned `>0.8 * 255` foreground crop, then use the 1.2 recenter
  scale, Pillow LANCZOS RGBA resize, and alpha-premultiplication on black.
  Unmasked RGB is treated as an already
  background-removed full frame; the pipeline never invokes `rembg` silently.
  Defaults are 25 steps per stage, guidance 5, interval 0.5–1, and
  `rescale_t=3`.

The converter consumes local upstream component `.json`/`.safetensors` pairs
and `pipeline.json`, then writes ordinary Diffusers component folders and
object-3D metadata. Its default mode converts only the reviewed
sparse-structure components. A local, already compatible
`TrellisDinov2Conditioner` folder is required; conversion from the original
Torch Hub `dinov2_vitl14_reg` state layout is not claimed.

## Experimental sparse SLAT path

`TrellisSparseTensor` is an immutable package bridge over `[batch, x, y, z]`
coordinates and features. It losslessly round-trips `SparseVoxelAsset`
metadata and supports released channelwise SLAT normalization.

`TrellisSLatFlowModel` and `TrellisSLatGaussianDecoder` provide backend-free
full-attention tiny configurations for CPU tests. The SLAT flow objective is
implemented and tested with precomputed normalized sparse latents, but the
recipe is deliberately not registered as reviewed training support.

These classes do not provide official production checkpoint parity:

- Released SLAT flow checkpoints require sparse-convolution input/output
  blocks and CUDA `spconv`. Production construction fails before inference
  until those blocks have a separately tested implementation.
- Released Gaussian decoder checkpoints use sparse Swin/window attention.
  The tiny full-attention decoder tests canonical position, scale, quaternion,
  opacity-logit, spherical-harmonic, and raw-parameter mappings only.
- The TRELLIS sparse mesh-field network is not ported, so no mesh decoder,
  component, or pipeline format is shipped.
- `TrellisSLatRadianceFieldDecoder` is explicitly unsupported because the
  package has no native radiance-field `Object3D` type.

The converter accepts `--include-experimental` only for synthetic/tiny
SLAT-flow and Gaussian-decoder layouts. It records that production checkpoint
parity has not passed.

## Backend and license boundaries

- `GsplatBackend` is an explicit Apache-2.0 gsplat adapter for canonical
  `GaussianSplatAsset` and `CameraRig` values. It contains no Graphdeco code.
- `SpconvBackend` lazily imports a selected CUDA-matched Apache-2.0 build and
  preserves TRELLIS batch coordinates and sparse metadata.
- `KaolinFlexiCubesBackend` accepts only the Apache-2.0 `kaolin.ops`
  implementation. It rejects legacy `kaolin.non_commercial` implementations.
- nvdiffrast, diffoctreerast, and mip-splatting rasterization are restricted,
  research-only dependencies. Their facades perform side-effect-free status
  checks and require explicit license acknowledgement; they never import or
  select those renderers silently.
- The restricted TRELLIS modified FlexiCubes submodule and all restricted
  renderer source are excluded.
- `utils3d` is not used by this family. The common registry's optional
  compatibility entry accepts only the pinned EasternJournalist source and
  rejects an unverified colliding distribution.

See `LICENSE-MIT`, `NOTICE`, and `diffusers_3d_integration.json` for exact
source and backend declarations.

## Training evidence

`TrellisSparseStructureFlowRecipe` is registered for full-model training only
with precomputed dense sparse-structure latents:

`t = sigmoid(N(1, 1))`,
`x_t = (1-t)x0 + (sigma_min + (1-sigma_min)t)noise`,
target `(1-sigma_min)noise-x0`, model timestep `t*1000`, and conditioning
dropout probability 0.1.

The conditioner and decoder remain frozen. LoRA is not registered because the
released project provides no LoRA target evidence. Tests cover the exact
objective, frozen components, a full optimizer step, and checkpoint
restoration. Training examples accept unit-range typed image conditions.
Recipe collation separately follows the pinned dataset transform exactly once:
the bbox includes every nonzero alpha pixel, applies 1.2 to the floating
half-size before integer truncation, resizes RGBA with LANCZOS, and multiplies
the resized RGB and alpha tensors. Separate masks participate in alpha.

## Explicit limitations

- No official full checkpoint, production-resolution GPU, render-quality, or
  end-to-end two-stage parity run was performed.
- The reviewed AutoPipeline registration advertises sparse-structure output
  only. Tiny SLAT/Gaussian outputs are experimental and do not expand that
  reviewed contract.
- Production SLAT, production Gaussian decoding, mesh decoding, radiance-field
  decoding, rendering, texture quality, and background removal are not
  claimed.
- CI and conversion tests are offline CPU tests and download no model weights.

Convert a local pipeline:

```bash
diffusers-3d-convert-trellis source/ output/ \
  --conditioner-path /local/path/to/trellis-dinov2-conditioner
```

Add `--include-experimental` only when intentionally converting a compatible
backend-free tiny SLAT layout.
