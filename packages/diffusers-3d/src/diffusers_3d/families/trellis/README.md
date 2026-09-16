# TRELLIS image-to-3D and text-to-3D

This family integrates the MIT-licensed Microsoft TRELLIS implementation at
revision `442aa1e1afb9014e80681d3bf604e8d728a86ee7`. The sparse-structure
stage, the SLAT flow, the Gaussian decoder, and the FlexiCubes mesh decoder
run in plain PyTorch on any device; rendering the resulting splats delegates
to the optional gsplat backend. `TrellisImageTo3DPipeline` conditions on
DINOv2 image tokens and `TrellisTextTo3DPipeline` on CLIP text tokens; both
share the same two flow stages and decoders.

## Reviewed sparse-structure path

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
object-3D metadata. It converts the sparse-structure components, the SLAT
flow model, and the Gaussian and mesh decoders. For image pipelines
`--conditioner-path` takes either a saved `TrellisDinov2Conditioner` folder or
the Transformers `dinov2_with_registers` checkpoint of the released
`dinov2_vitl14_reg` (`facebook/dinov2-with-registers-large`); the register
tokens fold into the conditioner and the token outputs match exactly. For text
pipelines it takes the released `openai/clip-vit-large-patch14` folder, whose
text tower and tokenizer become a `TrellisClipTextConditioner`. Conversion
from the original Torch Hub state layout is not claimed.

## SLAT flow, Gaussian decoder, and mesh decoder

`TrellisSparseTensor` is an immutable package bridge over `[batch, x, y, z]`
coordinates and features. It losslessly round-trips `SparseVoxelAsset`
metadata and supports released channelwise SLAT normalization.

`TrellisSLatFlowModel` is the released structured-latent flow: submanifold
sparse-convolution residual blocks that downsample into the transformer and
upsample back out with skip connections, around a dense-attention core with
RoPE. `TrellisSLatGaussianDecoder` is the released decoder with shifted-window
("swin") sparse attention and the grouped-by-attribute Gaussian parameter
layout. Sparse convolution, pooling, and window partitioning come from
[`sparse_ops.py`](sparse_ops.py), which replaces `spconv` and `xformers` with
plain PyTorch. The pooling reproduces upstream's `scatter_reduce(...,
include_self=True)` mean, because the released weights were trained with it.

`TrellisSLatMeshDecoder` is the released mesh decoder: the same swin torso,
two sparse subdivide blocks (64³ to 256³), and per-cube FlexiCubes features
(`sdf`, `deform`, `weights`, and six colour channels). Iso-surface extraction
is a pure-PyTorch port of the Apache-2.0 FlexiCubes fork TRELLIS pins
([`flexicubes.py`](flexicubes.py)); it builds the grid sparsely around the
active voxels instead of materializing the dense 256³ grid and returns a Z-up
`MeshAsset` with vertex `colors` and the predicted normal map in
`extras["normal_map"]`. `formats=("mesh",)` selects it in both pipelines.

`TrellisSLatRadianceFieldDecoder` is the released radiance-field decoder:
the same torso followed by a projection to the `Strivec` channels (rank-16
tri-vectors with 8 samples per axis, per-component density and DC colour).
It returns a `RadianceFieldAsset`, which stores exactly those channels plus
the grid transform. Rendering is not a model concern:
`diffusers_3d.backends.radiance_field.render_radiance_field` is an
independent pure-PyTorch volume renderer written from the representation's
definition (two samples per trivec cell, `softplus` density, `sigmoid`
colour, front-to-back compositing). It does not derive from the restricted
`diffoctreerast` rasterizer and claims no pixel parity with it.
`formats=("radiance_field",)` selects the decoder in both pipelines.

Tiny outputs of all four models match the pinned upstream code with `spconv`
and `xformers` shimmed to dense PyTorch (the mesh decoder additionally against
the pinned FlexiCubes submodule, up to vertex order; the radiance-field
decoder with the upstream `Strivec` built on CPU), and released state-dict
layouts are checked against the published safetensors headers.

## Textured GLB

Upstream's `to_glb` textures the mesh from renders of the Gaussians rather
than from the mesh decoder's vertex colours. `TrellisGlbPostprocessFacade.
to_textured_mesh(mesh, gaussians)` follows that recipe with permissive
backends: CuMesh repairs and simplifies the mesh to 5% of its faces, xatlas
unwraps it, gsplat renders the splats from 100 Hammersley-distributed views,
and `backends/texture_baking.py` projects every texel onto those renders and
averages the views that see it (depth- and alpha-tested against the render).
The result is a `MeshAsset` with `uvs` and a textured `PBRMaterial` that
`TrimeshBackend.export_mesh` writes as a GLB with a base-colour texture. The
upstream hole filling and the optimisation-based baking mode are not
reproduced.

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
- The FlexiCubes fork TRELLIS pins (`MaxtirError/FlexiCubes`, revision
  `815e075a`) is Apache-2.0; its lookup tables and extraction logic are ported
  into this family with attribution. All restricted renderer source is
  excluded.
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

`TrellisSLatFlowRecipe` is registered with the same objective over
precomputed SLAT latents (`TrellisSLatExample`), training the SLAT flow model
alone.

The conditioner and decoders remain frozen. LoRA is not registered because the
released project provides no LoRA target evidence. Tests cover the exact
objective, frozen components, a full optimizer step, and checkpoint
restoration for both recipes. Training examples accept unit-range typed image conditions.
Recipe collation separately follows the pinned dataset transform exactly once:
the bbox includes every nonzero alpha pixel, applies 1.2 to the floating
half-size before integer truncation, resizes RGBA with LANCZOS, and multiplies
the resized RGB and alpha tensors. Separate masks participate in alpha.

## Explicit limitations

- No production-resolution GPU or end-to-end two-stage parity run is part of
  this package's test matrix; the released checkpoints were run by hand on an
  A100 (see `docs/compatibility.md`) and render quality is not claimed.
- Rendering quality, texture quality, pixel parity of the radiance-field
  renderer with `diffoctreerast`, and background removal are not claimed.
- CI and conversion tests are offline CPU tests and download no model weights.

The converted releases are published as
`suvadityamuk/TRELLIS-image-large-diffusers-3d` and
`suvadityamuk/TRELLIS-text-large-diffusers-3d`, conditioners included, and
load with the auto-loaders directly. To convert a local release yourself:

```bash
diffusers-3d-convert-trellis source/ output/ \
  --conditioner-path /local/path/to/trellis-dinov2-conditioner
```

The converter writes the conditioner, sparse-structure flow and decoder, SLAT
flow, and the Gaussian, mesh, and radiance-field decoders.
