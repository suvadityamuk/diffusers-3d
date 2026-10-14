# TripoSG image-to-shape

This family integrates VAST-AI-Research TripoSG at revision
`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` and the `VAST-AI/TripoSG`
weights at revision `2c1c516d22d58db486a058d98d31bb6177344e06`. It turns one
image into an untextured mesh: a 1.5B-parameter rectified-flow transformer
samples 2048 shape tokens, the shape-VAE decoder evaluates a signed-distance
field on a hierarchical grid, and marching cubes extracts the surface. Every
step runs in plain PyTorch on any device, plus scikit-image for marching cubes.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/TripoSG-diffusers-3d").to("cuda", torch.float16)
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(42)).objects[0]
```

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `feature_extractor` | `transformers.BitImageProcessorPil` | shortest edge 256 (PIL bicubic), centre crop 224, ImageNet normalization |
| `image_encoder` | `transformers.Dinov2Model` | DINOv2-L/14; its `last_hidden_state` (257 tokens) conditions the flow |
| `scheduler` | `diffusers.FlowMatchEulerDiscreteScheduler` | see Sampling |
| `transformer` | `TripoSGTransformerModel` | 21 blocks, width 2048, timestep token, long skips, Q/K RMS norm; LoRA-loadable |
| `vae` | `TripoSGVAEDecoder` | decoder half of the shape VAE |

The two TripoSG models are written on Diffusers blocks (`FeedForward`,
`FP32LayerNorm`, `RMSNorm`, `Timesteps`, `TimestepEmbedding`, and an
`AttentionModuleMixin` attention that calls `dispatch_attention_fn`). The one
custom piece is the attention processor: the released checkpoints split heads
before query/key/value, so head `h` reads all three from the `h`-th chunk of
the concatenated projections. Keeping that layout lets the released tensors load
unchanged, so `diffusers-3d-convert-triposg` only drops the VAE encoder and
renames keys.

## Sampling

The defaults match upstream: 50 steps, guidance 7.0 with all-zero image tokens
as the unconditional input, 2048 tokens, sigmas `1 - i/N` followed by 0, and
the Euler update `x + (sigma - sigma_next) * v`. The stock scheduler adds
`(sigma_next - sigma) * v`, so the pipeline passes it `-v`. Negation is exact,
so the timesteps and every step match upstream bit for bit.

## Input

Pass RGBA, or RGB with a `mask`. The pipeline reproduces upstream's alpha
branch of `load_image`:

1. Take the bounding box of the uint8 alpha that is greater than zero.
2. Composite the image onto white.
3. Crop to the box and pad it square with 10% white margins.
4. Quantize to uint8.

Upstream calls `briaai/RMBG-1.4` (non-commercial license) whenever alpha is
missing or fails its histogram check: at least 1% of pixels must be at most 12
and at least 1% at least 244. This package never runs a background remover, so
those inputs raise. Upstream also first downsizes images larger than 2000 px
with OpenCV `INTER_AREA`; that step is not applied.

The release was made with transformers 4.45, whose `BitImageProcessor` resized
with PIL. In transformers 5 that class uses torchvision, so the pipeline stores
`BitImageProcessorPil`. That is the same PIL code path, but it has not been
compared numerically against 4.45.

## Surface extraction: how this differs from upstream

Upstream's `TripoSGPipeline` defaults to `use_flash_decoder=True`. That path
uses FlashVDM's coarse-to-fine decoder (a 63-cell grid doubled three times to
505 samples per axis) and `diso` CUDA dual marching cubes. It is not ported
because:

- that code is distributed under the Tencent Hunyuan FlashVDM Community
  License, whose territory excludes the EU, UK, and South Korea;
- `diso` is licensed CC BY-NC 4.0 and needs CUDA.

This family ports upstream's own MIT-licensed `use_flash_decoder=False` path
(`hierarchical_extract_geometry`):

- a dense `2**8` float16 grid over `[-1.005, 1.005]^3`;
- one refinement to `2**9`, which re-decodes only the children of cells whose
  sigmoid band value is not saturated, dilated by a k=5 max-pool;
- scikit-image Lewiner marching cubes on the result.

Grid, vertices, faces, and winding all match that path bit for bit. That
includes upstream's `gradient_direction="descent"`, which points faces outward
because the released decoder is negative inside the shape: the TRELLIS `T.png`
example gives a watertight mesh with positive signed volume. Compared with the
default flash path, meshes are primal marching cubes (vertices on grid
edges) instead of dual (one vertex per cell), sampled at 512 instead of 504
cells. Upstream's quirks are kept on purpose:

- dense samples include both bounds, while refined cells use a spacing of
  `bounds / 256`;
- vertices are scaled by `1 / 2**depth`;
- `bounds` is a single half-extent, because upstream's refinement assumes a
  symmetric cube.

Extraction runs one shape at a time, as upstream does, and returns
CPU `MeshAsset`s in the right-handed Y-up frame.

## Conversion

`diffusers-3d-convert-triposg VAST-AI/TripoSG out/` downloads the pinned
snapshot (or reads a local one) and writes a Diffusers folder plus the
`object3d_model_index.json` sidecar:

- transformer tensors are unchanged (554 keys);
- the VAE keeps its 231 decoder tensors, drops the 125 `encoder.*` and
  `quant.*` tensors, and strips the `decoder.` prefix;
- DINOv2 and its preprocessor are re-saved as stock Transformers components;
- `RectifiedFlowScheduler` (shift 1) becomes `FlowMatchEulerDiscreteScheduler`.

These layouts match the published safetensors headers exactly.

## Not ported

- the VAE encoder, which needs `torch_cluster` farthest-point sampling and is not
  used for image-to-shape;
- TripoSG-scribble;
- upstream's `pymeshlab` face reduction;
- training code.

## Verification

With tiny random weights on CPU, the transformer, the VAE decoder, the
scheduler loop, the hierarchical extraction of an analytic field
(128 → 256 cells), and the whole pipeline (4 steps with CFG, refined grid)
are bitwise equal to the pinned upstream code. No production-checkpoint GPU run
is part of the CPU test matrix. By hand on an A100 (2026-10-13), the published
checkpoint `suvadityamuk/TripoSG-diffusers-3d` turned the TRELLIS `T.png`
example into a watertight 1.21M-face mesh in 83 s (float32, 10.1 GB peak).

See `LICENSE-MIT` and `NOTICE`. The upstream transformer file says it is based
on Tencent HunyuanDiT, whose Tencent Hunyuan Community License does not apply in
the EU, UK, or South Korea; `NOTICE` carries that attribution, and constructing
the pipeline warns about it.
