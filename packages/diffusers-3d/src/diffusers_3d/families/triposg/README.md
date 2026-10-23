# TripoSG image-to-shape

VAST-AI-Research TripoSG, ported from upstream revision `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` with weights from
`VAST-AI/TripoSG` at `2c1c516d22d58db486a058d98d31bb6177344e06`. A rectified-flow transformer samples 2048 shape
tokens, the VAE decoder evaluates a signed-distance field, and marching cubes extracts an untextured mesh. MIT license;
the transformer derives from Tencent HunyuanDiT, whose license excludes the EU, UK, and South Korea.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/TripoSG-diffusers-3d").to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

- Images need RGBA, or RGB with a `mask`. At least 1% of the alpha must be clear foreground and 1% clear background,
  as upstream checks. Other inputs raise.
- Defaults: 50 steps, guidance 7.0, 2048 tokens, a `2**8` grid refined to `2**9` near the surface.
- Meshes are returned on the CPU in the right-handed Y-up frame. Marching cubes needs scikit-image.
- Constructing the pipeline warns about the HunyuanDiT license.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `feature_extractor` | `transformers.BitImageProcessorPil` | resize 256, centre crop 224, ImageNet normalization |
| `image_encoder` | `transformers.Dinov2Model` | DINOv2-L/14; all-zero tokens are the unconditional input |
| `scheduler` | `diffusers.FlowMatchEulerDiscreteScheduler` | stock; the pipeline passes `-v` to match upstream's update exactly |
| `transformer` | `TripoSGTransformerModel` | 21 blocks, width 2048, on Diffusers blocks; LoRA-loadable |
| `vae` | `TripoSGVAEDecoder` | decoder half of the shape VAE |

The attention splits heads before the q/k/v projections. This matches the released weights, which therefore load
unchanged.

## Differences from upstream

- Surface extraction uses upstream's MIT `use_flash_decoder=False` path: a dense grid, one refinement near the
  surface, and scikit-image marching cubes. Upstream's default FlashVDM decoder is under the Tencent Hunyuan FlashVDM
  license (excludes the EU, UK, and South Korea), and its `diso` extractor is CC BY-NC 4.0 and CUDA-only.
- That path's quirks are kept on purpose: inclusive dense bounds, refined spacing of `bounds / 256`, and a single
  symmetric `bounds` value.
- No background removal. Upstream runs `briaai/RMBG-1.4`, which has a non-commercial license.
- Images larger than 2000 px are not downsized first, as upstream does.
- `BitImageProcessorPil` reproduces the PIL resize of transformers 4.45, used for the release. It has not been
  compared numerically against 4.45.
- Not ported: the VAE encoder (needs `torch_cluster`), TripoSG-scribble, `pymeshlab` face reduction, and training.

## Conversion

```bash
diffusers-3d-convert-triposg VAST-AI/TripoSG output/
```

The source is a Hub id or a local snapshot; `--revision` defaults to the pinned weights. The converter keeps the
transformer tensors unchanged, drops the VAE encoder, and re-saves DINOv2 as a stock Transformers model.

## Verification

Tiny-weight CPU tests check that the transformer, VAE decoder, scheduler loop, hierarchical extraction, and a 4-step
guided pipeline are bit-identical to the pinned upstream code. Set `DIFFUSERS_3D_TRIPOSG_REFERENCE_ROOT` to the
upstream checkout to run them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
