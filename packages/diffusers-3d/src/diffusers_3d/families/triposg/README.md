# TripoSG image-to-shape

VAST-AI-Research TripoSG, ported from upstream revision `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` with weights from
`VAST-AI/TripoSG` at `2c1c516d22d58db486a058d98d31bb6177344e06`. A rectified-flow transformer samples 2048 shape
tokens, the VAE decoder evaluates a signed-distance field, and marching cubes extracts an untextured mesh. MIT license;
the transformer derives from Tencent HunyuanDiT, whose license excludes the EU, UK, and South Korea.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

# Upstream's inference script runs in float16.
pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/TripoSG-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

- As upstream, an alpha or `mask` is used when at least 1% of it is clear foreground and 1% clear background. Other
  inputs are shrunk to at most 2000 px and cut out with `briaai/RMBG-1.4` (non-commercial license; its weights load
  from BRIA's repository) and TripoSG's mask clean-up. `pipeline.disable_background_removal()` makes them raise.
- Defaults: 50 steps, guidance 7.0, 2048 tokens. On CUDA, the FlashVDM grid and DISO's `DiffDMC`, as upstream;
  elsewhere a `2**8` grid refined to `2**9` near the surface (see below).
- Meshes are returned on the CPU in the right-handed Y-up frame, with duplicate vertices merged by trimesh as
  upstream's output is. Marching cubes needs `pip install "diffusers-3d[portable]"`.
- Constructing the pipeline warns about the HunyuanDiT license.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `feature_extractor` | `transformers.BitImageProcessorPil` | `facebook/dinov2-large`'s preprocessor, loaded from there |
| `image_encoder` | `transformers.Dinov2Model` | `facebook/dinov2-large`, loaded from there; all-zero tokens are unconditional |
| `scheduler` | `diffusers.FlowMatchEulerDiscreteScheduler` | stock; the pipeline passes `-v` to match upstream's update exactly |
| `transformer` | `TripoSGTransformerModel` | 21 blocks, width 2048, on Diffusers blocks; LoRA-loadable |
| `vae` | `TripoSGVAEDecoder` | decoder half of the shape VAE |

The attention splits heads before the q/k/v projections. This matches the released weights, which therefore load
unchanged.

## Differences from upstream

- Extraction follows upstream's default where it can run. On CUDA devices `use_flash_decoder` defaults to on:
  upstream's FlashVDM grid (63³ cells refined to 504³ at `flash_octree_depth=9`, its top-k attention switched off as
  upstream leaves it) meshed with DISO's `DiffDMC`, which needs `pip install diso` (CUDA only, CC BY-NC 4.0). The
  grid reuses this package's hierarchical decoder with upstream's twice-dilated candidates, coarse 4³-block batching,
  and 6³-binned query order, so its decoder calls have upstream's shapes. Upstream's quirks are kept: the signed
  distance handed to `DiffDMC` is divided by 32, and vertices are scaled by 512 although the grid has 504 cells.
- Elsewhere, where `DiffDMC` cannot run, or with `use_flash_decoder=False`, it is upstream's
  `use_flash_decoder=False` path: a dense grid, one refinement near the surface, and scikit-image marching cubes.
  That path's quirks are kept too: inclusive dense bounds, refined spacing of `bounds / 256`, and a single symmetric
  `bounds` value. CPU meshes therefore differ from upstream's default CUDA ones.
- `BitImageProcessorPil` was compared with the `BitImageProcessor` of transformers 4.45.2, used for the release, on
  seven images of different sizes: the pixel values are identical.
- Not ported: the VAE encoder (needs `torch_cluster`), TripoSG-scribble, the inference script's optional `pymeshlab`
  face reduction (`--faces`, off by default), and training.

## Conversion

```bash
diffusers-3d-convert-triposg VAST-AI/TripoSG output/
```

The source is a Hub id or a local snapshot; `--revision` defaults to the pinned weights. The converter keeps the
transformer tensors unchanged and drops the VAE encoder. It does not write DINOv2 or its preprocessor.

## Verification

Tiny-weight CPU tests check that the transformer, VAE decoder, scheduler loop, hierarchical extraction with vertex
merging, the FlashVDM grid (every decoder call's shape, the signed distance handed to `DiffDMC`, and the mesh, with a
CPU stand-in for `DiffDMC`), image preparation (including the 2000 px downscale), and a 4-step guided pipeline are
bit-identical to the pinned upstream code. Set `DIFFUSERS_3D_TRIPOSG_REFERENCE_ROOT` to the
upstream checkout to run them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
