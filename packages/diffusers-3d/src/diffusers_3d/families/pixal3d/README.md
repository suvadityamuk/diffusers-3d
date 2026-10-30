# Pixal3D image-to-3D

Tencent ARC Pixal3D, ported from upstream revision `f7cf38429b0bd264f1995f0f8743a88b1c728b94` (the TRELLIS.2-based
`master` branch) with weights from `TencentARC/Pixal3D` at `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`. It is the
TRELLIS.2 cascade with pixel-aligned conditioning: each latent cell is projected into the image and samples DINOv3
features there. Everything runs in plain PyTorch on any device. MIT license.

```python
import math
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Pixal3D-diffusers-3d").to("cuda")
ovoxel = pipeline(ImageCondition(image=rgba), camera_angle_x=math.radians(40)).objects[0]
```

Only Pixal3D's own networks are converted and published. `from_pretrained` loads the third-party ones from their
original sources at pinned revisions: DINOv3 from `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated under Meta's
DINOv3 License; accept it on the Hub and log in first), valeo.ai's NAF upsampler from its
[release checkpoint](https://github.com/valeoai/NAF/releases) (checked against its SHA-256 and converted on load),
and, for single-view checkpoints, MoGe-2 from `Ruicheng/moge-2-vitl` (converted on load).

- `camera_angle_x` (radians, the horizontal field of view of the image) is optional. Without it the field of view comes
  from the `ImageCondition`'s `camera`, else from the MoGe-2 `depth_estimator`, run on the cropped image as
  upstream does. Off CUDA MoGe-2 runs in float32, because upstream's float16 autocast only works on CUDA.
- Images need RGBA, or RGB with a `mask`.
- `pipeline_type` is `1024_cascade` or `1536_cascade` (default). `formats` and sampler options work as in TRELLIS.2.
- `mesh_scale` (default 1.0) is upstream's object scale factor.
- Assets are right-handed Y-up with the camera on `+z`, unlike TRELLIS.2's Z-up. The `.npz` O-Voxel codec
  accepts only Z-up and rejects these assets.
- Multi-view checkpoints (converted from `pipeline_mv.json`) take a `MultiViewCondition` with one camera per square
  view; view 0 is the front view.

## Components

The decoders, schedulers, and image encoder are the TRELLIS.2 classes, unchanged. The release's three decoders are
byte-identical copies of `microsoft/TRELLIS-image-large` (sparse structure) and `microsoft/TRELLIS.2-4B` (shape, PBR)
files, so `from_pretrained` loads them from those repositories at pinned revisions and converts them on load. New:

| Component | Class | Notes |
| --- | --- | --- |
| `sparse_structure_flow_model` | `Pixal3DSparseStructureFlowModel` | TRELLIS.2 flow; cross-attention adds projected DINOv3 patch features |
| `shape_slat_flow_model(_1024)`, `texture_slat_flow_model(_1024)` | `Pixal3DSLatFlowModel` | same, with NAF-upsampled features concatenated |
| `feature_upsampler` | `Pixal3DNAFUpsampler` | valeo.ai NAF feature upsampler |

## Differences from upstream

- NAF runs in plain PyTorch. Upstream's dilated NATTEN window on upsampled keys equals a border-clamped window on the
  low-resolution grid, which is what this computes.
- Only active voxels are projected, so the 1536 cascade never builds upstream's dense `96³ × 2048` grid.
- `from_pretrained` keeps DINOv3 and NAF in float32 whatever `dtype` the rest loads in, as upstream does; a `dtype`
  dict that names `image_encoder` or `feature_upsampler` overrides this.
- Inputs without usable alpha are cut out automatically, as upstream does: they are shrunk to at most 1024 px in
  their own PIL mode and run through BiRefNet with `briaai/RMBG-2.0` weights (gated, non-commercial), which load from
  that repository at a pinned revision; the output is bit-identical to upstream's wrapper. As upstream, only RGBA
  alpha counts (LA and transparent palette inputs are cut out again). `pipeline.disable_background_removal()` turns
  it off; inputs that already carry alpha are shrunk exactly as upstream.
- Not ported: the `gated_proj` and `attention` multi-view fusion modes (no released weights), `low_vram` mode (use
  Diffusers offloading), the Direct3D-S2 `paper` branch, and training.

## Conversion

```bash
diffusers-3d-convert-pixal3d TencentARC-Pixal3D/ output/   # add --config-file pipeline_mv.json for multi-view
```

The converter loads every Pixal3D component with `strict=True` and writes the Diffusers layout. `model_index.json`
names DINOv3, NAF, and, for single-view folders, the MoGe-2 `depth_estimator` (see
[`families/moge`](../moge/README.md)); none of them is downloaded or written. `--no-depth-estimator` leaves MoGe-2
out.

## Verification

Tiny-weight CPU tests compare both flows, NAF, preprocessing (including the shrink of large inputs), the MoGe-2 field
of view, and single- and multi-view projection against the pinned upstream code. Set `DIFFUSERS_3D_PIXAL3D_REFERENCE_ROOT` and `DIFFUSERS_3D_NAF_REFERENCE_ROOT` to the upstream
checkouts to run them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE). NAF code is Apache-2.0.
