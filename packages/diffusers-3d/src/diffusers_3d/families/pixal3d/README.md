# Pixal3D image-to-3D

Tencent ARC Pixal3D, ported from upstream revision `f7cf38429b0bd264f1995f0f8743a88b1c728b94` (the TRELLIS.2-based
`master` branch) with weights from `TencentARC/Pixal3D` at `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`. It is the
TRELLIS.2 cascade with pixel-aligned conditioning: each latent cell is projected into the image and samples DINOv3
features there. Everything runs in plain PyTorch on any device. MIT license.

```python
import math
import torch
from transformers import DINOv3ViTModel
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/Pixal3D-diffusers-3d", image_encoder=image_encoder
).to("cuda")
ovoxel = pipeline(ImageCondition(image=rgba), camera_angle_x=math.radians(40)).objects[0]
```

The published repo omits the DINOv3 encoder because its weights are gated under Meta's DINOv3 License.

- `camera_angle_x` (radians, the horizontal field of view of the image) is required, unless the `ImageCondition`
  carries a `camera`. Upstream estimates it with MoGe-2; this package does not.
- Images need RGBA, or RGB with a `mask`.
- `pipeline_type` is `1024_cascade` or `1536_cascade` (default). `formats` and sampler options work as in TRELLIS.2.
- `mesh_scale` (default 1.0) is upstream's object scale factor.
- Assets are right-handed Y-up with the camera on `+z`, unlike TRELLIS.2's Z-up. The `.npz`/`.vxz` O-Voxel codecs
  accept only Z-up and reject these assets.
- Multi-view checkpoints (converted from `pipeline_mv.json`) take a `MultiViewCondition` with one camera per square
  view; view 0 is the front view.

## Components

The decoders, schedulers, and image encoder are the TRELLIS.2 classes, unchanged. New:

| Component | Class | Notes |
| --- | --- | --- |
| `sparse_structure_flow_model` | `Pixal3DSparseStructureFlowModel` | TRELLIS.2 flow; cross-attention adds projected DINOv3 patch features |
| `shape_slat_flow_model(_1024)`, `texture_slat_flow_model(_1024)` | `Pixal3DSLatFlowModel` | same, with NAF-upsampled features concatenated |
| `feature_upsampler` | `Pixal3DNAFUpsampler` | valeo.ai NAF feature upsampler |

## Differences from upstream

- NAF runs in plain PyTorch. Upstream's dilated NATTEN window on upsampled keys equals a border-clamped window on the
  low-resolution grid, which is what this computes.
- Only active voxels are projected, so the 1536 cascade never builds upstream's dense `96³ × 2048` grid.
- Inputs larger than 1024 px are not shrunk first, as upstream does, so large photos can differ slightly.
- Upstream runs DINOv3 and NAF in float32 under a bfloat16 load. To match that, pass float32 `image_encoder` and
  `feature_upsampler` components.
- No background removal and no field-of-view estimation. Upstream uses BiRefNet and MoGe-2.
- Not ported: the `gated_proj` and `attention` multi-view fusion modes (no released weights), `low_vram` mode (use
  Diffusers offloading), the Direct3D-S2 `paper` branch, and training.

## Conversion

```bash
diffusers-3d-convert-pixal3d TencentARC-Pixal3D/ output/ \
    --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m \
    --naf-checkpoint-path naf_release.pth   # add --config-file pipeline_mv.json for multi-view
```

The converter loads every component with `strict=True`, checks the NAF checkpoint's SHA-256, and writes the Diffusers
layout and `object3d_model_index.json`.

## Verification

Tiny-weight CPU tests compare both flows, NAF, preprocessing, and single- and multi-view projection against the pinned
upstream code. Set `DIFFUSERS_3D_PIXAL3D_REFERENCE_ROOT` and `DIFFUSERS_3D_NAF_REFERENCE_ROOT` to the upstream
checkouts to run them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE). NAF code is Apache-2.0.
