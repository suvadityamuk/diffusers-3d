# Hunyuan3D-Omni

Tencent Hunyuan3D-Omni, ported from upstream revision `4d47c0cc2bd0c4281963a7314ab330a5af36bfa8` with weights from
`tencent/Hunyuan3D-Omni` at `70e803bfb4e127d534049d8ab8c8cb511780d485`. It turns one image into an untextured mesh
under one 3D control: a point cloud, voxels, a bounding box, or a skeleton. Everything runs in plain PyTorch on any
device.

**License.** Code and weights are under the
[Tencent Hunyuan 3D Omni Community License](LICENSE-TENCENT-HUNYUAN-3D-OMNI), not Apache-2.0. It does not apply in
the EU, the UK, or South Korea. It also requires a separate license above 1 million monthly active users and forbids
using outputs to improve other AI models. Constructing the pipeline warns about it.

```python
import numpy as np
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

# A folder written by diffusers-3d-convert-hunyuan3d-omni.
pipeline = AutoPipelineForImageTo3D.from_pretrained("hunyuan3d-omni", dtype=torch.float16).to("cuda")
bones = torch.from_numpy(np.loadtxt("demos/pose/a_pose_bone.txt")).float()  # (bones, 6)
mesh = pipeline(ImageCondition(image=rgba), skeleton=bones, generator=torch.Generator("cuda").manual_seed(1234)).objects[0]
```

## Controls

Pass exactly one, as a tensor in the output mesh's normalized frame (right-handed, Y up, inside `[-1, 1]`). A leading
batch dimension gives each image its own control; without one it is shared.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `point_cloud` | `(points, 3)` | Points. Upstream centers the mesh's bounding box and scales its longest side to 1.96. |
| `voxels` | `(points, 3)` | Surface samples (longest side 1.9998); the encoder keeps the occupied cells of a 16³ grid. |
| `bbox` | `(3,)` | Side lengths along x, y, z of an origin-centered box; the demos use the longest side 1. |
| `skeleton` | `(bones, 6)` | Bone start and end points `(x1, y1, z1, x2, y2, z2)`. |

Defaults follow upstream's `inference.py` for every control: 50 steps, guidance 4.5, a 513³ occupancy grid
(`octree_resolution=512`), iso-level 0. Meshes are inside `[-1.01, 1.01]³`. Marching cubes needs
`pip install "diffusers-3d[portable]"`.

Meshes then get `inference.py`'s clean-up: pymeshlab's `FloaterRemover` (components under 0.5% of the faces) and
`DegenerateFaceRemover`, with upstream's PLY round trips. It runs twice for `bbox`, whose `inference.py` path calls it
in `infer_bbox` and again in `postprocess`. It needs `pip install "diffusers-3d[meshlab]"` (GPL); `postprocess=False`
skips it.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2Model` | `facebook/dinov2-large` at 518 px, loaded from there as upstream does |
| `control_encoder` | `Hunyuan3DOmniControlEncoder` | control rows plus 10 control-type tokens, appended to the image tokens |
| `transformer` | `Hunyuan3DDiTModel` | the Hunyuan3D-2.1 DiT, unchanged; the control tokens only lengthen its cross-attention context |
| `vae` | `Hunyuan3DShapeVAE` | the Hunyuan3D-2.1 ShapeVAE decoder with Omni's weights and scale factor |
| `scheduler` | `Hunyuan3DFlowMatchEulerDiscreteScheduler` | equals upstream's Euler ODE over `linspace(0, 1, steps)` |

Classifier-free guidance zeroes only the image tokens; both branches see the control tokens, as upstream.

## Differences from upstream

- Controls are tensors. The open3d and trimesh loading, normalization, and surface sampling in upstream's
  `inference.py` are left to the caller.
- Grid points are rounded to float16 before querying, and meshes are merged by trimesh twice, as upstream.
- Every image in a batch gets a mesh; upstream's extractor keeps only the first. A field that never crosses
  `mc_level` raises instead of returning an empty mesh.
- Not ported: FlashVDM (`fast_decode`, off by default, as is `--flashvdm` in `inference.py`), DISO (only used with
  it), the ShapeVAE encoder, and training (point dropout and noise).

## Conversion

```bash
diffusers-3d-convert-hunyuan3d-omni tencent/Hunyuan3D-Omni hunyuan3d-omni
```

The source is a Hub id or a local release folder; `--revision` defaults to the pinned weights and `--ema` converts
the EMA DiT. The converter reuses the Hunyuan3D-2.1 key mapping for the DiT and VAE, renames the control encoder, and
reads only DINOv2-large's config: the pipeline loads the encoder from `facebook/dinov2-large` at a pinned revision and
the folder never holds it. Weights stay float32 (the DiT is 12.2 GB); load with `dtype=torch.float16` to run like
upstream.

## Verification

Tiny-weight CPU tests compare the conditioning for all four controls, the DiT, the VAE decoder, the sampler, the
image preprocessing, upstream's own pipeline end to end, and `inference.py`'s pymeshlab clean-up, run once and
twice. Set `DIFFUSERS_3D_HUNYUAN3D_OMNI_REFERENCE_ROOT` to the
upstream checkout to run them.

See [LICENSE-TENCENT-HUNYUAN-3D-OMNI](LICENSE-TENCENT-HUNYUAN-3D-OMNI) and [NOTICE](NOTICE).
