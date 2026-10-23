# SAM 3D Objects image-to-3D

Meta SAM 3D Objects, ported from upstream revision `f91db411c50efee93d8db7aeb323885650f6f722` with the gated
`facebook/sam-3d-objects` weights. It reconstructs the masked object as Gaussian splats and a mesh, placed in the input
camera's frame. It is TRELLIS with a new shape-and-layout stage and new conditioning.

**License.** Code and weights are under the [SAM License](LICENSE-SAM), which is not open source. It allows commercial
use and redistribution, but forbids military, weapons, nuclear, and espionage uses. Constructing the pipeline warns
about it.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/sam-3d-objects-diffusers-3d").to("cuda")
condition = ImageCondition(image=rgb, mask=object_mask)  # (3, H, W) in [0, 1] and (1, H, W)
gaussians, mesh = pipeline(condition, pointmap=points, generator=torch.Generator("cuda").manual_seed(0)).objects
```

- `mask` is required. The pipeline never segments, and the image itself is not masked.
- `pointmap` is optional: `(3, H, W)` camera-space points in OpenCV axes, non-finite where invalid (for example MoGe's
  output). Without it the shape is still generated, but the layout is scale-free.
- `formats` picks from `sparse_structure`, `gaussian`, and `mesh`. The default is every loaded decoder.
- Defaults: 25 steps per stage, guidance 7.0 then 1.0, `rescale_t` 3.0 then 1.0, guidance in the noisier half.
- Each asset's `transform` places it in the PyTorch3D camera frame (x left, y up, z forward), in pointmap units.
  `metadata["layout"]` holds upstream's rotation, translation, and scale. `GsplatBackend` needs identity transforms,
  so bake or drop the transform before rendering.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2WithRegistersModel` | encodes the image and mask crops and full frames |
| `sparse_structure_conditioner`, `slat_conditioner` | `Sam3DConditionEmbedder` | image, mask, and pointmap tokens |
| `sparse_structure_flow_model` | `Sam3DSparseStructureFlowModel` | mixture-of-transformers flow for shape, rotation, translation, and scale |
| `sparse_structure_decoder` | `TrellisSparseStructureDecoder` | TRELLIS, unchanged |
| `slat_flow_model` | `TrellisSLatFlowModel` | TRELLIS, unchanged |
| `gaussian_decoder`, `mesh_decoder` | `TrellisSLatGaussianDecoder`, `TrellisSLatMeshDecoder` | TRELLIS, unchanged |
| `*_scheduler` | `TrellisFlowEulerScheduler` | run on `1 - t`, because SAM 3D integrates from noise at `t=0` |

## Differences from upstream

- MoGe depth estimation is not bundled. It is a separate model, so pass its pointmap instead.
- Each component runs in its own dtype. Upstream runs the sparse-structure stage under float16 autocast on CUDA.
- Not ported, all off in upstream's default inference: layout post-optimization, mesh simplification, hole filling,
  texture baking, and plane estimation.
- Not ported, unused by default inference: the 4-Gaussian decoder, the sparse-structure encoder, and shortcut sampling.

## Conversion

```bash
diffusers-3d-convert-sam3d sam-3d-objects/checkpoints/ output/
```

The source is the released `checkpoints/` folder with `pipeline.yaml`. The converter maps upstream keys onto the
Transformers and package modules and loads every component with strict key checks. The release keeps four copies of
DINOv2; the converter checks they match and writes one.

## Verification

Tiny-weight CPU tests compare the condition embedders, preprocessing, sparse-structure flow and sampler, and SLAT
sampler against the pinned upstream code. Set `DIFFUSERS_3D_SAM3D_REFERENCE_ROOT` to the upstream checkout to run
them. GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-SAM](LICENSE-SAM) and [NOTICE](NOTICE).
