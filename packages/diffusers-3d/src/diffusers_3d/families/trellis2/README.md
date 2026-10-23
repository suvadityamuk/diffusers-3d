# TRELLIS.2 image-to-3D

Microsoft TRELLIS.2, ported from upstream revision `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. A sparse-structure
stage, shape and texture SLAT flows, and two sparse decoders turn one image into an O-Voxel: a dual-grid surface with
PBR channels. All networks and mesh extraction run in plain PyTorch on any device. MIT license.

```python
import torch
from transformers import DINOv3ViTModel
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", image_encoder=image_encoder, dtype=torch.bfloat16
).to("cuda")
ovoxel = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
mesh = pipeline.postprocess_ovoxel(ovoxel, output_format="mesh")
```

The published repo omits the DINOv3 encoder because its weights are gated under Meta's DINOv3 License. A runnable
example is in [`examples/image_to_3d.py`](examples/image_to_3d.py).

- `formats` picks from `sparse_structure`, `shape_slat`, `texture_slat`, `o_voxel` (default), and `mesh`.
- `pipeline_type` is `512`, `1024`, `1024_cascade` (default), or `1536_cascade`. Every stage defaults to 12 steps.
- Images need RGBA, or RGB with a `mask`. RGB without a mask is treated as an already-cropped full frame.
- Assets are in the right-handed Z-up frame.
- `postprocess_ovoxel(asset, output_format="textured_mesh")` bakes a PBR-textured mesh. It needs CuMesh installed.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.DINOv3ViTModel` | last hidden state before the model's own norm, then an unparameterized layer norm |
| `sparse_structure_flow_model` | `Trellis2SparseStructureFlowModel` | dense transformer with RoPE and Q/K RMS norms |
| `sparse_structure_decoder` | `Trellis2SparseStructureDecoder` | the TRELLIS Conv3D decoder, unchanged |
| `shape_slat_flow_model`, `shape_slat_flow_model_1024` | `Trellis2SLatFlowModel` | sparse transformer, 512 and 1024 stages |
| `texture_slat_flow_model`, `texture_slat_flow_model_1024` | `Trellis2SLatFlowModel` | same, conditioned on the shape SLAT |
| `shape_slat_decoder` | `Trellis2ShapeDualGridDecoder` | sparse ConvNeXt U-Net to the dual grid |
| `pbr_decoder` | `Trellis2PBRSparseDecoder` | reuses the shape decoder's subdivision masks |
| `*_scheduler` | `Trellis2FlowEulerScheduler` | `w * cond + (1 - w) * uncond` guidance with rescale and interval |

## Differences from upstream

- FlexGEMM and xformers are replaced by plain PyTorch (`families/trellis/sparse_ops.py`).
- `ovoxel_to_mesh` is a plain-PyTorch port of `o_voxel.convert.flexible_dual_grid_to_mesh`. Its output matches
  upstream bit for bit.
- The textured mesh follows `o_voxel.postprocess.to_glb` on CuMesh. Texel rasterization and attribute baking are plain
  PyTorch instead of nvdiffrast, which is research-only licensed. CuMesh bundles cubvh under NVIDIA's non-commercial
  license; loading it warns.
- `.vxz` files and voxel rendering need a compiled `o_voxel`, which imports nvdiffrast. Loading it requires
  `accept_nvdiffrast_research_license=True`. `.npz` read/write is pure NumPy and is the lossless format.
- Inputs larger than 1024 px are not shrunk first, as upstream does, so large photos can differ slightly.
- No background removal. Upstream runs a background remover on images without alpha.

## Conversion

```bash
diffusers-3d-convert-trellis2 source/ output/ --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m
```

`source/` holds the upstream `pipeline.json` and component `.json`/`.safetensors` pairs. The converter writes every
component, including the DINOv3 encoder, and `object3d_model_index.json`.

## Verification

Tiny-weight CPU tests compare every network and the dual-grid mesh extraction against the pinned upstream code. Set
`DIFFUSERS_3D_TRELLIS2_REFERENCE_ROOT` to the upstream checkout to run them. GPU runs are in
[docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
