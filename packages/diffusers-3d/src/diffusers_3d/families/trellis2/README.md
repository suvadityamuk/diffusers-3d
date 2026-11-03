# TRELLIS.2 image-to-3D

Microsoft TRELLIS.2, ported from upstream revision `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. A sparse-structure
stage, shape and texture SLAT flows, and two sparse decoders turn one image into an O-Voxel: a dual-grid surface with
PBR channels. All networks and mesh extraction run in plain PyTorch on any device. MIT license.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", dtype=torch.bfloat16
).to("cuda")
ovoxel = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
mesh = pipeline.postprocess_ovoxel(ovoxel, output_format="mesh")
```

The DINOv3 encoder is not converted or published: `from_pretrained` loads it from
`facebook/dinov3-vitl16-pretrain-lvd1689m` at a pinned revision. Its weights are gated under Meta's DINOv3 License, so
accept the license on the Hub and log in first; without access, loading raises an error that says so.

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
| `sparse_structure_decoder` | `Trellis2SparseStructureDecoder` | TRELLIS-image-large's Conv3D decoder, loaded from there |
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
- `.vxz` files and voxel rendering from upstream's compiled `o_voxel` are not supported. `.npz` read/write is pure
  NumPy and is the lossless format.
- Inputs without usable alpha are cut out automatically, as upstream does: they are shrunk to at most 1024 px in
  their own PIL mode and run through BiRefNet with `briaai/RMBG-2.0` weights (gated, non-commercial), which load from
  that repository at a pinned revision; the output is bit-identical to upstream's wrapper. As upstream, only RGBA
  alpha counts (LA and transparent palette inputs are cut out again). `pipeline.disable_background_removal()` turns
  it off; inputs that already carry alpha are shrunk exactly as upstream.
- Upstream draws its noise in float32 from the CPU global RNG after `torch.manual_seed(seed)`; pass
  `generator=torch.Generator().manual_seed(seed)` (a CPU generator) for the same noise on any device. Loaded without
  `dtype`, the flows keep upstream's mixed precision and the noise is float32 too.

## Conversion

```bash
diffusers-3d-convert-trellis2 source/ output/
```

`source/` holds the upstream `pipeline.json` and component `.json`/`.safetensors` pairs. The converter writes every
TRELLIS.2 component. `model_index.json` names the DINOv3 encoder and the sparse-structure decoder TRELLIS.2 references
from `microsoft/TRELLIS-image-large`; neither is downloaded or written, and `from_pretrained` loads the decoder from
that repository at a pinned revision, converting it on load.

## Verification

Tiny-weight CPU tests compare every network and the dual-grid mesh extraction against the pinned upstream code. Set
`DIFFUSERS_3D_TRELLIS2_REFERENCE_ROOT` to the upstream checkout to run them. GPU runs are in
[docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
