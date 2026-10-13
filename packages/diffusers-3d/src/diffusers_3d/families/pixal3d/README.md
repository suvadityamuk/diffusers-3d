# Pixal3D image-to-3D

This family integrates Tencent ARC's MIT-licensed Pixal3D (SIGGRAPH 2026) at revision
`f7cf38429b0bd264f1995f0f8743a88b1c728b94`, the head of upstream's default `master` branch (the README calls it
`main`: "an improved implementation based on Trellis.2 backbone"; the `paper` branch, built on Direct3D-S2, is not
ported). Weights are `TencentARC/Pixal3D` at `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`. Everything runs in plain
PyTorch on any device; the PBR-textured mesh postprocess needs a user-installed CuMesh, as for TRELLIS.2.

## What is TRELLIS.2 and what is new

Pixal3D is TRELLIS.2's three-stage cascade (sparse structure 16³ latents decoded to 64³ and pooled to 32³; shape
SLAT 32³ then 64³/96³; texture SLAT on the same grid) with one change to conditioning:

- **Reused unchanged** (exact TRELLIS.2 classes, no new registrations): `Trellis2SparseStructureDecoder`,
  `Trellis2ShapeDualGridDecoder`, `Trellis2PBRSparseDecoder` (the released `ss_dec_conv3d_16l8_fp16`,
  `shape_dec_next_dc_f16c32_fp16`, and `tex_dec_next_dc_f16c32_fp16` files are byte-identical, same sha256, to
  `microsoft/TRELLIS-image-large` and `microsoft/TRELLIS.2-4B`), `Trellis2FlowEulerScheduler` (the sampler code is
  byte-identical), the stock Transformers `DINOv3ViTModel` `image_encoder` with TRELLIS.2's token path (all four
  upstream extractors load the same DINOv3-L/16 and its manual final layer norm), the latent normalizations, sampler
  defaults, the shape-decoder cascade upsampling, and every SLAT / O-Voxel output path of `Trellis2ImageTo3DPipeline`.
- **Projection-conditioned flows**: `Pixal3DSparseStructureFlowModel` and `Pixal3DSLatFlowModel` subclass the
  TRELLIS.2 flows. Each block's cross-attention becomes upstream's `ProjectAttention`: cross-attention over only the
  DINOv3 class and register tokens, plus a per-block `proj_linear` of pixel-aligned features
  (`x + proj_linear(proj) + cross_attn(norm2(x), global)`). The checkpoint keys are unchanged
  (`blocks.N.cross_attn.cross_attn_block.*`, `blocks.N.cross_attn.proj_linear.*`).
- **Pixel-aligned projection** (`conditioner.py`): each latent cell `i` of an `R`-grid sits at
  `linspace(-1, 1, R)[i]`, is rotated from the Y-up grid into the Z-up Blender world, scaled by `1 / (2 *
  mesh_scale)`, projected through a 32 mm-sensor pinhole camera, and samples DINOv3 patch features bilinearly. The
  sparse-structure stage uses the raw patch features (1024 channels); the SLAT stages concatenate a NAF-upsampled copy
  (2048 channels). Only active voxels are projected, so the 1536 cascade never builds upstream's dense `96³ × 2048`
  grid.
- **NAF feature upsampler**: `Pixal3DNAFUpsampler` is valeo.ai NAF (image encoder, axial RoPE, 9×9 cross-scale
  neighborhood attention) in plain PyTorch. Upstream nearest-upsamples keys/values and runs a dilated NATTEN window;
  that is exactly a border-clamped window on the low-resolution grid, computed here without NATTEN.
- **Pipeline**: `Pixal3DImageTo3DPipeline` subclasses `Trellis2ImageTo3DPipeline` and adds the `feature_upsampler`
  component, per-stage conditioning (DINOv3 at 512 for the sparse-structure and 512 shape stages, 1024 for the 1024
  shape and texture stages; NAF to 512, 512, and 1024), and upstream's cascade quantization
  (`((c + 0.5) / 512 * (grid - 1)).round()`, onto the projection grid's `linspace` cells, rather than TRELLIS.2's
  truncation onto `grid`). Only `1024_cascade` and `1536_cascade` exist; `1536_cascade` is the released default.

## Inputs and cameras

```python
import math, torch
from transformers import DINOv3ViTModel

from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained("/path/to/pixal3d", image_encoder=image_encoder).to("cuda")
output = pipeline(ImageCondition(rgba), camera_angle_x=math.radians(40), formats=("o_voxel",))
```

- **Single view** (`pipeline.json` weights): an `ImageCondition`, preprocessed like upstream (`alpha > 0.8` box grown
  by 1.1, premultiplied on black, LANCZOS-resized). Projection needs the horizontal field of view of the cropped image:
  pass `camera_angle_x` (radians, one value or one per image) or attach an `ImageCondition.camera`. Upstream estimates
  it with MoGe-2 (`Ruicheng/moge-2-vitl`, MIT) or takes `--fov`; this package never runs MoGe or background removal.
  The camera sits on the canonical front view at upstream's `distance_from_fov` (the unit cube fills the image width).
  Like the TRELLIS.2 port, the pipeline does not first shrink inputs larger than 1024 pixels as upstream's
  `preprocess_image` does, so large photos crop at full resolution and can differ slightly.
- **Multi-view** (`pipeline_mv.json` weights, converted with `--config-file pipeline_mv.json`): a
  `MultiViewCondition` with one OpenCV `CameraRig` camera per square view. View 0 must be the front view; every view
  is re-expressed relative to it (`F @ inv(C_0) @ C_i`), projected, and averaged, as upstream's `average` fusion.
  Views are not cropped. A checkpoint records which input it was trained for (`multiview`) and rejects the other.
- `mesh_scale` (default 1.0) is upstream's object scale factor.

**Output frame.** Latents are view-aligned: the grid is right-handed Y-up with the conditioning camera on `+z`, so
every asset is tagged `RIGHT_HANDED_Y_UP` (TRELLIS.2 assets are Z-up). Converting to Z-up gives upstream's Blender
world, with the camera at `(0, -d, 0)`. Upstream's GLB export applies `to_glb` (Z-up to Y-up) and then a fixed
`diag(-1, 1, -1)` turn. Meshing and the textured-mesh postprocess keep the asset's frame; the official O-Voxel
codecs (`official_tensors_from_ovoxel_asset`, `.npz` / `.vxz`) accept only the native Z-up frame and reject these
assets rather than mislabel them. The projection always samples in float32. Upstream also runs DINOv3 and NAF in
float32 while the flows run in `bfloat16`; to match that under a `bfloat16` load, pass float32 `image_encoder` and
`feature_upsampler` components to `from_pretrained`.

## Conversion

```bash
diffusers-3d-convert-pixal3d /path/to/TencentARC-Pixal3D /path/to/pixal3d \
    --image-encoder-path /path/to/dinov3-vitl16-pretrain-lvd1689m \
    --naf-checkpoint-path /path/to/naf_release.pth   # add --config-file pipeline_mv.json for multi-view
```

The converter loads every component with `strict=True`, requires `image_attn_mode == "proj"`, checks the dense
flow's stored `rope_phases` against the configured ones before dropping them, checks the NAF release's sha256
(`c096c1ab…c98f`) and derived RoPE periods, remaps NAF's `EncBlock` keys onto flat `nn.Sequential` encoders, and
writes the Diffusers layout, the `object3d_model_index.json` sidecar, and `pixal3d_conversion.json`.

**Released layout check** (safetensors headers only, recorded in `tests/families/pixal3d/test_models.py`): every
flow (single-view and `_mv`) has 700 tensors (TRELLIS.2's 640 plus 30 `proj_linear` weight/bias pairs);
`proj_linear` is `1536 × 1024` for the sparse-structure flow and `1536 × 2048` for the SLAT flows; the dense flow
additionally stores `rope_phases` (`complex64 [4096, 64]`). The NAF release holds 36 parameters plus RoPE periods.

## Parity

`tests/families/pixal3d/test_reference_parity.py` runs tiny random weights through the pinned upstream code and the
port on CPU float32 (maximum absolute differences measured on macOS arm64):

| Case | Result |
|---|---|
| Sparse-structure flow, projection mode | bitwise; stored `rope_phases` bitwise |
| SLAT flow, projection + texture concat (upstream SDPA backend) | 1.2e-7 |
| NAF upsampler vs upstream NAF on NATTEN's CPU Flex backend (border windows, dilations 2/4, guide downsizing) | 3.0e-7 |
| Single-view: `preprocess_image` + resize, `distance_from_fov`, global tokens, sparse-structure features | bitwise |
| Single-view NAF stages (512, 1024, texture), dense and sparse cells | 2.4e-7 |
| Multi-view: `to_cond_tensor`, global tokens | bitwise; projected features 4.8e-7 |

Not covered by parity: full-checkpoint numerics, `bfloat16` execution, and the RNG sequence of a whole run.

## Licenses

Pixal3D code and weights are MIT (see `LICENSE-MIT`, `NOTICE`, including upstream's Responsible Use section). NAF code
is Apache-2.0; its release checkpoint carries no separate license. DINOv3 weights are gated under the DINOv3 License
and are not redistributed; upstream's inference scripts load `camenduru/dinov3-vitl16-pretrain-lvd1689m`, an ungated,
byte-identical re-host of the gated weights (same license). RMBG-2.0 (Bria, non-commercial, gated) is used by upstream
only for background removal and is not used here.

## Not ported

Training and data tooling; the `gated_proj` (DINOv3 + VAE) and `attention` multi-view fusion modes (no released
weights use them); MoGe-2 FOV estimation and BiRefNet background removal (callers supply FOV and alpha); upstream's
`low_vram` mode (use Diffusers offloading); the `paper` branch.
