# Inference with diffusers-3d

This page shows what running a generative 3D model looks like today. It uses TRELLIS.2 throughout; TRELLIS
(`TrellisImageTo3DPipeline`) follows the same shape with the argument names noted at the end. A complete, runnable
version of everything below lives in
[`families/trellis2/examples/image_to_3d.py`](../src/diffusers_3d/families/trellis2/examples/image_to_3d.py):

```bash
# offline tiny CPU demo, exercises the whole API with random weights
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/

# against a converted checkpoint
python -m diffusers_3d.families.trellis2.examples.image_to_3d --model /path/to/trellis2 --image chair.png --output out/
```

## 1. Get a checkpoint

Converted checkpoints are published on the Hub, so the usual path is to load one directly (section 2):

| Repository | Pipeline | Contents |
|---|---|---|
| `suvadityamuk/TRELLIS-image-large-diffusers-3d` | `TrellisImageTo3DPipeline` | every component, DINOv2 conditioner included |
| `suvadityamuk/TRELLIS-text-large-diffusers-3d` | `TrellisTextTo3DPipeline` | every component, CLIP conditioner included |
| `suvadityamuk/TRELLIS.2-4B-diffusers-3d` | `Trellis2ImageTo3DPipeline` | every component except the gated DINOv3 conditioner |

`scripts/publish_hub_checkpoints.py` produced them with the converters below, so converting yourself gives the same
result. Official TRELLIS.2 releases are not in the Diffusers layout; the packaged CLI translates one, pointing at the
release's `pipeline.json` directory and at a locally downloaded DINOv3 conditioner (the production weights are gated
on the Hub under the DINOv3 License and are not redistributed):

```bash
diffusers-3d-convert-trellis2 /path/to/TRELLIS.2-release /path/to/trellis2 \
    --conditioner-path /path/to/dinov3-vitl16-pretrain-lvd1689m
```

The output is a standard Diffusers pipeline directory (`model_index.json` plus one subfolder per component) with an
extra `object3d_model_index.json` sidecar. The sidecar records the exact class of each component and which ones are
eligible for automatic loading. Every released component is converted: the DINOv3 conditioner, the sparse-structure
flow and decoder, the 512 and 1024 shape and texture SLAT flows, and the shape (dual-grid) and PBR decoders. A
`trellis2_conversion.json` report in the output directory lists what was written.

## 2. Load a pipeline

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, Trellis2Dinov3Conditioner

# The published TRELLIS.2 repository leaves out the gated DINOv3 conditioner; accept its license on the Hub and pass it in.
conditioner = Trellis2Dinov3Conditioner.from_dinov3_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", conditioner=conditioner, dtype=torch.bfloat16
)
pipeline = pipeline.to("cuda")
# A locally converted folder loads the same way and needs no conditioner argument:
# pipeline = AutoPipelineForImageTo3D.from_pretrained("/path/to/trellis2", dtype=torch.bfloat16)
```

`AutoPipelineForImageTo3D` reads the sidecar, checks every component class against the installed package, downloads
only eligible component folders (for Hub IDs), and instantiates the concrete class with remote code disabled.
`revision`, `cache_dir`, `token`, `local_files_only`, and `subfolder` are accepted. Keyword arguments named after
declared components (`conditioner=...`) supply that component as an object: its subfolder is neither downloaded nor
required, and its exact class is still checked. `trust_remote_code=True` is an error; reviewed families never need it.

The concrete class works too:

```python
from diffusers_3d import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("/path/to/trellis2")
```

Because the pipeline is a `DiffusionPipeline`, the usual Diffusers lifecycle applies: `.to()`, `save_pretrained()`,
`enable_model_cpu_offload()`, and component access by attribute (`pipeline.sparse_structure_flow_model`, ...).

## 3. Build the condition

Inputs are typed. `ImageCondition.image` is a float `(C, H, W)` tensor in `[0, 1]` with 1, 3, or 4 channels; an
optional `mask` (`(1, H, W)`) and `camera` can be attached.

```python
import numpy as np
from PIL import Image
from diffusers_3d import ImageCondition

with Image.open("chair.png") as image:
    pixels = torch.from_numpy(np.asarray(image.convert("RGBA"), dtype=np.float32))
condition = ImageCondition(image=pixels.permute(2, 0, 1) / 255.0)
```

If alpha (or `mask`) is present, the pipeline applies the released TRELLIS.2 preprocessing exactly: quantize to
uint8, crop to `alpha > 0.8 * 255`, recenter, premultiply on black, LANCZOS-resize to the conditioner size. Plain RGB is
treated as an already background-removed frame. The pipeline never runs a background remover on your behalf.

A bare tensor is also accepted and is wrapped in an `ImageCondition` for you. Pass a list to batch several images.

## 4. Generate

```python
output = pipeline(
    condition,
    formats=("sparse_structure", "o_voxel"),
    sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5},
    generator=torch.Generator("cuda").manual_seed(0),
)
```

A TRELLIS.2 run has three stages. The sparse-structure flow decides which voxels of a coarse grid are occupied. The
shape and texture SLAT flows then denoise one latent per occupied voxel (the cascade presets run a 512 stage, upsample
the grid with the shape decoder, and run a 1024 stage on the result). Finally the shape decoder subdivides the SLAT into
a dual-grid surface and the PBR decoder paints it.

- `formats` names the representations to return, in order, from `"sparse_structure"`, `"shape_slat"`,
  `"texture_slat"`, `"o_voxel"`, and `"mesh"`. It defaults to `("o_voxel",)` when the texture stage is loaded and to
  `("sparse_structure",)` otherwise. `"mesh"` runs O-Voxel meshing through `OVoxelBackend` and needs the compiled
  runtime; every other format runs in plain PyTorch on CPU or GPU.
- `*_sampler_params` are per-stage mappings. Omitted keys fall back to the released defaults serialized in
  `pipeline.config` (for the sparse-structure stage: 12 steps, guidance strength 7.5, guidance rescale 0.7,
  guidance interval `(0.6, 1.0)`, `rescale_t` 5.0).
- `pipeline_type` picks the released preset (`"512"`, `"1024"`, `"1024_cascade"`, `"1536_cascade"`) and defaults to
  `pipeline.config.default_pipeline_type`, which the converter sets to `"1024_cascade"`. `max_num_tokens` caps the
  cascade's second-stage grid the way the upstream pipeline does.
- `generator` gives reproducible noise. `sparse_structure_latents` lets you supply the initial noise yourself.
- `return_latents=False` drops the latent tensor from the output; `return_dict=False` returns
  `(objects, latents)`.

## 5. Use the result

`Object3DPipelineOutput.objects` is a tuple of tensor-native assets: one per input image per requested format, in
`formats` order. Every asset is a dataclass of tensors, has `.to(device, dtype)`, and carries a JSON-safe `metadata`
dict describing where it came from.

```python
asset = output.objects[0]           # SparseVoxelAsset
asset.coordinates                    # (N, 3) int64 grid indices
asset.features                       # (N, C) per-voxel channels
asset.grid_transform                 # grid -> object space (4x4)
asset.transform                      # object -> world space (4x4)
asset.coordinate_system              # CoordinateSystem.RIGHT_HANDED_Z_UP for TRELLIS families
asset.metadata["representation"]     # "sparse_structure"
asset.metadata["resolution"]         # 32 for the cascade presets
output.latents.latents               # (B, C, D, H, W) sparse-structure latents
```

Which asset type you get depends on the format:

| `formats` entry | Asset | Notes |
|---|---|---|
| `sparse_structure` | `SparseVoxelAsset` | Occupancy grid decoded from dense latents. |
| `shape_slat`, `texture_slat` | `SparseVoxelAsset` | Denoised structured latents on the final stage's grid. `metadata["stage"]` is `"shape"` or `"texture"`. |
| `o_voxel` | `OVoxelAsset` | Dual-grid surface plus PBR channels (`base_color`, `metallic`, `roughness`, `opacity`, `normals`, `emissive`). |
| `mesh` | `MeshAsset` | Extracted through `OVoxelBackend`, which needs the compiled O-Voxel runtime. |

## 6. Save assets

There is no single 3D file format, so serialization is per representation:

```python
from diffusers_3d import OVoxelAsset, MeshAsset, SparseVoxelAsset, TrimeshBackend, write_ovoxel_npz

if isinstance(asset, SparseVoxelAsset):
    torch.save(asset.to("cpu"), "structure.pt")          # tensors + metadata, load with weights_only=False
elif isinstance(asset, OVoxelAsset):
    write_ovoxel_npz("object.npz", asset)                 # pure NumPy, readable by the official TRELLIS.2 tools
elif isinstance(asset, MeshAsset):
    TrimeshBackend().export_mesh(asset, "object.glb")     # needs the `portable` extra
```

`.vxz`, dual-grid mesh extraction, voxel rendering, and PBR GLB baking delegate to the compiled O-Voxel runtime and
its research-licensed dependencies. They are never imported implicitly; see [backends.md](backends.md) for how to
install and select them, and call `pipeline.postprocess_ovoxel(asset, output_format="glb")` explicitly when they are
available.

## TRELLIS (v1) differences

`TrellisImageTo3DPipeline` accepts `formats` from `{"sparse_structure", "slat", "gaussian", "mesh", "radiance_field"}`
(default: every
loaded SLAT decoder output) and uses flat keyword arguments instead of per-stage mappings:
`sparse_structure_num_inference_steps`, `slat_num_inference_steps`, `guidance_scale`, `guidance_interval`, `rescale_t`.
`"gaussian"` returns a `GaussianSplatAsset` from the windowed-attention Gaussian decoder; rasterizing it requires the
optional `gsplat` backend. `"mesh"` returns a `MeshAsset` from the FlexiCubes mesh decoder in plain PyTorch, with vertex
colours and the predicted normal map in `extras["normal_map"]`. To export it with `TrimeshBackend`, call
`mesh.to_coordinate_system("right_handed_y_up")` first (TRELLIS outputs are Z-up) and drop the extras, which no file
format carries: `dataclasses.replace(mesh, extras={})`. `"radiance_field"` returns a `RadianceFieldAsset` (the
released tri-vector field); `diffusers_3d.backends.radiance_field.render_radiance_field(asset, cameras)` volume-renders
it in plain PyTorch. For a textured GLB like upstream's `to_glb`, pass the mesh and the splats to
`TrellisGlbPostprocessFacade().to_textured_mesh(...)` (CuMesh, xatlas, and gsplat) and export the result the same way.
`TrellisTextTo3DPipeline` (`microsoft/TRELLIS-text-*`) takes prompts or `TextCondition` values instead of
images and shares everything else. Both are published converted (`suvadityamuk/TRELLIS-image-large-diffusers-3d`,
`suvadityamuk/TRELLIS-text-large-diffusers-3d`) and load with `AutoPipelineForImageTo3D` / `AutoPipelineForTextTo3D`
without any extra argument. To convert official checkpoints yourself use `diffusers-3d-convert-trellis`, which takes the
same arguments as the TRELLIS.2 converter. Its `--conditioner-path` is the released `dinov2_vitl14_reg`, published on
the Hub as `facebook/dinov2-with-registers-large` (or `openai/clip-vit-large-patch14` for the text pipelines):

```bash
hf download microsoft/TRELLIS-image-large --local-dir /path/to/TRELLIS-image-large
hf download facebook/dinov2-with-registers-large --local-dir /path/to/dinov2-with-registers-large
diffusers-3d-convert-trellis \
    --source-directory /path/to/TRELLIS-image-large \
    --output-directory /path/to/trellis-diffusers \
    --conditioner-path /path/to/dinov2-with-registers-large
```

## What is and is not covered

Every network in both families (conditioners, sparse-structure flows and decoders, SLAT flows, the Gaussian, mesh, and
radiance-field decoders, and the shape and PBR decoders) runs in plain PyTorch on any device. Sparse convolutions, pooling, subdivision, and
windowed attention are implemented in `families/trellis/sparse_ops.py` and checked numerically against the pinned
upstream code with tiny weights, with the upstream CUDA kernels (`spconv`, FlexGEMM, `xformers`) replaced by dense
PyTorch equivalents in the test. Full-resolution generation on the released weights and the compiled mesh, render, and
PBR GLB paths are not in the CPU CI; the [GPU smoke workflow](../../../.github/workflows/diffusers_3d_gpu_smoke.yml)
runs them on a Hugging Face Jobs A100 on demand and weekly (`scripts/gpu_smoke.py`), and the numbers from the last
manual run are recorded in [compatibility.md](compatibility.md). See that page and the family READMEs for the exact evidence behind each claim.
