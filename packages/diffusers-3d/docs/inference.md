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

Official TRELLIS.2 releases are not in the Diffusers layout. Convert them once with the packaged CLI, pointing at the
release's `pipeline.json` directory and at a locally downloaded DINOv3 conditioner (the production weights are gated
on the Hub under the DINOv3 License and are not redistributed):

```bash
diffusers-3d-convert-trellis2 /path/to/TRELLIS.2-release /path/to/trellis2 \
    --conditioner-path /path/to/dinov3-vitl16-pretrain-lvd1689m
```

The output is a standard Diffusers pipeline directory (`model_index.json` plus one subfolder per component) with an
extra `object3d_model_index.json` sidecar. The sidecar records the exact class of each component and which ones are
eligible for automatic loading. By default only the reviewed sparse-structure components are converted;
`--include-experimental` adds SLAT and decoder components for tiny layouts only.

## 2. Load a pipeline

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D

pipeline = AutoPipelineForImageTo3D.from_pretrained("/path/to/trellis2")  # local dir or Hub repo ID
pipeline = pipeline.to("cuda", dtype=torch.float16)
```

`AutoPipelineForImageTo3D` reads the sidecar, checks every component class against the installed package, downloads
only eligible component folders (for Hub IDs), and instantiates the concrete class with remote code disabled.
`revision`, `cache_dir`, `token`, `local_files_only`, and `subfolder` are accepted. `trust_remote_code=True` is an
error; reviewed families never need it.

The concrete class works too, and is required for local artifacts that include experimental components:

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
    formats=("sparse_structure",),
    sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5},
    generator=torch.Generator("cuda").manual_seed(0),
)
```

- `formats` names the representations to return, in order. The reviewed TRELLIS.2 contract is `"sparse_structure"`.
  `"shape_slat"`, `"texture_slat"`, `"o_voxel"`, and `"mesh"` are experimental and currently only run with
  `pipeline_type="tiny"` on backend-free tiny components; asking for them on a converted `1024_cascade` checkpoint
  raises `NotImplementedError` with the reason.
- `*_sampler_params` are per-stage mappings. Omitted keys fall back to the released defaults serialized in
  `pipeline.config` (for the sparse-structure stage: 12 steps, guidance strength 7.5, guidance rescale 0.7,
  guidance interval `(0.6, 1.0)`, `rescale_t` 5.0).
- `pipeline_type` picks the released preset (`"512"`, `"1024"`, `"1024_cascade"`, `"1536_cascade"`, `"tiny"`) and
  defaults to `pipeline.config.default_pipeline_type`.
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
| `sparse_structure` | `SparseVoxelAsset` | Reviewed. Occupancy grid decoded from dense latents. |
| `shape_slat`, `texture_slat` | `SparseVoxelAsset` | Experimental tiny stages. `metadata["stage"]` is `"shape"` or `"texture"`. |
| `o_voxel` | `OVoxelAsset` | Experimental. Full PBR channel layout (`base_color`, `metallic`, `roughness`, `opacity`, `normals`, `emissive`, dual-grid fields). |
| `mesh` | `MeshAsset` | Experimental; extracted through `OVoxelBackend`, which needs the compiled O-Voxel runtime. |

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

`TrellisImageTo3DPipeline` accepts `formats` from `{"sparse_structure", "slat", "gaussian"}` and uses flat keyword
arguments instead of per-stage mappings: `sparse_structure_num_inference_steps`, `slat_num_inference_steps`,
`guidance_scale`, `guidance_interval`, `rescale_t`. `"gaussian"` returns a `GaussianSplatAsset`; rasterizing it
requires the optional `gsplat` backend. Convert official checkpoints with `diffusers-3d-convert-trellis`, which takes
the same arguments as the TRELLIS.2 converter.

## What is and is not covered

The reviewed inference contract for both families ends at CPU-capable sparse-structure output with measured tiny
parity against the pinned upstream implementation. Full-resolution GPU quality, the 1024 cascade with production
SLAT/O-Voxel stages, and compiled mesh/PBR export have not been run in this package's test matrix. See
[compatibility.md](compatibility.md) and the family READMEs for the exact evidence behind each claim.
