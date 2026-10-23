# Inference

Sections 1–6 use TRELLIS.2 as the example. Every family loads, conditions, and saves the same way; the differences are
listed under [Other families](#other-families). [Texture a mesh](#texture-a-mesh) covers the two texture stages.

A runnable version of this walkthrough is
[`families/trellis2/examples/image_to_3d.py`](../src/diffusers_3d/families/trellis2/examples/image_to_3d.py):

```bash
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/   # offline, random tiny weights
python -m diffusers_3d.families.trellis2.examples.image_to_3d --model /path/to/trellis2 --image chair.png --output out/
```

## 1. Load a pipeline

Converted checkpoints for every family are on the Hub; the
[repository README](../../../README.md#supported-models) lists them.

```python
import torch
from transformers import DINOv3ViTModel
from diffusers_3d import AutoPipelineForImageTo3D

# The DINOv3 encoder is gated, so the Hub repo leaves it out. Accept its license on the Hub, then pass it in.
image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", image_encoder=image_encoder, dtype=torch.bfloat16
).to("cuda")
```

The auto-loader reads the repo's `object3d_model_index.json`, checks each component class against the installed
package, downloads only the component folders it needs, and loads with remote code disabled. It accepts `revision`,
`cache_dir`, `token`, `local_files_only`, and `subfolder`. A keyword argument named after a component, such as
`image_encoder=`, replaces that component and skips its download. A locally converted folder already contains the
encoder and needs no extra argument.

The concrete class works too: `Trellis2ImageTo3DPipeline.from_pretrained(...)`. Pipelines are `DiffusionPipeline`s,
so `.to()`, `save_pretrained()`, and `enable_model_cpu_offload()` behave as usual.

## 2. Build the condition

`ImageCondition.image` is a float `(C, H, W)` tensor in `[0, 1]` with 1, 3, or 4 channels. You can attach a
`(1, H, W)` `mask` and a `camera`.

```python
import numpy as np
from PIL import Image
from diffusers_3d import ImageCondition

with Image.open("chair.png") as image:
    pixels = torch.from_numpy(np.asarray(image.convert("RGBA"), dtype=np.float32))
condition = ImageCondition(image=pixels.permute(2, 0, 1) / 255.0)
```

With alpha or a mask, the pipeline crops and recenters the object exactly as upstream does. Plain RGB is treated as an
image whose background is already removed. No pipeline runs a background remover. A bare tensor is wrapped in an
`ImageCondition` for you, and a list makes a batch.

## 3. Generate

```python
output = pipeline(
    condition,
    formats=("sparse_structure", "o_voxel"),
    sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5},
    generator=torch.Generator("cuda").manual_seed(0),
)
```

TRELLIS.2 runs in three stages. A sparse-structure flow picks the occupied voxels of a coarse grid. Shape and texture
SLAT flows denoise one latent per voxel; the cascade presets run a 512 stage, upsample, and run a 1024 stage. The shape
and PBR decoders then turn the latents into a surface with materials.

- `formats` lists the outputs to return, in order: `"sparse_structure"`, `"shape_slat"`, `"texture_slat"`,
  `"o_voxel"`, `"mesh"`. The default is `("o_voxel",)`, or `("sparse_structure",)` when the texture stage is not
  loaded.
- `*_sampler_params` override per-stage sampler settings. Missing keys use the released defaults stored in
  `pipeline.config`.
- `pipeline_type` picks the released preset: `"512"`, `"1024"`, `"1024_cascade"` (default), or `"1536_cascade"`.
  `max_num_tokens` caps the cascade's second stage as upstream does.
- `generator` makes noise reproducible; `sparse_structure_latents` supplies it directly.
- `return_dict=False` returns `(objects, latents)`, and `return_latents=False` drops the latents.

## 4. Use the result

`output.objects` is a tuple with one asset per input per format, in `formats` order. Each asset is a dataclass of
tensors with `.to(device, dtype)`, a `transform`, a `coordinate_system`, and JSON-safe `metadata`.

| Format | Asset | Contents |
|---|---|---|
| `sparse_structure` | `SparseVoxelAsset` | occupied coarse-grid `coordinates` and `features` |
| `shape_slat`, `texture_slat` | `SparseVoxelAsset` | denoised latents on the final grid |
| `o_voxel` | `OVoxelAsset` | dual-grid surface with `base_color`, `metallic`, `roughness`, `opacity`, `normals`, `emissive` |
| `mesh` | `MeshAsset` | the O-Voxel surface as a vertex-coloured mesh |

TRELLIS families are right-handed Z-up.

## 5. Load LoRA weights

Install PEFT with the `lora` extra. Pipelines load Diffusers-format LoRA checkpoints into their flow models:

```python
pipeline.load_lora_weights("user/trellis2-style-lora", adapter_name="style")
pipeline.set_adapters("style", adapter_weights=0.8)
pipeline.unload_lora_weights()
```

Checkpoint keys must start with the component they adapt, for example
`sparse_structure_flow_model.blocks.0.self_attn.to_qkv.lora_A.weight`. Other prefixes are rejected.

## 6. Save assets

```python
from diffusers_3d import MeshAsset, OVoxelAsset, SparseVoxelAsset, TrimeshBackend, write_ovoxel_npz

if isinstance(asset, SparseVoxelAsset):
    torch.save(asset.to("cpu"), "structure.pt")  # load with weights_only=False
elif isinstance(asset, OVoxelAsset):
    write_ovoxel_npz("object.npz", asset)  # readable by the official TRELLIS.2 tools
elif isinstance(asset, MeshAsset):
    TrimeshBackend().export_mesh(asset.to_coordinate_system("right_handed_y_up"), "object.glb")  # GLB is Y-up
```

For a PBR-textured GLB like upstream's `to_glb`, install CuMesh ([backends.md](backends.md)) and call
`pipeline.postprocess_ovoxel(asset, output_format="textured_mesh")`. Convert the result to `right_handed_y_up` before
exporting it.

## Texture a mesh

Two pipelines paint an existing mesh from a reference image. Both take a `MeshAsset` and an `ImageCondition` whose
alpha or mask isolates the object, and return a UV-textured `MeshAsset`. Load them with `AutoPipelineFor3D`:

```python
from diffusers_3d import AutoPipelineFor3D

shape = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = shape(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

paint = AutoPipelineFor3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d", dtype=torch.float16).to("cuda")
textured = paint(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
TrimeshBackend().export_mesh(textured.to("cpu"), "textured.glb")
```

- **`Hunyuan3DPaintPipeline`** simplifies the mesh to `max_faces=40000` (`None` keeps it), unwraps it with xatlas,
  generates albedo and metallic-roughness views, upscales them with Real-ESRGAN, and bakes a PBR material.
- **`Step1X3DMeshTexturePipeline`** generates six orthographic views with SDXL and a multi-view adapter, keeps them
  consistent through a shared latent texture for the first `texture_sync_ratio` of the steps, and bakes a base-colour
  texture. `output.previews` holds the views.

Both rasterize and bake in plain PyTorch and need the `portable` extra.

## Other families

All load with `AutoPipelineForImageTo3D` and return assets in `output.objects`. Each family README has the details.

| Family | Input | Output | Main arguments |
|---|---|---|---|
| TripoSG | RGBA | mesh, Y-up | `num_inference_steps=50`, `guidance_scale=7.0`, `dense_octree_depth`, `hierarchical_octree_depth` |
| Step1X-3D | RGBA | mesh | `num_inference_steps=50`, `guidance_scale=7.5`, `octree_resolution=384` |
| Hunyuan3D-2.1 | RGBA | mesh | `num_inference_steps=50`, `guidance_scale=5.0`, `octree_resolution=384`; load in float16 |
| SAM 3D Objects | RGB plus `mask`, optional `pointmap` | splats and mesh posed in the camera frame | `formats` from `"sparse_structure"`, `"gaussian"`, `"mesh"` |
| Pixal3D | RGBA plus `camera_angle_x` (horizontal FOV, radians) | same as TRELLIS.2 | TRELLIS.2's `formats` and `pipeline_type` (default `"1536_cascade"`); needs `image_encoder=` like TRELLIS.2 |

TripoSG, Hunyuan3D-2.1, SAM 3D Objects, and Hunyuan3D-2.1 Paint warn about their license terms when constructed.

### TRELLIS

`TrellisImageTo3DPipeline` and `TrellisTextTo3DPipeline` (prompts or `TextCondition`) need no extra loading argument.

- `formats` comes from `"sparse_structure"`, `"slat"`, `"gaussian"`, `"mesh"`, and `"radiance_field"`. The default is
  every loaded decoder.
- Sampling uses flat arguments: `sparse_structure_num_inference_steps`, `slat_num_inference_steps`, `guidance_scale`,
  `guidance_interval`, `rescale_t`.
- `"gaussian"` returns a `GaussianSplatAsset`; rendering it needs gsplat.
- `"mesh"` returns a vertex-coloured FlexiCubes `MeshAsset` with the normal map in `extras["normal_map"]`. Drop the
  extras before exporting: `dataclasses.replace(mesh, extras={})`.
- `"radiance_field"` returns a `RadianceFieldAsset`. Render it with
  `diffusers_3d.backends.radiance_field.render_radiance_field(asset, cameras)`.
- For a textured GLB, pass the mesh and splats to `TrellisGlbPostprocessFacade().to_textured_mesh(...)`.

## Convert a checkpoint yourself

The Hub repos were made with these converters, so converting gives the same result. Each one checks the release's
configuration, loads weights strictly, and writes a Diffusers pipeline folder with an `object3d_model_index.json`.

| Family | Command |
|---|---|
| TRELLIS | `diffusers-3d-convert-trellis <release> <out> --encoder-path facebook/dinov2-with-registers-large` (or `openai/clip-vit-large-patch14` for text) |
| TRELLIS.2 | `diffusers-3d-convert-trellis2 <release> <out> --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m` |
| TripoSG | `diffusers-3d-convert-triposg VAST-AI/TripoSG <out>` |
| Step1X-3D geometry | `diffusers-3d-convert-step1x3d <Step1X-3D-Geometry-1300m folder> <out>` |
| Step1X-3D texture | `diffusers-3d-convert-step1x3d-texture <Step1X-3D-Texture folder> <out>` |
| Hunyuan3D-2.1 shape | `diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 <out>` |
| Hunyuan3D-2.1 Paint | `diffusers-3d-convert-hunyuan3d-paint tencent/Hunyuan3D-2.1 <out>` (`--no-upscaler` skips Real-ESRGAN) |
| SAM 3D Objects | `diffusers-3d-convert-sam3d <checkpoints folder> <out>` |
| Pixal3D | `diffusers-3d-convert-pixal3d <release> <out> --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m --naf-checkpoint-path naf_release.pth` |

`<release>` is the folder holding the release's `pipeline.json`. Encoder paths accept a Hub id or a local folder.
