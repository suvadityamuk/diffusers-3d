# Inference

Sections 1–6 use TRELLIS.2 as the example. Every family loads, conditions, and saves the same way; the differences are
listed under [Other families](#other-families). [Texture a mesh](#texture-a-mesh) covers the two texture stages.

## 1. Load a pipeline

Converted checkpoints for every family are on the Hub; the
[repository README](../../../README.md#supported-models) lists them.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D

pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", dtype=torch.bfloat16
).to("cuda")
```

The DINOv3 image encoder is gated third-party weights. The library downloads it from `facebook/dinov3-vitl16-pretrain-lvd1689m`
at a pinned revision, and logs a notice that its license applies. Accept the license on that repository and log in with
`hf auth login`. If this account cannot read it, loading raises a `PermissionError` that says what to do. To use your own
copy, pass `image_encoder=...` and the library skips the download.

The auto-loader reads `_class_name` from the repo's `model_index.json` and calls that pipeline's `from_pretrained`,
so it takes the same arguments (`revision`, `cache_dir`, `token`, `local_files_only`, `subfolder`, `dtype`, and so
on). `AutoPipelineForImageTo3D`, `AutoPipelineForTextTo3D`, and `AutoPipelineForMeshTexturing` also check that the
checkpoint is that kind of pipeline. A keyword argument named after a component replaces that component and skips its
download. A locally converted folder already contains the
encoder and needs no extra argument.

The concrete class works too: `Trellis2ImageTo3DPipeline.from_pretrained(...)`. Pipelines are `DiffusionPipeline`s,
so `.to()`, `save_pretrained()`, and `enable_model_cpu_offload()` behave as usual.

## 2. Pass an image

Image pipelines take a file path, a PIL image, a `(H, W, C)` array, a float `(C, H, W)` tensor in `[0, 1]`, or an
`ImageCondition`. A list, or a batched `(B, C, H, W)` tensor, makes a batch. An RGBA image keeps its alpha as the
object mask.

```python
condition = "chair.png"
```

With alpha or a mask, the pipeline crops and recenters the object exactly as upstream does. Inputs without usable
alpha go through each family's upstream background remover, at the same point and resolution as upstream:

| Family | Upstream remover | Here |
|---|---|---|
| TRELLIS, TripoSR, Step1X-3D geometry | rembg (u2net) | automatic, same call (`background` extra) |
| TripoSplat, IDOL, Step1X-3D texture | BiRefNet (`ZhengPeng7/BiRefNet`, MIT) | automatic, ported network and weights |
| TRELLIS.2, Pixal3D | BiRefNet with `briaai/RMBG-2.0` weights (non-commercial, gated) | opt-in, same network with MIT weights |
| TripoSG | `briaai/RMBG-1.4` (non-commercial) | opt-in rembg |
| Hunyuan3D-2.1, Hunyuan3D-Omni | none by default | opt-in rembg (Hunyuan's own call) |

Remover weights download on first use. `pipeline.disable_background_removal()` turns removal off,
`pipeline.enable_background_removal()` turns an opt-in remover on, and `enable_background_removal(remover)` takes any
`PIL -> RGBA PIL` callable. Build an `ImageCondition` yourself to attach a separate `(1, H, W)` `mask` or a `camera`;
`diffusers_3d.as_image_conditions` shows what an input becomes.

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
mesh.save("object.glb")      # GLB, OBJ, PLY, or STL by suffix; written +Y-up; MeshAsset.load reads it back
splats.save("object.ply")    # the 3D Gaussian Splatting PLY layout; GaussianSplatAsset.load reads it back
ovoxel.save("object.npz")    # lossless, readable by upstream TRELLIS.2's tools; OVoxelAsset.load reads it back
torch.save(structure.to("cpu"), "structure.pt")  # any other asset; load with weights_only=False
```

Meshes also have `simplify(face_count)`, `repair()`, and `unwrap_uv()`, which run on the CPU with the `portable`
extra.

For a PBR-textured GLB like upstream's `to_glb`, install CuMesh ([backends.md](backends.md)) and call
`pipeline.postprocess_ovoxel(asset, output_format="textured_mesh")`. Convert the result to `right_handed_y_up` before
exporting it.

## Texture a mesh

Two pipelines paint an existing mesh from a reference image. Both take a `MeshAsset` and an image whose alpha or mask
isolates the object, and return a UV-textured `MeshAsset`. Load them with `AutoPipelineForMeshTexturing`:

```python
from diffusers_3d import AutoPipelineForImageTo3D, AutoPipelineForMeshTexturing

shape = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = shape("chair.png", generator=torch.Generator("cuda").manual_seed(0)).objects[0]

paint = AutoPipelineForMeshTexturing.from_pretrained(
    "suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d", dtype=torch.float16
).to("cuda")
textured = paint(mesh, "chair.png", generator=torch.Generator("cuda").manual_seed(0)).objects[0]
textured.save("textured.glb")
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
| Hunyuan3D-Omni | RGBA plus one of `point_cloud`, `voxels`, `bbox`, `skeleton` (tensors in the output frame) | mesh | `num_inference_steps=50`, `guidance_scale=4.5`, `octree_resolution=512` |
| TripoSR | RGBA | vertex-coloured mesh, Z-up; `pipeline.render_views(output.latents.latents)` renders the NeRF | `mc_resolution`, `mc_threshold`; one feed-forward pass, no sampling |
| TripoSplat | RGBA | Gaussian splats, Z-up | `num_inference_steps=20`, `guidance_scale=3.0`, `num_gaussians` (32768 to 262144); guidance runs sequentially as upstream, `batch_guidance=True` batches it |
| IDOL | RGBA | Gaussian splats, Z-up; predicted input camera in `metadata["camera"]` | `num_inference_steps=50`, `guidance_scale=4.0`, `num_gaussians`; runs TripoSplat's networks |
| Pixal3D | RGBA plus `camera_angle_x` (horizontal FOV, radians) | same as TRELLIS.2 | TRELLIS.2's `formats` and `pipeline_type` (default `"1536_cascade"`); loads its DINOv3 encoder like TRELLIS.2 |

TripoSG, Hunyuan3D-2.1, Hunyuan3D-Omni, SAM 3D Objects, and Hunyuan3D-2.1 Paint warn about their license terms when constructed.

### TRELLIS

`TrellisImageTo3DPipeline` and `TrellisTextTo3DPipeline` (prompts or `TextCondition`) need no extra loading argument.

- `formats` comes from `"sparse_structure"`, `"slat"`, `"gaussian"`, `"mesh"`, and `"radiance_field"`. The default is
  every loaded decoder.
- Sampling uses flat arguments: `sparse_structure_num_inference_steps`, `slat_num_inference_steps`, `guidance_scale`,
  `guidance_interval`, `rescale_t`.
- `"gaussian"` returns a `GaussianSplatAsset`; rendering it needs gsplat.
- `"mesh"` returns a vertex-coloured FlexiCubes `MeshAsset` with the normal map in `extras["normal_map"]`, which
  `save` leaves out of the file.
- `"radiance_field"` returns a `RadianceFieldAsset`. Render it with
  `diffusers_3d.backends.radiance_field.render_radiance_field(asset, cameras)`.
- For a textured GLB, pass the mesh and splats to `diffusers_3d.textured_mesh_from_splats(mesh, gaussians)`.

## Convert a checkpoint yourself

The Hub repos were made with these converters, so converting gives the same result. Each one checks the release's
configuration, loads weights strictly, and writes a Diffusers pipeline folder.

| Family | Command |
|---|---|
| TRELLIS | `diffusers-3d-convert-trellis <release> <out>` (DINOv2 or CLIP at the revision the converter pins) |
| TRELLIS.2 | `diffusers-3d-convert-trellis2 <release> <out> --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m` |
| TripoSG | `diffusers-3d-convert-triposg VAST-AI/TripoSG <out>` |
| Step1X-3D geometry | `diffusers-3d-convert-step1x3d <Step1X-3D-Geometry-1300m folder> <out>` |
| Step1X-3D texture | `diffusers-3d-convert-step1x3d-texture <Step1X-3D-Texture folder> <out>` |
| Hunyuan3D-2.1 shape | `diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 <out>` |
| Hunyuan3D-2.1 Paint | `diffusers-3d-convert-hunyuan3d-paint tencent/Hunyuan3D-2.1 <out>` (`--no-upscaler` skips Real-ESRGAN) |
| SAM 3D Objects | `diffusers-3d-convert-sam3d <checkpoints folder> <out>` |
| Pixal3D | `diffusers-3d-convert-pixal3d <release> <out> --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m --naf-checkpoint-path naf_release.pth` |

`<release>` is the folder holding the release's `pipeline.json`. Encoder paths accept a Hub id or a local folder.
