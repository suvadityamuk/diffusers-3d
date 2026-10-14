# Inference with diffusers-3d

This page shows what running a generative 3D model looks like today. Sections 2–7 use TRELLIS.2 as the running
example; every other family loads, conditions, and saves the same way, with the arguments listed under
[Other families](#other-families). [Section 8](#8-texture-a-mesh) textures a mesh with Hunyuan3D-2.1 Paint or Step1X-3D.
A complete, runnable version of the TRELLIS.2 walkthrough lives in
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
| `suvadityamuk/TRELLIS-image-large-diffusers-3d` | `TrellisImageTo3DPipeline` | every component, DINOv2 image encoder included |
| `suvadityamuk/TRELLIS-text-large-diffusers-3d` | `TrellisTextTo3DPipeline` | every component, CLIP text encoder and tokenizer included |
| `suvadityamuk/TRELLIS.2-4B-diffusers-3d` | `Trellis2ImageTo3DPipeline` | every component except the gated DINOv3 image encoder |
| `suvadityamuk/TripoSG-diffusers-3d` | `TripoSGImageTo3DPipeline` | every component, DINOv2 image encoder included |
| `suvadityamuk/Step1X-3D-Geometry-1300m-diffusers-3d` | `Step1X3DImageTo3DPipeline` | every geometry-stage component, DINOv2 image encoder included |
| `suvadityamuk/Hunyuan3D-2.1-diffusers-3d` | `Hunyuan3DImageTo3DPipeline` | every shape-stage component, DINOv2 image encoder included; Tencent Hunyuan 3D 2.1 Community License |
| `suvadityamuk/sam-3d-objects-diffusers-3d` | `Sam3DObjectsPipeline` | every component, DINOv2 image encoder included; SAM License |
| `suvadityamuk/Pixal3D-diffusers-3d` | `Pixal3DImageTo3DPipeline` | every component except the gated DINOv3 image encoder |
| `suvadityamuk/Step1X-3D-Texture-diffusers-3d` | `Step1X3DMeshTexturePipeline` | every texture-stage component, SDXL base included; CreativeML Open RAIL++-M |
| `suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d` | `Hunyuan3DPaintPipeline` | every Paint component, DINOv2-giant and Real-ESRGAN included; Tencent Hunyuan 3D 2.1 Community License |

Each repository ships its family's license and notice files, plus the license texts of any third-party base weights
it redistributes; read them before use.

`scripts/publish_hub_checkpoints.py` produced the repositories with the packaged converters, so converting yourself
gives the same result. Every converter writes a Diffusers pipeline folder from an official release:

| Family | Command |
|---|---|
| TRELLIS | `diffusers-3d-convert-trellis <release> <out> --encoder-path facebook/dinov2-with-registers-large` (or `openai/clip-vit-large-patch14` for text) |
| TRELLIS.2 | `diffusers-3d-convert-trellis2 <release> <out> --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m` |
| TripoSG | `diffusers-3d-convert-triposg VAST-AI/TripoSG <out>` |
| Step1X-3D geometry | `diffusers-3d-convert-step1x3d <Step1X-3D-Geometry-1300m folder> <out>` |
| Step1X-3D texture | `diffusers-3d-convert-step1x3d-texture <Step1X-3D-Texture folder> <out>` (fetches the SDXL base and VAE) |
| Hunyuan3D-2.1 shape | `diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 <out>` |
| Hunyuan3D-2.1 Paint | `diffusers-3d-convert-hunyuan3d-paint tencent/Hunyuan3D-2.1 <out>` (fetches DINOv2-giant and Real-ESRGAN; `--no-upscaler` skips it) |
| SAM 3D Objects | `diffusers-3d-convert-sam3d <sam-3d-objects checkpoints/ folder> <out>` |
| Pixal3D | `diffusers-3d-convert-pixal3d <release> <out> --image-encoder-path facebook/dinov3-vitl16-pretrain-lvd1689m --naf-checkpoint-path naf_release.pth` |

Each one validates the release's configuration, rejects missing or unexpected weights, and loads image and text
encoders through Transformers' own `from_pretrained`, so they stay stock Transformers components. For TRELLIS.2, point
the CLI at the release's `pipeline.json` directory and at the DINOv3 image encoder, a downloaded folder or its Hub id
(the production weights are gated on the Hub under the DINOv3 License and are not redistributed):

```bash
diffusers-3d-convert-trellis2 /path/to/TRELLIS.2-release /path/to/trellis2 \
    --image-encoder-path /path/to/dinov3-vitl16-pretrain-lvd1689m
```

The output is a standard Diffusers pipeline directory (`model_index.json` plus one subfolder per component) with an
extra `object3d_model_index.json` sidecar. The sidecar records the exact class of each component and which ones are
eligible for automatic loading. Every released component is converted: the DINOv3 image encoder, the sparse-structure
flow and decoder, the 512 and 1024 shape and texture SLAT flows, and the shape (dual-grid) and PBR decoders. A
`trellis2_conversion.json` report in the output directory lists what was written.

## 2. Load a pipeline

```python
import torch
from transformers import DINOv3ViTModel

from diffusers_3d import AutoPipelineForImageTo3D

# The published TRELLIS.2 repository leaves out the gated DINOv3 image encoder; accept its license on the Hub and pass it in.
image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", image_encoder=image_encoder, dtype=torch.bfloat16
)
pipeline = pipeline.to("cuda")
# A locally converted folder loads the same way and needs no image_encoder argument:
# pipeline = AutoPipelineForImageTo3D.from_pretrained("/path/to/trellis2", dtype=torch.bfloat16)
```

`AutoPipelineForImageTo3D` reads the sidecar, checks every component class against the installed package, downloads
only eligible component folders (for Hub IDs), and instantiates the concrete class with remote code disabled.
`revision`, `cache_dir`, `token`, `local_files_only`, and `subfolder` are accepted. Keyword arguments named after
declared components (`image_encoder=...`) supply that component as an object: its subfolder is neither downloaded nor
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
uint8, crop to `alpha > 0.8 * 255`, recenter, premultiply on black, LANCZOS-resize to the pipeline `image_size`. Plain
RGB is treated as an already background-removed frame. The pipeline never runs a background remover on your behalf.

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
  `("sparse_structure",)` otherwise. Every format, `"mesh"` included, runs in plain PyTorch on CPU or GPU.
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
| `mesh` | `MeshAsset` | The O-Voxel's dual-grid surface, one vertex per voxel, coloured by base colour and opacity. |

## 6. Load LoRA weights

Each pipeline loads LoRA weights saved in the Diffusers format into its flow models. Install PEFT first with
`pip install "peft>=0.17.0"` (or the `lora` extra: `pip install -e "packages/diffusers-3d[lora]"`); without it,
loading raises an `ImportError` with that command.

```python
pipeline.load_lora_weights("user/trellis2-style-lora", adapter_name="style")
pipeline.set_adapters("style", adapter_weights=0.8)  # scale it; fuse_lora() / unfuse_lora() also work
pipeline.unload_lora_weights()
```

A Diffusers-format checkpoint is a safetensors file whose keys start with the component they adapt, for example
`sparse_structure_flow_model.blocks.0.self_attn.to_qkv.lora_A.weight`. TRELLIS accepts `sparse_structure_flow_model`
and `slat_flow_model`; TRELLIS.2 accepts `sparse_structure_flow_model` and the four shape and texture SLAT flow
models. Checkpoints with any other key prefix are rejected rather than converted.

## 7. Save assets

There is no single 3D file format, so serialization is per representation:

```python
from diffusers_3d import OVoxelAsset, MeshAsset, SparseVoxelAsset, TrimeshBackend, write_ovoxel_npz

if isinstance(asset, SparseVoxelAsset):
    torch.save(asset.to("cpu"), "structure.pt")          # tensors + metadata, load with weights_only=False
elif isinstance(asset, OVoxelAsset):
    write_ovoxel_npz("object.npz", asset)                 # pure NumPy, readable by the official TRELLIS.2 tools
elif isinstance(asset, MeshAsset):
    asset = asset.to_coordinate_system("right_handed_y_up")  # TRELLIS families are Z-up; GLB is Y-up
    TrimeshBackend().export_mesh(asset, "object.glb")     # needs the `portable` extra
```

For a PBR-textured mesh like upstream's `to_glb`, install CuMesh, which compiles against your PyTorch and CUDA (see
[backends.md](backends.md)):

```bash
pip install --no-build-isolation -r packages/diffusers-3d/requirements/backends/cumesh.txt
```

Then call `pipeline.postprocess_ovoxel(asset, output_format="textured_mesh")`; convert the result with
`to_coordinate_system("right_handed_y_up")` before exporting it. `.vxz` and voxel rendering still need the compiled
O-Voxel runtime and its research-licensed nvdiffrast dependency, which are never imported implicitly.

## 8. Texture a mesh

Two pipelines paint an existing mesh from a reference image. Both take a `MeshAsset`, for example a shape pipeline's
output, and an `ImageCondition` whose alpha or `mask` isolates the object, and return a UV-textured `MeshAsset`. Load
them with `AutoPipelineFor3D` (task `mesh-texturing`) or their concrete classes:

```python
import torch
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition, TrimeshBackend

shape = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = shape(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

paint = AutoPipelineFor3D.from_pretrained("suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d", dtype=torch.float16).to("cuda")
textured = paint(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
TrimeshBackend().export_mesh(textured.to("cpu"), "textured.glb")
```

- `Hunyuan3DPaintPipeline` (Hunyuan3D-2.1 Paint) simplifies the mesh to `max_faces=40000` (needs
  `fast-simplification` from the `portable` extra; `max_faces=None` keeps it), unwraps it with xatlas, picks up to
  `max_num_views` cameras, generates albedo and metallic-roughness views with 15 UniPC steps, upscales them with
  Real-ESRGAN, and bakes a base-colour plus metallic-roughness `PBRMaterial`.
- `Step1X3DMeshTexturePipeline` (Step1X-3D) generates six orthographic views with SDXL, the released multi-view
  adapter, and a T2I-Adapter fed position and normal renders, keeps them consistent by denoising through a shared
  latent texture for the first half of the steps (`texture_sync_ratio`), and bakes a base-colour texture.
  `output.previews` holds the six generated views.

Both rasterize, bake, and inpaint in plain PyTorch through `diffusers_3d.backends` (`mesh_rasterizer` and
`multiview_baking`) instead of upstream's CUDA rasterizers, nvdiffrast, pytorch3d, and OpenCV. Neither runs a
background remover: pass RGBA or a mask. The texture-gap fill differs slightly from OpenCV's inpainting; see the
family READMEs for measured differences.

## Other families

Every family's converted checkpoint loads with `AutoPipelineForImageTo3D.from_pretrained` and returns tensor-native
assets in `output.objects`. Each family README lists its components, numerics, and differences from upstream.

- **TripoSG** (`TripoSGImageTo3DPipeline`): RGBA in, an untextured `MeshAsset` (right-handed Y-up) out.
  `num_inference_steps=50`, `guidance_scale=7.0`, and `dense_octree_depth` / `hierarchical_octree_depth` control the
  marching-cubes grid.
- **Step1X-3D** (`Step1X3DImageTo3DPipeline`): RGBA in, an untextured mesh out; `num_inference_steps=50`,
  `guidance_scale=7.5`, `octree_resolution=384`.
- **Hunyuan3D-2.1** (`Hunyuan3DImageTo3DPipeline`): RGBA in, an untextured mesh out; `num_inference_steps=50`,
  `guidance_scale=5.0`, `octree_resolution=384`. Load in float16 as released.
- **SAM 3D Objects** (`Sam3DObjectsPipeline`): an RGB image plus a `mask` selecting the object, and optionally a
  camera-space `pointmap`. `formats` picks from `"sparse_structure"`, `"gaussian"`, and `"mesh"`; the object comes back
  posed in the input camera's frame.
- **Pixal3D** (`Pixal3DImageTo3DPipeline`): RGBA in plus `camera_angle_x`, the input camera's horizontal field of view
  in radians. It shares TRELLIS.2's `formats` and `pipeline_type` (default `"1536_cascade"`) and, like TRELLIS.2, takes
  the gated DINOv3 image encoder as an `image_encoder=` argument when loading from the Hub.

TripoSG, Hunyuan3D-2.1, SAM 3D Objects, and the Hunyuan3D-2.1 Paint stage warn on construction about their
licenses' territory or use restrictions.

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
without any extra argument. Their encoders are stock Transformers components: `image_encoder` is a
`Dinov2WithRegistersModel` and the text pipeline has a `CLIPTextModel` `text_encoder` and a `CLIPTokenizer` `tokenizer`.
To convert official checkpoints yourself use `diffusers-3d-convert-trellis`. Its `--encoder-path` is a Hub id or folder of
the released encoder: `facebook/dinov2-with-registers-large` (the released `dinov2_vitl14_reg`), or
`openai/clip-vit-large-patch14` for the text pipelines:

```bash
hf download microsoft/TRELLIS-image-large --local-dir /path/to/TRELLIS-image-large
diffusers-3d-convert-trellis /path/to/TRELLIS-image-large /path/to/trellis-diffusers \
    --encoder-path facebook/dinov2-with-registers-large
```

## What is and is not covered

Every network in every family runs in plain PyTorch on any device, and so do mesh extraction, rasterization,
texture baking, and inpainting. For TRELLIS and TRELLIS.2 that covers the image and text encoders, sparse-structure
flows and decoders, SLAT flows, the Gaussian, mesh, and radiance-field decoders, and the shape and PBR decoders. Sparse convolutions, pooling, subdivision, and
windowed attention are implemented in `families/trellis/sparse_ops.py` and checked numerically against the pinned
upstream code with tiny weights, with the upstream CUDA kernels (`spconv`, FlexGEMM, `xformers`) replaced by dense
PyTorch equivalents in the test. Full-resolution generation on the released weights and the compiled mesh, render, and
PBR GLB paths are not in the CPU CI; the [GPU smoke workflow](../../../.github/workflows/diffusers_3d_gpu_smoke.yml)
runs them on a Hugging Face Jobs A100 on demand and weekly (`scripts/gpu_smoke.py`), and the numbers from the last
manual run are recorded in [compatibility.md](compatibility.md). The other families and both texture stages were run
by hand on the same hardware against their released checkpoints; that page records those runs too. See it and the
family READMEs for the exact evidence behind each claim.
