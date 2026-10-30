<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center">diffusers-3d</h1>

<p align="center">
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
</p>

`diffusers-3d` runs generative 3D models with [🤗 Diffusers](https://github.com/huggingface/diffusers). Checkpoints load
with `from_pretrained`, pipelines are ordinary `DiffusionPipeline`s, and every network runs in plain PyTorch on CPU or
GPU. Pipelines return typed, tensor-native 3D objects (meshes, Gaussian splats, sparse voxels, O-Voxels) instead of
files, so you choose when and how to export them.

The repository has two parts:

- **`src/diffusers`**: the Diffusers core the package builds on.
- **`packages/diffusers-3d`**: the `diffusers_3d` package with the 3D object types, model families, and backends.

## Supported models

Converted checkpoints are on the Hub for most families. Each repository ships the family's license and notice files.

| Family | Pipeline | Task | Hub checkpoint |
|---|---|---|---|
| [TRELLIS](packages/diffusers-3d/src/diffusers_3d/families/trellis/README.md) | `TrellisImageTo3DPipeline` | image → splats, mesh, radiance field | [`TRELLIS-image-large-diffusers-3d`](https://huggingface.co/suvadityamuk/TRELLIS-image-large-diffusers-3d) |
| | `TrellisTextTo3DPipeline` | text → splats, mesh, radiance field | [`TRELLIS-text-large-diffusers-3d`](https://huggingface.co/suvadityamuk/TRELLIS-text-large-diffusers-3d) |
| [TRELLIS.2](packages/diffusers-3d/src/diffusers_3d/families/trellis2/README.md) | `Trellis2ImageTo3DPipeline` | image → O-Voxel with PBR, mesh | [`TRELLIS.2-4B-diffusers-3d`](https://huggingface.co/suvadityamuk/TRELLIS.2-4B-diffusers-3d)¹ |
| [TripoSG](packages/diffusers-3d/src/diffusers_3d/families/triposg/README.md) | `TripoSGImageTo3DPipeline` | image → mesh | [`TripoSG-diffusers-3d`](https://huggingface.co/suvadityamuk/TripoSG-diffusers-3d) |
| [Step1X-3D](packages/diffusers-3d/src/diffusers_3d/families/step1x3d/README.md) | `Step1X3DImageTo3DPipeline` | image → mesh | [`Step1X-3D-Geometry-1300m-diffusers-3d`](https://huggingface.co/suvadityamuk/Step1X-3D-Geometry-1300m-diffusers-3d) |
| | `Step1X3DMeshTexturePipeline` | mesh + image → textured mesh | [`Step1X-3D-Texture-diffusers-3d`](https://huggingface.co/suvadityamuk/Step1X-3D-Texture-diffusers-3d) |
| [Hunyuan3D-2.1](packages/diffusers-3d/src/diffusers_3d/families/hunyuan3d/README.md) | `Hunyuan3DImageTo3DPipeline` | image → mesh | [`Hunyuan3D-2.1-diffusers-3d`](https://huggingface.co/suvadityamuk/Hunyuan3D-2.1-diffusers-3d) |
| | `Hunyuan3DPaintPipeline` | mesh + image → PBR-textured mesh | [`Hunyuan3D-2.1-Paint-diffusers-3d`](https://huggingface.co/suvadityamuk/Hunyuan3D-2.1-Paint-diffusers-3d) |
| [SAM 3D Objects](packages/diffusers-3d/src/diffusers_3d/families/sam3d_objects/README.md) | `Sam3DObjectsPipeline` | masked image → posed splats, mesh | [`sam-3d-objects-diffusers-3d`](https://huggingface.co/suvadityamuk/sam-3d-objects-diffusers-3d) |
| [Pixal3D](packages/diffusers-3d/src/diffusers_3d/families/pixal3d/README.md) | `Pixal3DImageTo3DPipeline` | image + field of view → O-Voxel, mesh | [`Pixal3D-diffusers-3d`](https://huggingface.co/suvadityamuk/Pixal3D-diffusers-3d)¹ |
| [Hunyuan3D-Omni](packages/diffusers-3d/src/diffusers_3d/families/hunyuan3d_omni/README.md) | `Hunyuan3DOmniImageTo3DPipeline` | image + point cloud, voxels, bbox, or skeleton → mesh | convert locally² |
| [TripoSR](packages/diffusers-3d/src/diffusers_3d/families/triposr/README.md) | `TripoSRImageTo3DPipeline` | image → vertex-coloured mesh | convert locally² |
| [TripoSplat](packages/diffusers-3d/src/diffusers_3d/families/triposplat/README.md) | `TripoSplatImageTo3DPipeline` | image → Gaussian splats | convert locally² |
| [IDOL](packages/diffusers-3d/src/diffusers_3d/families/idol/README.md) | `IdolImageTo3DPipeline` | image → Gaussian splats and input camera | convert locally² |

¹ The DINOv3 image encoder is gated under Meta's DINOv3 License and is not redistributed. Accept the license on
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and pass
the encoder in when loading (see below).

² Not published yet. Convert the upstream release with `diffusers-3d-convert-hunyuan3d-omni`,
`diffusers-3d-convert-triposr`, `diffusers-3d-convert-triposplat`, or `diffusers-3d-convert-idol`; see each family README.

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .                                  # Diffusers core
uv pip install -e "packages/diffusers-3d[portable]"  # diffusers_3d with CPU mesh tools
```

Requires Python 3.10+, PyTorch 2.6+, Transformers 5.5+, and Accelerate 1.1+. The `portable` extra (trimesh,
scikit-image, xatlas, fast-simplification) is needed for marching cubes, UV unwrapping, and GLB export. Other extras
are `gaussian` (gsplat) and `lora` (PEFT). Compiled CUDA backends such as CuMesh are installed separately; see
[backends.md](packages/diffusers-3d/docs/backends.md).

## Quickstart

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/TripoSG-diffusers-3d").to("cuda")
pipeline.enable_background_removal()  # only for photos without alpha; needs the `background` extra
mesh = pipeline("chair.jpg", generator=torch.Generator("cuda").manual_seed(0)).objects[0]
mesh.save("mesh.glb")
```

TRELLIS.2 and Pixal3D take the gated DINOv3 encoder as an argument:

```python
from transformers import DINOv3ViTModel

image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", image_encoder=image_encoder, dtype=torch.bfloat16
).to("cuda")
output = pipeline("chair.png", formats=("o_voxel", "mesh"))
```

The texture stages load with `AutoPipelineForMeshTexturing` and paint any mesh. The
[inference guide](packages/diffusers-3d/docs/inference.md) covers conditioning, `formats`, saving each asset type,
texturing, LoRA, and converting checkpoints yourself.

## How it works

- **Typed outputs.** `MeshAsset`, `GaussianSplatAsset`, `SparseVoxelAsset`, and `OVoxelAsset` are dataclasses of
  tensors. They validate their shapes, carry a transform and coordinate system, and move with `.to(device, dtype)`.
- **Standard Diffusers layout.** A converted checkpoint is a `model_index.json` plus one folder per component. Models
  subclass `ModelMixin`, encoders are stock Transformers models, and schedulers are Diffusers schedulers, so
  `save_pretrained`, `.to()`, and CPU offload work as usual.
- **Installed code only.** `AutoPipelineFor3D` maps the `_class_name` in `model_index.json` to a pipeline class in
  this package, then calls its `from_pretrained`. It never runs code from the Hub.
- **Opt-in backends.** Optional libraries (trimesh, scikit-image, xatlas, gsplat, CuMesh) are never imported at package
  import time. Each adapter imports its library when constructed and tells you how to install it if it is missing.
- **Checked against upstream.** Each family README names the upstream commit it ports. Tests compare tiny random-weight
  models against that pinned code on CPU. GPU runs on the released weights are recorded in
  [compatibility.md](packages/diffusers-3d/docs/compatibility.md).

## Documentation

| Guide | Contents |
|---|---|
| [Inference](packages/diffusers-3d/docs/inference.md) | Loading, conditioning, outputs, saving, texturing, LoRA, conversion |
| [Backends](packages/diffusers-3d/docs/backends.md) | Optional geometry and rendering backends and how to install them |
| [Compatibility](packages/diffusers-3d/docs/compatibility.md) | Supported versions and what has been verified, per backend and family |
| [Testing](packages/diffusers-3d/docs/testing.md) | Test markers, CI lanes, and commands |
| [Contributing](packages/diffusers-3d/CONTRIBUTING.md) | Adding a model family |

## Contributing

New model families are the main way to contribute. Start with the
[contribution guide](packages/diffusers-3d/CONTRIBUTING.md). Run `make style` and `make quality` before opening a PR,
and `make test-3d` for the package tests.

## License

The Diffusers core is Apache-2.0. The `diffusers-3d` package combines Apache-2.0 glue code with ported code that keeps
its upstream license: MIT (TRELLIS, TRELLIS.2, TripoSG, TripoSR, TripoSplat, Pixal3D, and Step1X-3D's texture sync),
Apache-2.0 (IDOL), Meta's SAM License (SAM 3D Objects), and the Tencent Hunyuan 3D 2.1 and Hunyuan 3D Omni Community
Licenses (Hunyuan3D-2.1 and Hunyuan3D-Omni, not valid in the EU, UK, or South Korea). Model weights keep their own
licenses. Pipelines with restrictive terms warn when constructed. Gated DINOv3 weights are never redistributed.
