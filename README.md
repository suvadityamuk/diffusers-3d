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

`diffusers-3d` is an object-native companion to [🤗 Diffusers](https://github.com/huggingface/diffusers) for
generative 3D. It gives 3D models the same treatment Diffusers gives image and video models: pretrained weights load
with `from_pretrained`, and pipelines compose from swappable models and schedulers. What is different is the output.
Instead of pixels, pipelines return meshes, Gaussian splats, sparse voxels, and O-Voxels as typed, tensor-native
objects that keep their geometry, materials, and coordinate frames intact.

The repository contains two things:

- **`src/diffusers`** — the Diffusers core that `diffusers-3d` builds on: `ModelMixin`, `DiffusionPipeline`,
  schedulers, attention and embedding layers, modular pipeline blocks, hooks, guiders, and loading utilities.
- **`packages/diffusers-3d`** — the `diffusers_3d` package: 3D object contracts, the model families, and backends.

## What it does

**3D outputs are typed tensors, not files.** A pipeline hands back `MeshAsset` (`vertices`, `faces`, optional
`normals`/`uvs`/`colors` and a tuple of `PBRMaterial`s), `GaussianSplatAsset` (`means`, `log_scales`,
`quaternions_wxyz`, `opacity_logits`, `sh_coefficients`), `SparseVoxelAsset` (`coordinates` + `features`), or
`OVoxelAsset` (active cells plus dual-grid vertices, intersection flags, and per-cell base color, metallic,
roughness, opacity, normals, emissive). Each one is a dataclass of `torch.Tensor`s that checks its own shapes when
constructed, carries a 4x4 `transform` (object to world), a `grid_transform` where a grid is involved, and a
`coordinate_system` enum, and moves with `.to(device, dtype)` like a model would. You decide when to turn it into a
GLB, NPZ, or PLY, and which fields you are willing to lose in the process.

**3D models look like any other Diffusers model.** A converted checkpoint is a directory with `model_index.json`
and one subfolder per component, each holding a `config.json` and a `diffusion_pytorch_model.safetensors`. The flow
models, decoders, and conditioners subclass `ModelMixin`; pipelines subclass `DiffusionPipeline`; schedulers
subclass `SchedulerMixin`. So `from_pretrained`, `save_pretrained`, `.to()`, and CPU offload work without any
3D-specific plumbing, and a pipeline saves back into the same layout it was loaded from.

**Each integration says what it reproduces and how it was checked.** A family README names the upstream repository
and commit it was ported from. The test suite builds tiny versions of every model from `tiny_config()` and compares
outputs and selected gradients against that pinned upstream code on CPU. Where a stage has not been run at production
scale (for TRELLIS.2, the 1024 cascade and the compiled O-Voxel/PBR stages), the pipeline config records it under
`capability_limitations` and the pipeline raises `NotImplementedError` with that reason rather than producing
unverified output.

**Loading from the Hub without running remote code.** `AutoPipelineForImageTo3D.from_pretrained(repo_or_path)`
reads an `object3d_model_index.json` sidecar that lists, for each component, its subfolder and the installed class
that must load it. The loader checks those class names against the package before it downloads anything, downloads
only the component folders the sidecar marks as eligible, and then calls the concrete pipeline class on the local
snapshot. `trust_remote_code=True` is rejected; a reviewed family does not need it.

**Geometry libraries are opt-in and tracked.** CPU tools (trimesh, scikit-image, xatlas), CUDA kernels (spconv,
FlexGEMM, CuMesh, gsplat, the compiled O-Voxel runtime), and research-licensed code (nvdiffrast) each sit behind a
`BackendSpec` in a registry. A spec records the package, the supported device/dtype/Torch combinations, and, for
source builds, the git URL and commit the wheel must have been built from. Nothing is imported at package import
time; you construct a backend such as `TrimeshBackend()` when you need it, it verifies its own install, and
restricted dependencies require an explicit license acknowledgement argument.

## Supported models

| Family | Pipeline | Task | Outputs | Needs a compiled backend |
|---|---|---|---|---|
| [TRELLIS](packages/diffusers-3d/src/diffusers_3d/families/trellis/README.md) | `TrellisImageTo3DPipeline`, `TrellisTextTo3DPipeline` | image → 3D, text → 3D | sparse structure, SLAT, Gaussian splats, FlexiCubes mesh, tri-vector radiance field | rendering the splats (gsplat); simplifying the mesh for the textured GLB (CuMesh) |
| [TRELLIS.2](packages/diffusers-3d/src/diffusers_3d/families/trellis2/README.md) | `Trellis2ImageTo3DPipeline` | image → 3D | sparse structure, shape/texture SLAT, O-Voxel (dual grid + PBR) | meshing and GLB export (O-Voxel runtime) |

Every network in both pipelines runs in plain PyTorch on CPU or GPU. The sparse convolutions, pooling, subdivision,
and windowed attention that upstream implements with `spconv`, FlexGEMM, and `xformers` live in
[`sparse_ops.py`](packages/diffusers-3d/src/diffusers_3d/families/trellis/sparse_ops.py) and are checked against the
pinned upstream code numerically. A textured GLB of a TRELLIS mesh (`TrellisGlbPostprocessFacade`) bakes gsplat renders of
the splats into a base-colour texture, and radiance fields render through a pure-PyTorch volume renderer.

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .                                  # diffusers core
uv pip install -e packages/diffusers-3d               # diffusers_3d
```

Optional extras: `portable` (trimesh, scikit-image, xatlas for CPU mesh I/O), `gaussian` (gsplat). Compiled and
research backends are installed separately; see [backends.md](packages/diffusers-3d/docs/backends.md). Requires
Python 3.10+, PyTorch 2.6+, Transformers 5.5+, Accelerate 1.1+.

## Quickstart

### Converted checkpoints on the Hub

The released checkpoints are already converted into the Diffusers layout and published, so the normal path is to
load them directly:

| Repository | Pipeline | Notes |
|---|---|---|
| [`suvadityamuk/TRELLIS-image-large-diffusers-3d`](https://huggingface.co/suvadityamuk/TRELLIS-image-large-diffusers-3d) | `TrellisImageTo3DPipeline` | complete, including the DINOv2 conditioner |
| [`suvadityamuk/TRELLIS-text-large-diffusers-3d`](https://huggingface.co/suvadityamuk/TRELLIS-text-large-diffusers-3d) | `TrellisTextTo3DPipeline` | complete, including the CLIP conditioner |
| [`suvadityamuk/TRELLIS.2-4B-diffusers-3d`](https://huggingface.co/suvadityamuk/TRELLIS.2-4B-diffusers-3d) | `Trellis2ImageTo3DPipeline` | everything except the DINOv3 conditioner, which is gated under Meta's license and is passed in at load time |

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, Trellis2Dinov3Conditioner

conditioner = Trellis2Dinov3Conditioner.from_dinov3_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", conditioner=conditioner, dtype=torch.bfloat16
).to("cuda")
```

The auto-loader reads the `object3d_model_index.json` sidecar first, checks every declared component class against
the installed package, downloads only those subfolders, and never enables remote code. A component passed as a
keyword argument (`conditioner=...`) is used instead of a subfolder, which is how the gated encoder stays out of the
public repository. `scripts/publish_hub_checkpoints.py` is the script that produced these repositories.

### Converting a checkpoint yourself

Microsoft publishes TRELLIS.2 in its own layout: a `pipeline.json` that names each model and its sampler settings,
and a `ckpts/` folder with one `<name>.json` config and one `<name>.safetensors` file per model
(`sparse_structure_flow_model`, `sparse_structure_decoder`, `shape_slat_flow_model_512`, ...). That is not
something `from_pretrained` can read, and the release does not bundle its image encoder at all; it expects you to
fetch `facebook/dinov3-vitl16-pretrain-lvd1689m` from the Hub, which is gated behind the DINOv3 license.

`diffusers-3d-convert-trellis2` does the one-time translation (this is what produced the Hub repositories above):

1. reads `pipeline.json` and checks it describes the released `Trellis2ImageTo3DPipeline` with the expected
   components and a DINOv3 conditioner;
2. for each supported model, instantiates the matching `diffusers_3d` class from the upstream config, loads the
   upstream safetensors into it with `strict=True` (so any key mismatch fails the conversion), and writes it out as
   a Diffusers subfolder with `config.json` + `diffusion_pytorch_model.safetensors`;
3. copies the DINOv3 weights you downloaded (`--conditioner-path`) into a `conditioner/` subfolder as a
   `Trellis2Dinov3Conditioner`, so the pipeline is self-contained afterwards;
4. moves the sampler defaults and SLAT normalization statistics from `pipeline.json` into the pipeline config;
5. writes `model_index.json` plus the `object3d_model_index.json` sidecar that the auto-loader uses to verify
   component classes.

All eight released networks are converted: the conditioner, the sparse-structure flow and decoder, the 512 and 1024
shape and texture SLAT flows, and the shape and PBR decoders. A `trellis2_conversion.json` report in the output
directory lists what was converted and the upstream commit the conversion targets.

```bash
diffusers-3d-convert-trellis2 /path/to/TRELLIS.2 /path/to/trellis2 \
    --conditioner-path /path/to/dinov3-vitl16-pretrain-lvd1689m
```

Everything after this point, including `save_pretrained`, stays in the Diffusers layout.

### Generate

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "suvadityamuk/TRELLIS.2-4B-diffusers-3d", conditioner=conditioner, dtype=torch.bfloat16
).to("cuda")  # or a locally converted folder

rgba = ...  # (4, H, W) float tensor in [0, 1]; alpha drives foreground cropping
output = pipeline(
    ImageCondition(image=rgba),
    formats=("sparse_structure", "o_voxel"),
    sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5},
    generator=torch.Generator("cuda").manual_seed(0),
)

voxels, ovoxel = output.objects  # SparseVoxelAsset, OVoxelAsset
voxels.coordinates               # (N, 3) int64 grid indices of the occupied coarse cells
ovoxel.active_coordinates        # (M, 3) surface cells after the shape decoder's subdivision
ovoxel.dual_grid_vertex_offsets  # (M, 3) dual-grid vertex per cell, plus split_weights for the tessellation
ovoxel.base_color                # (M, 3) in [0, 1]; also metallic, roughness, opacity, normals, emissive
```

`formats` picks any of `sparse_structure`, `shape_slat`, `texture_slat`, `o_voxel`, and `mesh`; `pipeline_type`
selects the released `512`, `1024`, `1024_cascade`, or `1536_cascade` preset. The full walkthrough — loading,
conditioning, batching, and saving each asset type — is the runnable
[TRELLIS.2 example](packages/diffusers-3d/src/diffusers_3d/families/trellis2/examples/image_to_3d.py). It also has an
offline mode built from tiny components, so the whole API can be exercised on CPU without a checkpoint:

```bash
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/                # all stages
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --sparse-only --output out/  # first stage only
```

## Documentation

| Guide | Contents |
|---|---|
| [Inference](packages/diffusers-3d/docs/inference.md) | Converting checkpoints, loading, conditioning, `formats`, asset types, saving |
| [Backends](packages/diffusers-3d/docs/backends.md) | Portable, accelerated, and research backends; provenance and license gates |
| [Compatibility](packages/diffusers-3d/docs/compatibility.md) | Supported Python/Torch/Transformers/Accelerate ranges and test lanes |
| [Testing](packages/diffusers-3d/docs/testing.md) | Marker policy and exact commands |
| [Contributing](packages/diffusers-3d/CONTRIBUTING.md) | Adding a model family: conversion, parity, registration, backends |
| [Package README](packages/diffusers-3d/README.md) | Status, release gate, licensing, current limitations |

Reference docs for the Diffusers core live under [`docs/source/en`](docs/source/en).

## Design

- `Object3D` is a structural protocol; anything with the right tensors is an object.
- Diffusers owns model loading, scheduling, offloading, and pipeline lifecycle. `diffusers-3d` adds 3D contracts on
  top rather than replacing them.
- Optional CUDA and research-only dependencies are never imported or selected implicitly.
- Every claim of parity names the upstream revision and the test that measured it.

## Contributing

New model families are the main way to contribute. Start with the
[contribution guide](packages/diffusers-3d/CONTRIBUTING.md) and the
[templates](packages/diffusers-3d/templates/README.md). Run `make style` and `make quality` before opening a PR;
`make test-3d` runs the `diffusers_3d` suite.

## License

The Diffusers core is Apache-2.0. The `diffusers-3d` package uses an `Apache-2.0 AND MIT` aggregate: package-owned
glue is Apache-2.0, while TRELLIS- and TRELLIS.2-derived family code retains its MIT terms. Model weights,
DINOv3 conditioner weights, and research backends such as nvdiffrast carry their own licenses and are not
redistributed.
