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
with `from_pretrained`, pipelines compose from swappable models and schedulers, and fine-tuning runs through a
trainer built on Accelerate. What is different is the output. Instead of pixels, pipelines return meshes, Gaussian
splats, sparse voxels, and O-Voxels as typed, tensor-native objects that keep their geometry, materials, and
coordinate frames intact.

The repository contains two things:

- **`src/diffusers`** — the Diffusers core that `diffusers-3d` builds on: `ModelMixin`, `DiffusionPipeline`,
  schedulers, attention and embedding layers, modular pipeline blocks, hooks, guiders, and loading utilities.
- **`packages/diffusers-3d`** — the `diffusers_3d` package: 3D object contracts, the model families, backends, and
  the training stack.

## What it does

**Tensor-native 3D objects.** `MeshAsset`, `GaussianSplatAsset`, `SparseVoxelAsset`, and `OVoxelAsset` are
dataclasses of tensors with explicit `transform`, `grid_transform`, and `coordinate_system` fields, validated on
construction and movable with `.to(device, dtype)`. Representation-specific channels (PBR materials, dual-grid
vertices, spherical harmonics) are first-class fields, not lossy conversions. Every pipeline returns an
`Object3DPipelineOutput` whose first value is always a tuple of these objects.

**Diffusers-style pipelines for 3D models.** Model families are integrated as ordinary Diffusers models and
pipelines. Components live in subfolders, configs are JSON, weights are safetensors, and `save_pretrained` /
`from_pretrained` round-trip. Family conversion CLIs turn official releases into this layout once.

**Reviewed, verifiable integrations.** Each family records the exact upstream revision it reproduces and ships tiny
CPU parity tests against it. Pipeline configs and asset metadata state what has been measured and what has not, so
"supported" always has a specific meaning.

**Secure Hub auto-loading.** `AutoPipelineForImageTo3D.from_pretrained(repo_or_path)` resolves the concrete
pipeline from a schema-v2 sidecar that names every component's class, validates identities before downloading, fetches
only eligible components, and never enables remote code.

**Optional geometry backends, never implicit.** Portable CPU tooling (trimesh, scikit-image, xatlas), accelerated
CUDA kernels (spconv, FlexGEMM, CuMesh, gsplat, O-Voxel), and research-licensed dependencies (nvdiffrast) are
discovered through a registry with provenance and license checks. Nothing is imported or selected silently; a backend
is chosen explicitly and reports its own status.

**Recipe-gated fine-tuning.** Training goes through per-stage `TrainingRecipe3D` classes that fix the objective,
the example type, and which components may be trained. `Object3DTrainer` handles Accelerate, optimizer, scheduling,
and exact-resume checkpoints with a self-describing manifest. Generic, unreviewed targets are rejected.

## Supported models

| Family | Pipeline | Task | Reviewed output | Experimental stages |
|---|---|---|---|---|
| [TRELLIS](packages/diffusers-3d/src/diffusers_3d/families/trellis/README.md) | `TrellisImageTo3DPipeline` | image → 3D | sparse structure | SLAT, Gaussian splats |
| [TRELLIS.2](packages/diffusers-3d/src/diffusers_3d/families/trellis2/README.md) | `Trellis2ImageTo3DPipeline` | image → 3D | sparse structure | shape/texture SLAT, O-Voxel, PBR mesh |

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .                                  # diffusers core
uv pip install -e "packages/diffusers-3d[training]"  # diffusers_3d + Accelerate/PEFT training stack
```

Optional extras: `portable` (trimesh, scikit-image, xatlas for CPU mesh I/O), `gaussian` (gsplat). Compiled and
research backends are installed separately; see [backends.md](packages/diffusers-3d/docs/backends.md). Requires
Python 3.10+, PyTorch 2.6+, Transformers 5.5+, Accelerate 1.1+.

## Quickstart

Convert an official TRELLIS.2 release once, then generate:

```bash
diffusers-3d-convert-trellis2 /path/to/TRELLIS.2 /path/to/trellis2 --conditioner-path /path/to/dinov3
```

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("/path/to/trellis2").to("cuda")

rgba = ...  # (4, H, W) float tensor in [0, 1]; alpha drives foreground cropping
output = pipeline(
    ImageCondition(image=rgba),
    formats=("sparse_structure",),
    sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5},
    generator=torch.Generator("cuda").manual_seed(0),
)

voxels = output.objects[0]      # SparseVoxelAsset
voxels.coordinates              # (N, 3) int64 grid indices
voxels.features                 # (N, C) per-voxel channels
voxels.metadata                 # {"family": "trellis2", "representation": "sparse_structure", "resolution": 32, ...}
```

The full walkthrough — loading, conditioning, batching, experimental stages, and saving each asset type — is the
runnable [TRELLIS.2 example](packages/diffusers-3d/src/diffusers_3d/families/trellis2/examples/image_to_3d.py).
It also has an offline mode built from tiny components, so the whole API can be exercised on CPU without a
checkpoint:

```bash
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/
python -m diffusers_3d.families.trellis2.examples.image_to_3d --experimental --output out/   # + SLAT and O-Voxel stages
```

## Fine-tuning

```python
from diffusers_3d import FullFineTune, Object3DTrainer, TrainingConfig3D, Trellis2SparseStructureFlowRecipe

recipe = Trellis2SparseStructureFlowRecipe(pipeline)
trainer = Object3DTrainer(
    recipe,
    dataset,                                              # yields Trellis2SparseStructureExample
    FullFineTune(("sparse_structure_flow_model",)),
    TrainingConfig3D(base_model="/path/to/trellis2", dataset_fingerprint="latents-v1",
                     output_dir="runs/ss", train_batch_size=8, max_train_steps=2000, mixed_precision="bf16"),
)
summary = trainer.prepare().train()
trainer.save_checkpoint()
pipeline.save_pretrained("checkpoints/trellis2-finetuned")
```

See [finetuning.md](packages/diffusers-3d/docs/finetuning.md) for data preparation, component policies, checkpoints
and resume, and [inference.md](packages/diffusers-3d/docs/inference.md) for the complete inference contract.

## Documentation

| Guide | Contents |
|---|---|
| [Inference](packages/diffusers-3d/docs/inference.md) | Converting checkpoints, loading, conditioning, `formats`, asset types, saving |
| [Fine-tuning](packages/diffusers-3d/docs/finetuning.md) | Recipes, datasets, strategies, `TrainingConfig3D`, checkpoints, reuse |
| [Backends](packages/diffusers-3d/docs/backends.md) | Portable, accelerated, and research backends; provenance and license gates |
| [Compatibility](packages/diffusers-3d/docs/compatibility.md) | Supported Python/Torch/Transformers/Accelerate ranges and test lanes |
| [Testing](packages/diffusers-3d/docs/testing.md) | Marker policy and exact commands |
| [Contributions](packages/diffusers-3d/docs/contributions.md) | Experimental → reviewed → upstream lifecycle |
| [Package README](packages/diffusers-3d/README.md) | Status, release gate, licensing, current limitations |

Reference docs for the Diffusers core live under [`docs/source/en`](docs/source/en).

## Design

- `Object3D` is a structural protocol; anything with the right tensors is an object. Training authorization is
  nominal and registry-based.
- Diffusers owns model loading, scheduling, offloading, and pipeline lifecycle. `diffusers-3d` adds 3D contracts on
  top rather than replacing them.
- Model stages have separate recipes, objectives, component policies, and checkpoint manifests.
- Optional CUDA and research-only dependencies are never imported or selected implicitly.
- Every claim of parity names the upstream revision and the test that measured it.

## Contributing

Integrations move through three levels: experimental Hub blocks using Modular Diffusers remote code, reviewed
package families with exact registrations and parity tests, and stable primitives proposed upstream to Diffusers.
Start with the [contribution guide](packages/diffusers-3d/CONTRIBUTING.md), the
[lifecycle checklists](packages/diffusers-3d/docs/contributions.md), and the
[templates](packages/diffusers-3d/templates/README.md). Run `make style` and `make quality` before opening a PR;
`make test-3d` runs the `diffusers_3d` suite.

## License

The Diffusers core is Apache-2.0. The `diffusers-3d` package uses an `Apache-2.0 AND MIT` aggregate: package-owned
glue is Apache-2.0, while TRELLIS- and TRELLIS.2-derived family code retains its MIT terms. Model weights,
DINOv3 conditioner weights, and research backends such as nvdiffrast carry their own licenses and are not
redistributed.
