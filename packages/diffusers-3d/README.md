# Diffusers 3D

`diffusers-3d` is an object-native companion to
[Diffusers](https://github.com/huggingface/diffusers). It provides tensor-native representations, modular inference
pipelines, optional geometry backends, and recipe-gated fine-tuning for generative 3D models.

The package deliberately does not provide a generic training path for image, video, audio, or other Diffusers
models. A trainable target must be an exact, reviewed object-3D model or pipeline type with a registered
model-specific recipe.

## Status

The package is pre-alpha. Public contracts use schema version `1`; model integrations and optional compiled
backends remain capability-gated.

## Design principles

- `Object3D` is a structural data protocol. Training authorization is nominal and registry-based.
- Meshes, Gaussian splats, sparse voxels, and O-Voxels remain tensor-native and preserve representation-specific
  channels.
- Diffusers owns model loading, scheduling, offloading, and pipeline lifecycle behavior.
- Optional CUDA and research-only dependencies are never imported or selected implicitly.
- Model stages have separate training recipes, objectives, component policies, and checkpoint manifests.

## Installation

Install the package from this repository:

```bash
pip install -e packages/diffusers-3d
```

Portable mesh processing is optional:

```bash
pip install -e "packages/diffusers-3d[portable]"
```

Source-built or CUDA-specific dependencies such as nvdiffrast, spconv, FlexGEMM, CuMesh, and O-Voxel require an
explicit backend installation. See [docs/backends.md](docs/backends.md).

## Contribution levels

1. Experimental Hub blocks use Modular Diffusers remote code and are inference-only.
2. Reviewed package integrations have exact loading registrations, manifests, parity tests, and dependency review.
   Training requires a separately reviewed recipe.
3. Stable dependency-light primitives may be proposed upstream to Diffusers. Upstream availability alone does not
   make a component trainable through this package.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the integration and promotion requirements.
