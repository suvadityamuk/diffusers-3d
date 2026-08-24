# Diffusers 3D

`diffusers-3d` is an object-native companion to
[Diffusers](https://github.com/huggingface/diffusers). It provides tensor-native representations, modular inference
pipelines, optional geometry backends, and recipe-gated fine-tuning for generative 3D models.

The package deliberately does not provide a generic training path for image, video, audio, or other Diffusers
models. A trainable target must be an exact, reviewed object-3D model or pipeline type with a registered
model-specific recipe.

## Status

The package is pre-alpha. Object, loading, and training contracts use schema version `1`; contribution manifests use
schema version `2`. Model integrations and optional compiled backends remain capability-gated.

## First-release readiness

The code release gate requires all of the following before publication:

- [x] Production execution and training registries contain only reviewed Hunyuan3D classes and are immutable.
- [x] Pre-release-only scaffold code and tests are absent from source and built packages.
- [x] Tiny Hunyuan3D model, pipeline, auto-loading, training, and checkpoint round trips pass.
- [x] The Hunyuan3D integration manifest validates with only its expected restricted-license warnings.
- [x] Source and unpacked wheel scans pass with the default release checker.

This checklist covers code readiness only. Selecting a final version and publishing a release are separate maintainer
actions; the package remains at its development version until then.

## Current Hunyuan3D limitations

- The shape VAE is decode-only. Its point-cloud encoder is not included.
- Training supports precomputed shape latents only; surface-sample encoding is not implemented.
- No official approximately 7 GB checkpoint/GPU quality run has been performed, so production mesh quality and
  production-resolution GPU parity are not claimed.
- Hunyuan-derived code and converted checkpoints are governed by the restricted Tencent Hunyuan 3D 2.1 Community
  License Agreement, not the package's Apache-2.0 glue-code license.

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

See the [contribution guide](CONTRIBUTING.md), [lifecycle and review checklists](docs/contributions.md), and
[contribution templates](templates/README.md) for integration and promotion requirements.
