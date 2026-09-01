# Diffusers 3D

`diffusers-3d` is an object-native companion to
[Diffusers](https://github.com/huggingface/diffusers). It provides tensor-native representations, modular inference
pipelines, optional geometry backends, and recipe-gated fine-tuning for generative 3D models.

The package deliberately does not provide a generic training path for image, video, audio, or other Diffusers
models. A trainable target must be an exact, reviewed object-3D model or pipeline type with a registered
model-specific recipe.

## Status

The package is pre-alpha. Loading metadata uses schema version `2`, training manifests use schema version `3`, and
contribution manifests use schema version `2`. Model integrations and optional compiled backends remain
capability-gated. The reviewed model families are TRELLIS and TRELLIS.2, including TRELLIS.2's optional O-Voxel
paths.

## First-release readiness

The code release gate requires all of the following before publication:

- [x] Production execution and training registries contain only reviewed TRELLIS and TRELLIS.2 classes and are
  immutable.
- [x] Pre-release-only scaffold code and tests are absent from source and built packages.
- [x] Tiny TRELLIS and TRELLIS.2 model, pipeline, auto-loading, training, and checkpoint round trips pass.
- [x] Every reviewed family schema-v2 manifest validates with only its declared license/backend warnings.
- [x] Source, unpacked wheel, and unpacked sdist scans pass with the default release checker.

This checklist covers code readiness only. Selecting a final version and publishing a release are separate maintainer
actions; the package remains at its development version until then.

The distribution uses an `Apache-2.0 AND MIT` aggregate license expression. Package-owned glue is Apache-2.0, while
the TRELLIS and TRELLIS.2-derived family code retains MIT terms. Wheel and sdist metadata ship the local Apache
license and both applicable MIT licenses and notices. Those terms apply to their respective files; the aggregate
expression does not relicense any family code or model artifact.

## Current TRELLIS.2 limitations

- The reviewed contract ends at CPU-capable sparse-structure output. Tiny SLAT and O-Voxel stages are experimental.
- No full 4B checkpoint, 1024 cascade, production GPU quality, compiled O-Voxel mesh/render, or PBR GLB run has been
  performed.
- O-Voxel schema/uint8 packing and deterministic Morton-ordered NPZ are pure package code. `.vxz`, native dual-grid
  conversion, and voxel rendering require a separately compiled, pinned O-Voxel runtime.
- FlexGEMM and CuMesh are MIT source builds pinned by this package to audited commits with direct-URL provenance and
  runtime revision/build attestations.
- Production DINOv3 weights are gated under the separate DINOv3 License. nvdiffrast is a restricted research
  dependency and requires explicit acknowledgement; neither is redistributed or selected silently.

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

## Secure reviewed Hub loading

`AutoPipelineFor3D.from_pretrained()` accepts local directories or Hub repository IDs for reviewed package
integrations. Schema-v2 sidecars contain an exact immutable record for every constructor component: its name,
installed fully-qualified class, component subfolder, optionality, review status, and auto-loading eligibility.

For Hub IDs, the sidecar is validated before component download. The auto-loader then downloads only
`model_index.json`, the sidecar, and eligible component folders into a local Hub snapshot, validates every Diffusers
library/class tuple, and invokes the installed concrete pipeline class on that local path with remote code disabled.
Experimental optional SLAT and decoder components are not eligible for this path; use the concrete family pipeline
directly for explicitly experimental local artifacts.

Schema-v1 sidecars are no longer accepted. Re-run the matching converter or save the reviewed pipeline with this
package version to create a schema-v2 sidecar. `revision`, `cache_dir`, `token`, `local_files_only`, and `subfolder`
still apply to Hub snapshot resolution; they are consumed by the auto-loader rather than forwarded to the concrete
local pipeline load. `trust_remote_code=True` remains an error and never changes component eligibility.

## Contribution levels

1. Experimental Hub blocks use Modular Diffusers remote code and are inference-only.
2. Reviewed package integrations have exact loading registrations, manifests, parity tests, and dependency review.
   Training requires a separately reviewed recipe.
3. Stable dependency-light primitives may be proposed upstream to Diffusers. Upstream availability alone does not
   make a component trainable through this package.

See the [contribution guide](CONTRIBUTING.md), [lifecycle and review checklists](docs/contributions.md), and
[contribution templates](templates/README.md) for integration and promotion requirements.

Verification commands and the distinction between tiny CPU parity and production GPU quality are documented in
[docs/testing.md](docs/testing.md) and [docs/compatibility.md](docs/compatibility.md).
