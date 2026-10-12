# Diffusers 3D

`diffusers-3d` is an object-native companion to
[Diffusers](https://github.com/huggingface/diffusers). It provides tensor-native representations, modular inference
pipelines, and optional geometry backends for running generative 3D models.

## Status

The package is pre-alpha. Loading metadata uses schema version `2`. Model integrations and optional compiled
backends remain capability-gated. The reviewed model families are TRELLIS and TRELLIS.2, including TRELLIS.2's optional O-Voxel
paths.

## First-release readiness

The code release gate requires all of the following before publication:

- [x] Production execution registries contain only reviewed TRELLIS and TRELLIS.2 classes and are immutable.
- [x] Tiny TRELLIS and TRELLIS.2 model, pipeline, auto-loading, and save/load round trips pass.
- [x] Built wheel and sdist pass `tools/verify_wheel.py` and `tools/verify_sdist.py`.

This checklist covers code readiness only. Selecting a final version and publishing a release are separate maintainer
actions; the package remains at its development version until then.

The distribution uses an `Apache-2.0 AND MIT` aggregate license expression. Package-owned glue is Apache-2.0, while
the TRELLIS and TRELLIS.2-derived family code retains MIT terms. Wheel and sdist metadata ship the local Apache
license and both applicable MIT licenses and notices. Those terms apply to their respective files; the aggregate
expression does not relicense any family code or model artifact.

## Current TRELLIS.2 limitations

- Every network runs in plain PyTorch, with tiny-weight parity against the pinned upstream code in the CPU test
  matrix. The full 4B checkpoint, the compiled O-Voxel mesh/render path, and the PBR GLB export run in the separate
  GPU smoke workflow (`scripts/gpu_smoke.py`), which checks coarse invariants only; quality is not claimed.
- O-Voxel schema/mixed lossless packing and deterministic lexicographic NPZ across the uint16 coordinate domain are
  pure package code. Explicit 30-bit Morton ordering remains available through coordinate 1023. Unit-domain PBR
  channels use uint8 while out-of-cell dual vertices and unbounded split weights retain float16/float32. NPZ files
  contain only official-reader fields. `.vxz`, native dual-grid conversion, and voxel rendering require a separately
  compiled, pinned O-Voxel runtime; the pinned VXZ v0 codec cannot losslessly store floating-point attributes and
  rejects such writes.
- FlexGEMM and CuMesh are MIT source builds pinned by this package to audited commits with direct-URL provenance and
  runtime API/toolchain checks. Raw upstream modules do not need custom revision or build attributes.
- Production DINOv3 weights are gated under the separate DINOv3 License. nvdiffrast is a restricted research
  dependency and requires explicit acknowledgement; neither is redistributed or selected silently.

## Design principles

- `Object3D` is a structural data protocol.
- Meshes, Gaussian splats, sparse voxels, and O-Voxels remain tensor-native and preserve representation-specific
  channels.
- Diffusers owns model loading, scheduling, offloading, and pipeline lifecycle behavior.
- Optional CUDA and research-only dependencies are never imported or selected implicitly.

## Installation

Install the package from this repository:

```bash
pip install -e packages/diffusers-3d
```

The core package requires PyTorch 2.6 or newer and `accelerate>=1.1.0`.

Portable mesh processing is optional:

```bash
pip install -e "packages/diffusers-3d[portable]"
```

Source-built or CUDA-specific dependencies such as nvdiffrast, spconv, FlexGEMM, CuMesh, and O-Voxel require an
explicit backend installation. See [docs/backends.md](docs/backends.md).

## Usage

[docs/inference.md](docs/inference.md) covers checkpoint conversion, loading, conditioning, `formats`, and saving
each asset type. The runnable [TRELLIS.2 example](src/diffusers_3d/families/trellis2/examples/image_to_3d.py) exercises the whole
inference API and has an offline tiny-component mode:

```bash
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/
```

## Secure reviewed Hub loading

`AutoPipelineFor3D.from_pretrained()` accepts local directories or Hub repository IDs for reviewed package
integrations. Schema-v2 sidecars contain an exact immutable record for every constructor component: its name,
installed fully-qualified class, component subfolder, optionality, review status, and auto-loading eligibility.

For Hub IDs, the sidecar is validated before component download. The auto-loader then downloads only
`model_index.json`, the sidecar, and eligible component folders into a local Hub snapshot, validates every Diffusers
library/class tuple, and invokes the installed concrete pipeline class on that local path with remote code disabled.
Components whose sidecar entry is marked ineligible are rejected before download.

Schema-v1 sidecars are no longer accepted. Re-run the matching converter or save the reviewed pipeline with this
package version to create a schema-v2 sidecar. `revision`, `cache_dir`, `token`, `local_files_only`, and `subfolder`
still apply to Hub snapshot resolution; they are consumed by the auto-loader rather than forwarded to the concrete
local pipeline load. `trust_remote_code=True` remains an error and never changes component eligibility.

## Contributing

See the [contribution guide](CONTRIBUTING.md) and [contribution templates](templates/README.md) for adding a model
family.

Verification commands and the distinction between tiny CPU parity and production GPU quality are documented in
[docs/testing.md](docs/testing.md) and [docs/compatibility.md](docs/compatibility.md).
