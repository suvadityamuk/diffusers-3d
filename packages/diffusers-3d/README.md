# Diffusers 3D

`diffusers-3d` is an object-native companion to
[Diffusers](https://github.com/huggingface/diffusers). It provides tensor-native representations, modular inference
pipelines, and optional geometry backends for running generative 3D models.

## Status

The package is pre-alpha. Loading metadata uses schema version `2`. Model integrations and optional compiled
backends remain capability-gated. The reviewed model families are TRELLIS, TRELLIS.2 (including its optional O-Voxel
paths), TripoSG, SAM 3D Objects, Step1X-3D (geometry and texture), Pixal3D, and Hunyuan3D-2.1 (shape and Paint texture stages).

## First-release readiness

The code release gate requires all of the following before publication:

- [x] Production execution registries contain only reviewed family classes and are immutable.
- [x] Tiny model, pipeline, auto-loading, and save/load round trips pass for every family.
- [x] Built wheel and sdist pass `tools/verify_wheel.py` and `tools/verify_sdist.py`.

This checklist covers code readiness only. Selecting a final version and publishing a release are separate maintainer
actions; the package remains at its development version until then.

The distribution uses an `Apache-2.0 AND MIT AND LicenseRef-SAM-License AND LicenseRef-Tencent-Hunyuan-3D-2.1-Community`
aggregate license expression. Package-owned glue and the Step1X-3D-derived code are Apache-2.0; the TRELLIS-,
TRELLIS.2-, TripoSG-, and Pixal3D-derived code and Step1X-3D's SyncMVD-derived texture synchronisation retain MIT
terms; the SAM 3D Objects-derived code is under Meta's SAM
License (not an open-source license); and the Hunyuan3D-2.1-derived code is under the Tencent Hunyuan 3D 2.1 Community
License, which does not apply in the European Union, the United Kingdom, or South Korea. Wheel and sdist metadata ship
every applicable license and notice. Those terms apply to their respective files; the aggregate expression does not
relicense any family code or model artifact. Pipelines whose code or weights carry restrictive terms warn when
constructed.

## Current TRELLIS.2 limitations

- Every network runs in plain PyTorch, with tiny-weight parity against the pinned upstream code in the CPU test
  matrix, and mesh extraction matches the pinned upstream code bit for bit. The full 4B checkpoint and the CuMesh
  textured-mesh export run in the separate GPU smoke workflow (`scripts/gpu_smoke.py`), which checks coarse invariants only; quality is not claimed.
- O-Voxel schema/mixed lossless packing and deterministic lexicographic NPZ across the uint16 coordinate domain are
  pure package code. Explicit 30-bit Morton ordering remains available through coordinate 1023. Unit-domain PBR
  channels use uint8 while out-of-cell dual vertices and unbounded split weights retain float16/float32. NPZ files
  contain only official-reader fields. `.vxz` and voxel rendering require a separately
  compiled, pinned O-Voxel runtime; the pinned VXZ v0 codec cannot losslessly store floating-point attributes and
  rejects such writes.
- FlexGEMM is an MIT source build pinned by this package to an audited commit with direct-URL provenance and runtime
  API/toolchain checks. CuMesh is accepted from any install and only runtime-checked; it bundles cubvh under NVIDIA's
  non-commercial license, and loading it warns that compliance is on the user.
- Production DINOv3 weights are gated under the separate DINOv3 License. nvdiffrast is a restricted research
  dependency and requires explicit acknowledgement; neither is redistributed or selected silently.

## Current limitations of the other families

- No pipeline removes backgrounds: pass RGBA or a mask. Upstream runs a background remover on images without alpha.
- TripoSG, Step1X-3D, and Hunyuan3D-2.1 extract meshes with scikit-image marching cubes on their released grids, and
  return them raw. Upstream's optional CUDA-only fast decoders and extractors (FlashVDM, `diso`) and its pymeshlab
  post-processing are not ported, mostly for licensing reasons; each family README lists what differs.
- The texture stages fill texels no view sees with the package's own fill instead of `cv2.inpaint`; on those texels
  the result differs from upstream by about 8–11/255 on average. Covered texels match upstream's bake closely
  (49 dB for Hunyuan3D-2.1 Paint, 95 dB for Step1X-3D's bake).
- Step1X-3D's texture stage draws all its randomness from the pipeline's `generator`, so its generated views cannot
  match upstream's for a given seed; Hunyuan3D-2.1 Paint does reproduce upstream's views (61 dB).
- SAM 3D Objects needs a mask selecting the object; depth from a pointmap is optional, and MoGe depth estimation and
  layout post-optimization are not ported.
- Pixal3D and TRELLIS.2 need the gated DINOv3 image encoder passed in when loading the published checkpoints.

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

Portable mesh processing (trimesh, scikit-image, xatlas, fast-simplification) is optional, but TripoSG, Step1X-3D,
Hunyuan3D-2.1, both texture stages, and GLB export need it:

```bash
pip install -e "packages/diffusers-3d[portable]"
```

Source-built or CUDA-specific dependencies such as nvdiffrast, spconv, FlexGEMM, CuMesh, and O-Voxel require an
explicit backend installation. See [docs/backends.md](docs/backends.md).

## Usage

Converted checkpoints for every pipeline are published on the Hub (see the root README); load them with
`AutoPipelineForImageTo3D.from_pretrained`, or `AutoPipelineFor3D.from_pretrained` for the texture stages. [docs/inference.md](docs/inference.md) covers loading, conditioning,
`formats`, saving each asset type, texturing a mesh, and the converter for every family. The runnable [TRELLIS.2 example](src/diffusers_3d/families/trellis2/examples/image_to_3d.py) exercises the whole
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
