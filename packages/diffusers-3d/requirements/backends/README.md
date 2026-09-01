# Backend constraints

This directory records audited source revisions and build constraints for dependencies that cannot be represented by
portable PyPI extras.

Every source-identity pin must include:

- the immutable source commit;
- package and import names;
- license classification;
- build-isolation requirements;
- a checksum for any externally hosted wheel.

Source pins do not establish a supported build matrix. A full compatibility
record additionally includes the tested Python, PyTorch, CUDA/HIP, compiler,
operating-system, and GPU architecture combination plus the package report from
the manual accelerated lane.

Do not add mutable branches or unverified third-party wheels.

The three TRELLIS.2 native records are intentionally separate because the
`o_voxel` source metadata names FlexGEMM and CuMesh without immutable
revisions. Install the two pinned dependencies first, then install O-Voxel
with dependency resolution disabled so its mutable transitive Git
requirements cannot replace those pins:

```bash
python -m pip install \
  -r packages/diffusers-3d/requirements/backends/flex-gemm.txt \
  -r packages/diffusers-3d/requirements/backends/cumesh.txt
python -m pip install --no-deps --no-build-isolation \
  -r packages/diffusers-3d/requirements/backends/o-voxel.txt
```

These records establish source identity only. They do not claim a completed
GPU compatibility run; record the Python, PyTorch, accelerator runtime,
compiler, operating system, GPU architecture, and package report in the
manual accelerated lane before treating a build as supported.

Restricted research backends are pinned separately and require explicit
license review before installation:

```bash
python -m pip install --no-build-isolation \
  -r packages/diffusers-3d/requirements/backends/nvdiffrast.txt \
  -r packages/diffusers-3d/requirements/backends/diffoctreerast.txt \
  -r packages/diffusers-3d/requirements/backends/mip-splatting.txt
```

The pinned nvdiffrec tree has no Python build metadata. Its source-only record
is `requirements/backends/nvdiffrec.txt`; the registry rejects it unless an
audited package exposes matching PEP 610 direct-URL provenance.
