# Backend constraints

This directory records audited source revisions and build constraints for dependencies that cannot be represented by
portable PyPI extras.

Constraint files must include:

- the immutable source commit;
- package and import names;
- license classification;
- supported Python, PyTorch, CUDA/HIP, compiler, and operating-system combinations;
- build-isolation requirements;
- a checksum for any externally hosted wheel.

Do not add mutable branches or unverified third-party wheels.

The three TRELLIS.2 native records are intentionally separate because the
`o_voxel` source metadata names FlexGEMM and CuMesh without immutable
revisions. Install all three records together so the top-level direct
requirements constrain those transitive source dependencies:

```bash
python -m pip install \
  -r packages/diffusers-3d/requirements/backends/flex-gemm.txt \
  -r packages/diffusers-3d/requirements/backends/cumesh.txt
python -m pip install --no-build-isolation \
  -r packages/diffusers-3d/requirements/backends/o-voxel.txt
```

These records establish source identity only. They do not claim a completed
GPU compatibility run; record the Python, PyTorch, accelerator runtime,
compiler, operating system, GPU architecture, and package report in the
manual accelerated lane before treating a build as supported.
