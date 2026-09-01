# Compatibility and verification matrix

This document records tested compatibility, not theoretical importability. Run `diffusers-3d-report` for the exact
versions and side-effect-free `BACKEND_REGISTRY` discovery status of a particular environment.

## Package stack

| Component | Declared compatibility | CI/verification status |
|---|---|---|
| `diffusers-3d` | `0.1.0.dev0` | Source tests plus wheel and sdist verification |
| Diffusers | `>=0.40.0.dev0,<0.41` | Local repository checkout, current `0.40` development minor |
| Transformers | `>=5.5.0` | Exact 5.5.0 minimum lane plus latest-resolved CPU lanes |
| Python | `>=3.10` | Core CPU matrix: 3.10, 3.11, 3.12 |
| PyTorch | `>=2.4` | CPU wheels in CI; no upper bound is claimed |

The verification environment recorded on 2026-08-24 used Python 3.12.3, Diffusers 0.40.0.dev0,
diffusers-3d 0.1.0.dev0, and PyTorch 2.13.0+cu130. That run was CPU-only despite the CUDA-enabled PyTorch build.
The workflow installs the local Diffusers checkout exactly once and installs the diffusers-3d wheel with `--no-deps`,
avoiding competing editable and released Diffusers installations.

DINOv3 classes first shipped in Transformers 4.56.0. That line requires
Hugging Face Hub `<1.0`, which cannot satisfy Diffusers 0.40's Hub 1.x
requirement, and it predates relevant security fixes. The declared 5.5.0
floor is the first intentionally supported floor for this package stack.

## Backend matrix

“CPU API” means real package tensors plus a fake implementation of the optional backend API. It verifies adaptation,
shape, dtype, policy, and error behavior; it is not compiled-backend or numerical GPU parity.

| Backend | Policy / license | CPU lane | GPU or compiled lane | Real-checkpoint status |
|---|---|---|---|---|
| trimesh | portable / permissive | Real optional package, I/O and round trips | N/A | N/A |
| scikit-image | portable / permissive | Real optional package, marching cubes | N/A | N/A |
| xatlas | portable / permissive | Real optional package, UV remapping | N/A | N/A |
| utils3d | research-only / permissive | Registry, provenance, and selection policy only | Not run | N/A |
| gsplat | accelerated / permissive | CPU API adapter test | Not run | None |
| spconv | accelerated / permissive | CPU API sparse-tensor test | Not run | None |
| FlexGEMM | accelerated / permissive | CPU API plus revision/build attestation | Not run | None |
| CuMesh | accelerated / permissive | CPU API geometry/BVH operations plus attestation | Not run | None |
| Kaolin | accelerated / permissive subset only | CPU API FlexiCubes test; non-commercial module rejected | Not run | None |
| O-Voxel | accelerated / permissive | Real pure tensor/uint8/NPZ codec; native API fake | Not run | None |
| nvdiffrast | research-only / restricted | License gate/facade only | Not run | None |
| nvdiffrec render | research-only / restricted | Registry metadata only | Not run | None |
| diffoctreerast | research-only / restricted | License gate/facade only | Not run | None |
| mip-Gaussian rasterizer | research-only / restricted | License gate/facade only | Not run | None |

No accelerated or research-only backend is selected implicitly. Backend discovery checks module and distribution
metadata without importing optional modules.

## Family and checkpoint matrix

The shipped reviewed families are TRELLIS and TRELLIS.2.

| Family | License boundary | Tiny CPU evidence | Pinned-source parity | Production GPU / real checkpoint |
|---|---|---|---|---|
| TRELLIS | MIT upstream; Apache-2.0 glue; restricted renderers separate | Sparse structure plus experimental tiny SLAT equations, conversion, training, save/load | Sparse flow/decoder forward and flow backward | Not run |
| TRELLIS.2 | MIT upstream; DINOv3 and nvdiffrast separately restricted | Reviewed sparse structure plus experimental tiny SLAT/O-Voxel/PBR channels, conversion, training, save/load | Sparse flow/decoder forward and flow backward | Not run |

All converter tests use synthetic tiny state dictionaries. Pinned-source
parity first verifies the exact commit, expected origin, clean worktree, and
tracked source tree, then uses deterministic tiny random weights rather than
downloaded production checkpoints. Therefore these results establish
architecture, state-key, equation, gradient, and serialization compatibility
only. They do not establish production image-to-3D quality, full-resolution
memory behavior, compiled O-Voxel output, PBR GLB quality, or
published-checkpoint parity.

## Commands

```bash
diffusers-3d-report
python -m pytest packages/diffusers-3d/tests
python -m pytest packages/diffusers-3d/tests -m portable
python -m pytest packages/diffusers-3d/tests -m reference_parity
```

See [testing.md](testing.md) for the core selector, manifest validation, wheel install/import/release scan, and the
manual GPU/research policy.
