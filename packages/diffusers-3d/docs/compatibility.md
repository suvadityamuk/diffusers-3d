# Compatibility

This page records what has been tested, not what merely imports. Run `diffusers-3d-report` to see the versions and
backend status of your own environment.

## Package stack

| Dependency | Supported | Tested in CI |
|---|---|---|
| Python | `>=3.10` | 3.10, 3.11, 3.12 |
| PyTorch | `>=2.6` | latest CPU wheel |
| Diffusers | `>=0.40.0.dev0,<0.42` | this repository's checkout |
| Transformers | `>=5.5.0` | 5.5.0 exactly, and latest |
| Accelerate | `>=1.1.0` | 1.1.0 exactly, and latest |

- Transformers 5.5.0 is the floor because DINOv3 first shipped in 4.56, which needs `huggingface_hub<1.0` and so
  conflicts with Diffusers 0.40.
- Accelerate 1.1.0 is the oldest release that loads checkpoints with `weights_only=True`. Older releases are rejected
  before any checkpoint is read.
- Changing a serialized object or pipeline field needs an explicit migration.

## Backends

An optional backend counts as supported only when its `BackendSpec` matches the running device, dtype, and version.
Source-built backends must also be installed from their pinned commit. No accelerated or research backend is selected
implicitly.

"CPU API" below means a test with a fake of the backend's API. It checks the adapter, not the compiled code.

| Backend | License class | CI coverage | Manual GPU run |
|---|---|---|---|
| trimesh, scikit-image, xatlas, fast-simplification | portable | real package on CPU | n/a |
| gsplat | accelerated | CPU API | A100, 2026-09-15: TRELLIS splats rendered at 512×512 in 0.4–0.6 s |
| CuMesh | accelerated; bundled cubvh is NVIDIA non-commercial | CPU API | A100, 2026-09-15: simplified a 2.5M-face O-Voxel mesh to 100k faces in 0.6 s |
| O-Voxel | accelerated | real NPZ codec; CPU API for native code | A100, 2026-09-15: voxel renders. Mesh extraction no longer uses it |
| FlexGEMM | accelerated | CPU API and provenance check | A100, 2026-09-15. No pipeline uses it any more |
| spconv | accelerated | CPU API | not run. No pipeline uses it any more |
| Kaolin FlexiCubes | accelerated, permissive subset only | CPU API | not run |
| utils3d | research-only | registry only | A100, 2026-09-15. No pipeline uses it any more |
| nvdiffrast | research-only | license gate only | A100, 2026-10-12, only to compare with upstream `to_glb` |
| nvdiffrec, diffoctreerast, mip-splatting | research-only | registry or license gate only | not run |

## Families

CI covers every family with tiny random-weight models on CPU: forward passes, pipelines, conversion, save/load, and
parity against the pinned upstream code. Those tests check architecture, state keys, and equations. They do not
measure output quality.

The released checkpoints were run by hand on one A100 80GB. Unless noted, runs used Transformers 5.19, the TRELLIS
`T.png` example image, and the Hub checkpoints, on 2026-10-13. Every mesh was rendered and inspected.

| Family | Upstream parity (tiny weights, CPU) | Released checkpoint on A100 |
|---|---|---|
| TRELLIS | Sparse-structure flow and decoder, SLAT flow, Gaussian, mesh, and radiance-field decoders | 2026-09-14/15. Image: 300k–500k splats in about 6 s at 3.7 GB (float16), and a 688k-face mesh. Text: splats in about 5 s at 3 GB. Textured GLB baked in 5 s with 99% of texels observed |
| TRELLIS.2 | Sparse-structure flow and decoder, SLAT flow, shape and PBR decoders | 2026-09-14, bfloat16: `512` in 7 s at 18 GB, `1024_cascade` in 26 s at 32 GB. 2026-10-12: `ovoxel_to_mesh` bit-identical to upstream on a 2.15M-face output. `to_textured_mesh` took 9.1 s against 17.0 s for upstream `to_glb`, and baked base colour within 2/255 of upstream on 99.8% of texels |
| TripoSG | Transformer, VAE decoder, scheduler, extraction, and a 4-step pipeline, bit-identical | float32: watertight 1.21M-face mesh in 83 s at 10.1 GB |
| Step1X-3D geometry | Transformer, VAE, grid, marching cubes, image encoder, and a 4-step pipeline, bit-identical | float32: 0.93M-face mesh in 95 s at 9.4 GB |
| Step1X-3D texture | Attention processors and betas bit-identical. Renders, texture sync, and bake within 3e-7 | float16 on a 200k-face mesh: 78 s at 20.5 GB, against 188 s for upstream. Bake matches upstream's `MeshRender` at 94.9 dB on covered texels. Inpainted texels differ from `cv2.inpaint` by 11.2/255 |
| Hunyuan3D-2.1 shape | Transformer, VAE, scheduler, preprocessing, and upstream's whole pipeline | float16: 1.10M-face mesh in 32 s at 7.3 GB. Not compared with upstream on GPU |
| Hunyuan3D-2.1 Paint | Both UNets, RoPE, view selection, and baking, bit-identical or within 1e-6 | float16 on a 40k-face mesh: 23.2 s at 12.1 GiB, against 64.4 s for upstream. Views match upstream at 61 dB PSNR. Covered texels match within 0.6/255. Inpainted texels differ from `cv2.inpaint` by 8.1/255 |
| SAM 3D Objects | Condition embedders and SLAT sampler bit-identical. Preprocessing, flows, and layout within 1e-4 | float32 with the alpha as mask: splats and a 0.58M-face mesh in 37 s at 11.1 GB |
| Pixal3D | Sparse-structure flow bit-identical. SLAT flow, NAF upsampler, and conditioning within 5e-7 | bfloat16, `1536_cascade`: O-Voxel and a 14.6M-face mesh in 40 s at 35.8 GB |

The TRELLIS and TRELLIS.2 GPU runs can be repeated with `scripts/gpu_smoke.py`; see [testing.md](testing.md).
