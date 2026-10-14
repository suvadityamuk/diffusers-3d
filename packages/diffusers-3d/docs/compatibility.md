# Compatibility and verification matrix

This document records tested compatibility, not theoretical importability. Run `diffusers-3d-report` for the exact
versions and side-effect-free `BACKEND_REGISTRY` discovery status of a particular environment.

## Package stack

| Component | Declared compatibility | CI/verification status |
|---|---|---|
| `diffusers-3d` | `0.1.0.dev0` | Source tests plus wheel and sdist verification |
| Diffusers | `>=0.40.0.dev0,<0.42` | Local repository checkout, current `0.40` and `0.41` development minors |
| Accelerate | `>=1.1.0` | Exact minimum safety lane plus latest-resolved CPU lanes |
| Transformers | `>=5.5.0` | Exact 5.5.0 minimum lane plus latest-resolved CPU lanes |
| Python | `>=3.10` | Core CPU matrix: 3.10, 3.11, 3.12 |
| PyTorch | `>=2.6` | CPU wheels in CI; no upper bound is claimed |

The verification environment recorded on 2026-08-24 used Python 3.12.3, Diffusers 0.40.0.dev0,
diffusers-3d 0.1.0.dev0, and PyTorch 2.13.0+cu130. That run was CPU-only despite the CUDA-enabled PyTorch build.
CI also verifies the repository's current 0.41 development checkout. The workflow installs the local Diffusers
checkout exactly once and installs the diffusers-3d wheel with `--no-deps`,
avoiding competing editable and released Diffusers installations.

DINOv3 classes first shipped in Transformers 4.56.0. That line requires
Hugging Face Hub `<1.0`, which cannot satisfy Diffusers 0.40's Hub 1.x
requirement, and it predates relevant security fixes. The declared 5.5.0
floor is the first intentionally supported floor for this package stack.

Accelerate 1.1.0 is the minimum checkpoint-safety baseline. Runtime checks reject older releases before checkpoint
deserialization. Direct utility loads explicitly request `weights_only=True`; `Accelerator.load_state` receives the
same explicit load option when that release exposes `load_kwargs`, while Accelerate 1.1's internal utility already
uses weights-only loading by default. Exact save/load additionally rejects more than one process, every
`distributed_type` except `DistributedType.NO`, and registered custom checkpoint objects before persistence or
deserialization, so Accelerate's custom `weights_only=False` pickle path is unreachable.

## Backend matrix

“CPU API” means real package tensors plus a fake implementation of the optional backend API. It verifies adaptation,
shape, dtype, policy, and error behavior; it is not compiled-backend or numerical GPU parity.

| Backend | Policy / license | CPU lane | GPU or compiled lane | Real-checkpoint status |
|---|---|---|---|---|
| trimesh | portable / permissive | Real optional package, I/O and round trips | N/A | N/A |
| scikit-image | portable / permissive | Real optional package, marching cubes | N/A | N/A |
| xatlas | portable / permissive | Real optional package, UV remapping | N/A | N/A |
| fast-simplification | portable / permissive | Real optional package, quadric decimation through `TrimeshBackend` (Hunyuan3D-2.1 Paint's optional remesh) | N/A | N/A |
| utils3d | research-only / permissive | Registry, provenance, and selection policy only | Manual run only, not CI: built from the pinned revision on one A100 80GB (2026-09-15) and exercised on real `microsoft/TRELLIS.2-4B` / `microsoft/TRELLIS-image-large` outputs; provenance verified. Neither family uses it any more | N/A |
| gsplat | accelerated / permissive | CPU API adapter test | Manual run only, not CI: built from the pinned revision on one A100 80GB (2026-09-15) and exercised on real `microsoft/TRELLIS.2-4B` / `microsoft/TRELLIS-image-large` outputs: rasterized 300k-500k TRELLIS splats from four 512x512 turntable views in 0.4-0.6 s (recognizable renders) | None |
| spconv | accelerated / permissive | CPU API sparse-tensor test | Not run (the pipelines no longer need it) | None |
| FlexGEMM | accelerated / permissive | CPU API plus PEP 610 source verification; CUDA device type only (including PyTorch HIP builds) | Manual run only, not CI: built from the pinned revision on one A100 80GB (2026-09-15) and exercised on real `microsoft/TRELLIS.2-4B` / `microsoft/TRELLIS-image-large` outputs. The TRELLIS.2 mesh and textured-mesh paths no longer use it | None |
| CuMesh | accelerated / restricted (bundled cubvh is NVIDIA non-commercial) | CPU API geometry/BVH operations; any install accepted | Manual run only, not CI: built from the pinned revision on one A100 80GB (2026-09-15) and exercised on real `microsoft/TRELLIS.2-4B` / `microsoft/TRELLIS-image-large` outputs: repaired and simplified a 2.5M-face O-Voxel mesh to 100k faces in 0.6 s | None |
| Kaolin | accelerated / permissive subset only | CPU API FlexiCubes test; non-commercial module rejected | Not run | None |
| O-Voxel | accelerated / permissive | Real mixed uint8/float NPZ codec; native API fake | Manual run only, not CI: built from the pinned revision on one A100 80GB (2026-09-15) and exercised on real `microsoft/TRELLIS.2-4B` / `microsoft/TRELLIS-image-large` outputs: `render_voxels` 512x512 base-color views. Mesh extraction no longer uses it (`ovoxel_to_mesh` is plain PyTorch) | None |
| nvdiffrast | research-only / restricted | License gate/facade only | Manual run only, not CI: built on one A100 80GB (2026-10-12) only to compare against upstream `to_glb`; no package path uses it any more | None |
| nvdiffrec render | research-only / restricted | Registry metadata only | Not run | None |
| diffoctreerast | research-only / restricted | License gate/facade only | Not run | None |
| mip-Gaussian rasterizer | research-only / restricted | License gate/facade only | Not run | None |

No accelerated or research-only backend is selected implicitly. Backend discovery checks module and distribution
metadata without importing optional modules.

## Family and checkpoint matrix

The shipped reviewed families are TRELLIS, TRELLIS.2, TripoSG, Step1X-3D (geometry and texture), Hunyuan3D-2.1
(shape and Paint), SAM 3D Objects, and Pixal3D. Unless noted, the GPU runs below used one A100 80GB on 2026-10-13
with Transformers 5.19, the TRELLIS `T.png` example image, and the converted checkpoints published on the Hub; every
mesh was rendered and inspected, and every geometry-stage mesh has positive signed volume (outward-facing faces).

| Family | License boundary | Tiny CPU evidence | Pinned-source parity | Production GPU / real checkpoint |
|---|---|---|---|---|
| TRELLIS | MIT upstream; Apache-2.0 FlexiCubes port and glue; restricted renderers separate | Sparse structure, SLAT flow, Gaussian, mesh, and radiance-field decoders, stock Transformers DINOv2 image encoder and CLIP text encoder, image and text pipelines, conversion, save/load, texture baking, radiance-field renderer | Sparse-structure flow/decoder, SLAT flow, swin Gaussian decoder, swin mesh decoder + FlexiCubes, and swin radiance-field decoder forward (spconv/xformers shimmed to dense PyTorch) | Manual run only, not CI: `microsoft/TRELLIS-image-large` (image encoder from `facebook/dinov2-with-registers-large`) converted and run on one A100 80GB in float16 (2026-09-14); returned recognizable 300k-500k Gaussian splats for two example images in about 6 s at 3.7 GB peak; on 2026-09-15 in bfloat16 the mesh decoder returned a 345k-vertex / 688k-face FlexiCubes mesh for the same image and `microsoft/TRELLIS-text-large` returned recognizable splats for two prompts in about 5 s at 3 GB peak. `scripts/gpu_smoke.py` (2026-09-15, bfloat16): all three SLAT decoders in one 6.7 s pass at 7 GB peak; the radiance-field decoder gave a 16k-voxel rank-16 field whose pure-PyTorch 256x256 render (0.8 s) has 0.95 silhouette IoU with the gsplat render of the same latent; `TrellisGlbPostprocessFacade` baked a 1024x1024 texture onto the 34k-face simplified mesh in 5 s with 99% of texels observed |
| TRELLIS.2 | MIT upstream; DINOv3 and nvdiffrast separately restricted | Sparse structure, 512/1024 SLAT flows, shape and PBR decoders, conversion, save/load | Sparse-structure flow/decoder, SLAT flow, and shape/PBR decoder forward (FlexGEMM/xformers shimmed to dense PyTorch) | Manual run only, not CI: `microsoft/TRELLIS.2-4B` converted and run on one A100 80GB in bfloat16 (2026-09-14); `512` and `1024_cascade` returned recognizable O-Voxel objects for two example images in 7 s / 26 s at 18 GB / 32 GB peak. On 2026-10-12 (A100, bfloat16, `512`, TRELLIS `T.png`): `ovoxel_to_mesh` matched upstream `flexible_dual_grid_to_mesh` bit for bit on a 1.06M-voxel output (2.15M faces), and `to_textured_mesh` produced a 299k-face mesh with 2048x2048 textures in 9.1 s versus 17.0 s for upstream `to_glb` on the same input (one cold run each). Baked onto upstream's own atlas, the plain-PyTorch rasterizer and trilinear sampler matched upstream's base colour within 2/255 on 99.8% of 2.6M texels (mean 0.5/255). With the default `remesh=True` (upstream's demo settings), `to_textured_mesh` took 14.8 s versus 23.7 s for upstream and its baking matched upstream's remeshed atlas just as closely. Whole textures are not comparable texel by texel: CuMesh simplification is not deterministic, and two upstream runs on the same input gave 299,030 and 298,861 faces and differently packed atlases. |
| TripoSG | MIT upstream; transformer derived from Tencent HunyuanDiT (not EU, UK, or South Korea), warns on construction | Stock Transformers DINOv2 image encoder, transformer, VAE decoder, scheduler, hierarchical marching cubes, pipeline, conversion, save/load, LoRA | Transformer, VAE decoder, scheduler loop, hierarchical extraction, and a 4-step guided pipeline bit-identical | Manual run only, not CI: `VAST-AI/TripoSG` converted in 25 s and run in float32: 83 s at 10.1 GB peak for a watertight 1.21M-face mesh |
| Step1X-3D (geometry) | Apache-2.0 | Stock Transformers DINOv2-with-registers image encoder, Flux-style transformer, VAE decoder, scheduler, pipeline, conversion, save/load, LoRA | Transformer, VAE tokens and queries, dense grid, marching cubes, image-encoder tokens, and a 4-step guided pipeline bit-identical | Manual run only, not CI: `stepfun-ai/Step1X-3D` geometry converted in 22 s and run in float32: 95 s at 9.4 GB peak for a 0.93M-face mesh |
| Step1X-3D (texture) | Apache-2.0 plus MIT (SyncMVD-derived texture sync) and CreativeML Open RAIL++-M (SDXL base weights) | Multi-view attention processors on a stock SDXL `UNet2DConditionModel`, stock `T2IAdapter`, `AutoencoderKL`, CLIP text encoders and `DDPMScheduler`, texture sync, pipeline, conversion, save/load, shared `multiview_baking` backend | Attention processors and scheduler betas bit-identical; control renders within 1.2e-6; texture-sync step within 2.4e-7; final `MeshRender` bake within 3e-7 on covered texels, with the package rasterizer in place of nvdiffrast, pytorch3d, and `custom_rasterizer` | Manual run only, not CI: converted in 40 s and run in float16 on the geometry stage's mesh simplified to 200k faces: 78 s at 20.5 GB peak (50 denoising steps 23 s), against 188 s for upstream's generation. Upstream's `MeshRender` baking our views on our atlas matches our bake at 94.9 dB on covered texels (trust masks differ by 3 of 1.07M texels); inpainted texels differ from `cv2.inpaint` by 11.2/255 on average. Generated views differ from upstream's in colour because the random streams differ |
| Hunyuan3D-2.1 (shape) | Tencent Hunyuan 3D 2.1 Community License (not EU, UK, or South Korea); warns on construction | DINOv2 image encoder, MoE DiT, ShapeVAE decoder, scheduler, pipeline, conversion, save/load, LoRA | Transformer, VAE decoder, scheduler, OpenCV-equivalent preprocessing, and upstream's own pipeline end to end (conditioning, latents, mesh) | Manual run only, not CI: `tencent/Hunyuan3D-2.1` converted in 42 s and run in float16: 32 s at 7.3 GB peak for a 1.10M-face mesh. Not compared with upstream on GPU |
| Hunyuan3D-2.1 (Paint) | Tencent Hunyuan 3D 2.1 Community License plus CreativeML Open RAIL++-M (SD 2.1 UNets and VAE) and BSD-3-Clause (Real-ESRGAN weights); warns on construction | Multiview PBR UNet on two stock `UNet2DConditionModel`s, stock `AutoencoderKL`, `UniPCMultistepScheduler`, `Dinov2Model`, and `BitImageProcessorPil`, RRDBNet upscaler, pipeline, conversion, save/load, auto-loading, GLB export, shared `multiview_baking` backend | Tiny `UNet2p5DConditionModel` (both UNets, every added attention) and 3D RoPE/voxel indices bit-identical; view selection identical; `back_project` + `fast_bake_texture` bit-identical (`back_sample`) or within 1e-6 (`linear`), with the package rasterizer in place of `custom_rasterizer` | Manual run only, not CI: `tencent/Hunyuan3D-2.1` converted (33 s) and run on one A100 80GB in float16 (2026-10-13) on the shape pipeline's 1.1M-face mesh for TRELLIS `T.png`, simplified to 40k faces. The first call took 23.2 s at 12.1 GiB peak, excluding GLB export. One upstream Paint call on the same prepared mesh and atlas took 64.4 s at 13.4 GiB, including writing its OBJ and JPEG textures. The generated views match upstream at 61 dB PSNR (SSIM 0.9997). In the 2048x2048 base-colour texture, texels a view covers match within 0.6/255 on average (48.9 dB). The 62% of covered texels that no view sees are inpainted, and there `fill_texture` differs from upstream's `cv2.inpaint` by 8.1/255 on average (23.4 dB) |
| SAM 3D Objects | SAM License (not open source); warns on construction | One stock DINOv2-with-registers image encoder (the release's four copies are identical), condition fusers, sparse-structure and SLAT flows, Gaussian and FlexiCubes decoders, pipeline, conversion, save/load | Condition embedders and SLAT sampler bit-identical; preprocessing within 4.8e-7; sparse-structure flow within 5.4e-7; sparse-structure sampler with layout within 1e-4 | Manual run only, not CI: `facebook/sam-3d-objects` converted in 25 s and run in float32 with the image's alpha as the mask: 37 s at 11.1 GB peak for Gaussian splats plus a 0.58M-face mesh in the input camera's frame |
| Pixal3D | MIT upstream; NAF upsampler code Apache-2.0; DINOv3 separately restricted | TRELLIS.2 models reused unchanged, pixel-aligned projection conditioning, NAF upsampler, stock Transformers DINOv3 image encoder, pipeline, conversion, save/load | Sparse-structure flow bit-identical; SLAT flow within 1.2e-7; NAF upsampler within 3e-7 of upstream NAF on NATTEN's CPU backend; single- and multi-view conditioning within 4.8e-7 | Manual run only, not CI: `TencentARC/Pixal3D` converted in 60 s and run in bfloat16 (`1536_cascade`, `camera_angle_x=0.35`): 40 s at 35.8 GB peak for an O-Voxel and a 14.6M-face mesh |

All converter tests use synthetic tiny state dictionaries. Pinned-source
parity first verifies the exact commit, expected origin, clean worktree, and
tracked source tree, then uses deterministic tiny random weights rather than
downloaded production checkpoints. Therefore these results establish
architecture, state-key, equation, gradient, and serialization compatibility
only. They do not establish production image-to-3D quality, full-resolution
memory behavior, PBR texture quality, or published-checkpoint parity.

Reviewed local and Hub auto-loading uses object-3D model-index schema version 2. Hub repository IDs are resolved to
local immutable snapshots containing only the sidecar, Diffusers model index, and reviewed eligible component
folders. Diffusers then imports exact classes from the installed `diffusers_3d` package with
`trust_remote_code=False`. Schema-v1 sidecars and components marked ineligible in the sidecar are rejected by the
auto-loader.

## Commands

```bash
diffusers-3d-report
python -m pytest packages/diffusers-3d/tests
python -m pytest packages/diffusers-3d/tests -m portable
python -m pytest packages/diffusers-3d/tests -m reference_parity
```

See [testing.md](testing.md) for the core selector, wheel install/import check, and the
manual GPU/research policy.
