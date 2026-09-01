# TRELLIS.2 image-to-3D

This family integrates the MIT-licensed Microsoft TRELLIS.2 implementation at
revision `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. It is a distinct
`trellis2` family. The reviewed package contract is the portable
image-to-sparse-structure stage; sparse SLAT, O-Voxel decoding, mesh conversion,
and PBR/GLB postprocess remain explicitly experimental or capability-gated.

## Reviewed portable path

- `Trellis2SparseStructureFlowModel` preserves the released state layout,
  RoPE, shared modulation, scaled initialization, and self/cross-attention Q/K
  RMS norms. Tiny CPU float32 output and selected backward gradients match the
  pinned implementation with identical state and inputs.
- `Trellis2SparseStructureDecoder` intentionally subclasses the TRELLIS dense
  decoder because TRELLIS.2 imports that architecture unchanged. Output
  metadata records the exact `trellis-image-large` checkpoint semantics.
- `Trellis2Dinov3Conditioner` uses public Transformers DINOv3 classes, ImageNet
  normalization, the released token path, and manual unparameterized final
  layer normalization. Tiny tests build from configuration without downloads.
- `Trellis2FlowEulerScheduler` implements the TRELLIS.2
  `w*conditional + (1-w)*negative` CFG equation, x0 guidance rescale and
  interval, rational `rescale_t`, `t*1000` model input, and Euler updates.
- `Trellis2ImageTo3DPipeline` always supports
  `formats=("sparse_structure",)` on CPU and round-trips through standard
  Diffusers save/load and `AutoPipelineForImageTo3D`. Released defaults are 12
  steps with the per-stage strengths, rescale values, intervals, and 1024
  cascade configuration serialized in the pipeline config. The upstream sparse
  target mapping is preserved: `512`, `1024_cascade`, and `1536_cascade` pool
  decoded occupancy to resolution 32, while `1024` uses 64. Portable tiny
  pipelines use their decoder's native output resolution.

## O-Voxel object and codecs

`OVoxelAsset` retains active XYZ coordinates, fractional dual vertices,
three-axis intersection flags, split weights, base color, metallic, roughness,
alpha, signed normals, emissive, the native right-handed Z-up grid transform,
resolution, and AABB.

The package-owned O-Voxel adapter has independent capability surfaces:

- schema conversion and mixed official packing are pure PyTorch: unit-domain
  channels use uint8 while unbounded split weights retain float16/float32;
- `.npz` read/write is pure NumPy/Python and uses deterministic 30-bit Morton
  ordering plus explicit grid and dtype/layout metadata by default;
- `.vxz` read/write delegates to `o_voxel.io.read_vxz`/`write_vxz`;
- dual-grid mesh extraction delegates to
  `o_voxel.convert.flexible_dual_grid_to_mesh`;
- voxel rendering delegates to `o_voxel.rasterize.VoxelRenderer`.

`.vxz` is never advertised as a pure codec. It does not encode the original
grid resolution, so `OVoxelBackend.read_vxz` requires the caller to supply it.
VXZ v0 also accepts only uint8 attributes, so assets containing unbounded split
weights are rejected on write; NPZ is the lossless serialization path.
The pinned `o_voxel` package eagerly imports its PBR postprocess module, which
imports nvdiffrast; consequently native loading requires both a compiled,
provenance-verified O-Voxel build and explicit
`accept_nvdiffrast_research_license=True`. Pure schema and `.npz` paths never
import that runtime.

## Experimental sparse and PBR stages

`Trellis2SLatFlowModel` provides a backend-free tiny full-attention core. The
texture form accepts a coordinate-aligned `concat_cond` shape SLAT. Production
construction requires FlexGEMM selection and then raises
`NotImplementedError` until released sparse input/output blocks have measured
checkpoint parity.

`Trellis2ShapeDualGridDecoder` and `Trellis2PBRSparseDecoder` implement
deterministic tiny contracts that return native `OVoxelAsset` values and
preserve the complete material-channel layout. They do not claim official
checkpoint parity. Production decoders require the released sparse UNet,
FlexGEMM, and compiled O-Voxel runtime.

The pipeline can expose tiny shape SLAT, texture SLAT, and O-Voxel stages and
can return a `MeshAsset` through the explicit O-Voxel backend. GLB/trimesh
postprocess is not a pipeline `formats` value: call
`postprocess_ovoxel(asset, output_format="glb")` explicitly when the optional,
license-gated stack is available. The serialized 1024 cascade fails explicitly
when a production SLAT/O-Voxel stage is requested, until sparse/GPU parity is
measured.

## Backend and license boundaries

- `FlexGemmBackend` is limited to released submanifold sparse convolution and
  3D grid sampling. `CuMeshBackend` is limited to repair, simplify, remesh, UV,
  and BVH operations. This package pins FlexGEMM at
  `6dd94a859c26ee8246888502eada3dd8ad85532e` and CuMesh at
  `12289e1062f0603f2f0d0771b02e1395d247f26f`; discovery and runtime loading
  require matching PEP 610 source provenance followed by runtime API/toolchain
  checks. Raw upstream modules do not need custom attestation attributes.
- Compiled O-Voxel source is not vendored. Native conversion, `.vxz`, and voxel
  rendering were not run in the package CPU test matrix.
- `Trellis2PBRPostprocessFacade` requires O-Voxel, CuMesh, FlexGEMM, and
  nvdiffrast together. nvdiffrast is research/restricted and requires an
  explicit license acknowledgement; it is never invoked silently.
- Production `facebook/dinov3-vitl16-pretrain-lvd1689m` weights are gated on
  the Hub and governed by the separate DINOv3 License. The package does not
  redistribute them.
- Restricted nvdiffrast/nvdiffrec source is excluded.

## Conversion and training

`diffusers-3d-convert-trellis2` consumes an official `pipeline.json`, local
component JSON/safetensors pairs, and a local compatible DINOv3 conditioner
folder. The default conversion includes only reviewed sparse-structure
components. `--include-experimental` accepts only matching portable-tiny
layouts and never treats production sparse checkpoints as compatible.

`Trellis2SparseStructureFlowRecipe` is registered for full-model training only
with precomputed dense sparse-structure latents and a frozen conditioner and
decoder:

`t = sigmoid(N(1, 1))`,
`x_t = (1-t)x0 + (sigma_min + (1-sigma_min)t)noise`,
target `(1-sigma_min)noise-x0`, model timestep `t*1000`, and conditioning
dropout probability `0.1`.

Tiny shape and texture SLAT recipes use uniform timesteps and precomputed
normalized coordinate-aligned sparse latents, but remain experimental and
unregistered. No LoRA or SC-VAE recipe is claimed.

## Explicit limitations

- No official full 4B checkpoint, full 1024 cascade, production-resolution GPU,
  compiled O-Voxel mesh conversion, voxel rendering, PBR GLB export, or visual
  quality run was performed.
- Official production parity is claimed only for the reviewed portable
  sparse-structure model/decoder equations measured by the pinned tiny tests,
  not for experimental SLAT or O-Voxel networks.
- Background removal and production DINOv3 checkpoint acquisition are outside
  the offline CPU contract.

See `LICENSE-MIT`, `NOTICE`, and `diffusers_3d_integration.json` for exact
source, evidence, backend, and license declarations.
