# TRELLIS.2 image-to-3D

This family integrates the MIT-licensed Microsoft TRELLIS.2 implementation at
revision `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. It is a distinct
`trellis2` family. The whole released network stack (sparse-structure stage,
512/1024 shape and texture SLAT flows, shape and PBR decoders) runs in plain
PyTorch; mesh conversion and PBR/GLB postprocess delegate to the compiled
O-Voxel runtime and stay capability-gated.

A runnable end-to-end example lives in
[`examples/image_to_3d.py`](examples/image_to_3d.py); see
[docs/inference.md](../../../../docs/inference.md) and
[docs/finetuning.md](../../../../docs/finetuning.md) for the usage guides.

## Reviewed sparse-structure path

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
- `Trellis2ImageTo3DPipeline` round-trips through standard Diffusers save/load
  and `AutoPipelineForImageTo3D`. Released defaults are 12 steps with the
  per-stage strengths, rescale values, intervals, and 1024 cascade
  configuration serialized in the pipeline config. The upstream sparse target
  mapping is preserved: `512`, `1024_cascade`, and `1536_cascade` pool decoded
  occupancy by 2, while `1024` keeps the decoder's native grid.
- Typed RGBA alpha and separate masks are quantized to uint8 before the pinned
  `>0.8 * 255` foreground crop. The cropped RGBA image is premultiplied on
  black before Pillow LANCZOS resizes RGB to the conditioner size, matching
  the released ordering. Unmasked RGB is treated as an already
  background-removed full frame; the pipeline never invokes a background
  remover silently.

## O-Voxel object and codecs

`OVoxelAsset` retains active XYZ coordinates, fractional dual vertices,
three-axis intersection flags, split weights, base color, metallic, roughness,
alpha, signed normals, emissive, the native right-handed Z-up grid transform,
resolution, and AABB.

The package-owned O-Voxel adapter has independent capability surfaces:

- schema conversion and mixed official packing are pure PyTorch: unit-domain
  channels use uint8 while out-of-cell dual vertices and unbounded split
  weights retain float16/float32;
- `.npz` read/write is pure NumPy/Python and defaults to deterministic
  lexicographic ordering across the uint16 coordinate domain. Files contain
  only fields accepted by the official reader; pass resolution and AABB back
  to the package reader when they cannot be inferred. Explicit 30-bit Morton
  ordering remains available when every coordinate is at most 1023;
- `.vxz` read/write delegates unsorted global coordinates to
  `o_voxel.io.read_vxz`/`write_vxz`, whose official runtime performs
  chunk-local ordering;
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

## SLAT flows and O-Voxel decoders

`Trellis2SLatFlowModel` is the released full-attention sparse transformer with
RoPE over voxel coordinates, shared modulation, and Q/K RMS norms. The texture
form takes a coordinate-aligned `concat_cond` shape SLAT. The 512 and 1024
checkpoints load with `strict=True`; the `rope_phases` buffer is derived from
the config and is not part of the state dict, matching upstream.

`Trellis2ShapeDualGridDecoder` and `Trellis2PBRSparseDecoder` are the released
sparse UNets: ConvNeXt blocks, channel-to-spatial upsampling, and a
subdivision head per stage that predicts which children to keep. The PBR
decoder reuses the shape decoder's subdivision masks so both write the same
cells of one `OVoxelAsset`. `upsample_coordinates` runs the shape decoder's
first stages to grow the grid for the cascade presets. Sparse convolution,
pooling, and subdivision come from `families/trellis/sparse_ops.py` and run
on any device; tiny outputs match the pinned upstream code with FlexGEMM and
xformers replaced by dense PyTorch in the test.

`formats` accepts `sparse_structure`, `shape_slat`, `texture_slat`, `o_voxel`,
and `mesh` (through the explicit O-Voxel backend). GLB postprocess is not a
`formats` value: call `postprocess_ovoxel(asset, output_format="glb")` when
the optional, license-gated stack is available.

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
folder. It converts every released component (conditioner, sparse-structure
flow and decoder, the 512 and 1024 shape and texture SLAT flows, and the shape
and PBR decoders); a release that ships only the sparse-structure stage is
also accepted.

`Trellis2SparseStructureFlowRecipe` is registered for full-model training only
with precomputed dense sparse-structure latents and a frozen conditioner and
decoder:

`t = sigmoid(N(1, 1))`,
`x_t = (1-t)x0 + (sigma_min + (1-sigma_min)t)noise`,
target `(1-sigma_min)noise-x0`, model timestep `t*1000`, and conditioning
dropout probability `0.1`.

The shape and texture SLAT recipes use uniform timesteps and precomputed
normalized coordinate-aligned sparse latents; they run against the ported
flow models but are not registered yet. All recipe collators separately follow the pinned dataset
transform exactly once: the bbox includes every nonzero alpha pixel, uses the
unscaled floating half-size before integer truncation, resizes RGBA with
LANCZOS, and multiplies the resized RGB and alpha tensors. Separate masks
participate in alpha. No LoRA or SC-VAE recipe is claimed.

## Explicit limitations

- No official full 4B checkpoint, production-resolution GPU, compiled O-Voxel
  mesh conversion, voxel rendering, PBR GLB export, or visual quality run was
  performed in this package's test matrix.
- Parity is measured with tiny weights against the pinned upstream code on
  CPU, for every network. Released state-dict layouts are checked against the
  published safetensors headers.
- Background removal and production DINOv3 checkpoint acquisition are outside
  the offline CPU contract.

See `LICENSE-MIT`, `NOTICE`, and `diffusers_3d_integration.json` for exact
source, evidence, backend, and license declarations.
