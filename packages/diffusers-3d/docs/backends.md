# Backend policy

Backends implement package-owned protocols and are classified as:

- `portable`: CPU-importable, ordinarily installable, and permissively licensed;
- `accelerated`: optional compiled or JIT GPU implementations with a tested compatibility matrix;
- `research_only`: exact-reference implementations that are restricted or unsuitable for automatic selection.

The package never imports optional backends at top level. Backend selection is explicit when multiple
implementations can change numerical or visual output.

## Supported portable adapters

`TrimeshBackend`, `ScikitImageBackend`, and `XAtlasBackend` implement CPU mesh I/O/processing, marching cubes, and
UV unwrapping respectively. Constructing an adapter explicitly selects its exact entry in `BACKEND_REGISTRY` before
the optional module is imported. Their NumPy/CPU conversions are non-differentiable.

`ScikitImageBackend.extract_surface` generically defaults to
`gradient_direction="ascent"` and `allow_degenerate=False`. Family integrations must pass different upstream
settings explicitly when required.

Trimesh I/O supports OBJ, PLY, GLB, and STL. The adapter rejects channels, transforms, or coordinate-system metadata
that a requested format cannot preserve instead of silently dropping them. XAtlas remaps vertex-aligned channels
through its returned vertex mapping and rejects alignment-ambiguous custom channels.

## Accelerated and reference adapters

- Geometry: CuMesh.
- Surface extraction: permissive Kaolin/FlexiCubes.
- Sparse compute: spconv and FlexGEMM.
- Gaussian rendering: gsplat.
- Native representations: O-Voxel conversion and codecs.
- Exact reference rendering: nvdiffrast, nvdiffrec render utilities, diffoctreerast, and model-specific rasterizers.

The TRELLIS.2 adapters intentionally expose only reviewed narrow API surfaces:

- `FlexGemmBackend` delegates submanifold sparse convolution and 3D grid sampling. `CuMeshBackend` delegates repair,
  simplify, narrow-band remesh, UV unwrap, BVH construction, and unsigned distance. The package pins FlexGEMM at
  `6dd94a859c26ee8246888502eada3dd8ad85532e` and CuMesh at
  `12289e1062f0603f2f0d0771b02e1395d247f26f`. Discovery requires matching `direct_url.json` VCS provenance, and
  the loaded runtime wrappers must attest to the pinned revision and caller-supplied build ID. Installable direct
  source records are in `requirements/backends/flex-gemm.txt` and `requirements/backends/cumesh.txt`.
- `OVoxelBackend` provides pure tensor schema conversion, official uint8 packing, and deterministic Morton-ordered
  NPZ without loading an extension. `.vxz`, flexible-dual-grid mesh extraction, and voxel rendering are separate
  native capabilities delegated to the O-Voxel API from pinned TRELLIS.2 revision
  `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. `.vxz` does not contain grid resolution metadata, so callers must
  supply it when reading. The pinned `o-voxel` subdirectory requirement is recorded in
  `requirements/backends/o-voxel.txt`; install it together with the two direct dependency records.
- The pinned `o_voxel` top-level package eagerly imports its nvdiffrast-dependent postprocess module. Native O-Voxel
  loading therefore requires explicit nvdiffrast license acknowledgement even for codec/conversion members. Pure
  schema and NPZ paths remain CPU-safe and never perform that import.
- `Trellis2PBRPostprocessFacade` gates the combined O-Voxel, CuMesh, FlexGEMM, and nvdiffrast path. It never runs
  during ordinary pipeline output or backend discovery.

These native TRELLIS.2 paths are adapter/API tested with CPU fakes. A production CUDA mesh, render, or GLB quality
run has not been performed and is not claimed.

`utils3d` means the EasternJournalist repository used by TRELLIS, not the unrelated PyPI distribution. It must be
installed from an audited pinned revision.

Research discovery also requires exact PEP 610 direct-URL provenance for
nvdiffrast `253ac4fcea7de5f396371124af597e6cc957bfae`,
diffoctreerast `b09c20b84ec3aace4729e6e18a613112320eca3a`,
mip-splatting `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`, and
nvdiffrec `a3e73909a01887c8a135235ff860dd23a045cc1b`. The
corresponding records are in `requirements/backends/`. A matching module and
distribution version without the pinned source record remain unavailable.

CUDA-specific spconv distributions cannot be selected with standard Python environment markers. Users must install
the wheel matching their CUDA environment before selecting the adapter.

nvdiffrast and other reference-only backends require separate license review and are never included by ordinary
extras.
