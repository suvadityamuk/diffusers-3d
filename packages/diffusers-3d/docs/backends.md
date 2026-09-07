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

Trimesh I/O supports OBJ, PLY, GLB, and STL. The adapter rejects channels, transforms, or coordinate-system metadata
that a requested format cannot preserve instead of silently dropping them. XAtlas remaps vertex-aligned channels
through its returned vertex mapping and rejects alignment-ambiguous custom channels.

## Planned accelerated and reference adapters

- Geometry: CuMesh.
- Surface extraction: permissive Kaolin/FlexiCubes.
- Sparse compute: spconv and FlexGEMM.
- Gaussian rendering: gsplat.
- Native representations: O-Voxel conversion and codecs.
- Exact reference rendering: nvdiffrast, nvdiffrec render utilities, diffoctreerast, and model-specific rasterizers.

`utils3d` means the EasternJournalist repository used by TRELLIS, not the unrelated PyPI distribution. It must be
installed from an audited pinned revision.

CUDA-specific spconv distributions cannot be selected with standard Python environment markers. Users must install
the wheel matching their CUDA environment before selecting the adapter.

nvdiffrast and other reference-only backends require separate license review and are never included by ordinary
extras.
