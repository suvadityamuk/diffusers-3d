# Portions of this file are derived from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file contains a clean package adapter; the compiled O-Voxel source is not vendored.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import BinaryIO

import numpy as np
import torch

from ..objects import CoordinateSystem, MeshAsset, OVoxelAsset
from .voxel_index import VoxelIndex

OVOXEL_REFERENCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
_LEGACY_OVOXEL_METADATA_PREFIX = "__diffusers_3d_ovoxel_"


def ovoxel_grid_transform(
    resolution: int | Sequence[int],
    aabb: torch.Tensor | Sequence[Sequence[float]] = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map native O-Voxel grid corners to its right-handed Z-up object AABB."""

    grid_size = torch.tensor(_resolution_tuple(resolution), device=device, dtype=dtype)
    bounds = torch.as_tensor(aabb, device=device, dtype=dtype)
    if bounds.shape != (2, 3) or not bool(torch.isfinite(bounds).all()) or bool((bounds[1] <= bounds[0]).any()):
        raise ValueError("aabb must contain finite increasing lower and upper 3D bounds")
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[range(3), range(3)] = (bounds[1] - bounds[0]) / grid_size
    transform[:3, 3] = bounds[0]
    return transform


def morton_encode_3d(coordinates: torch.Tensor) -> torch.Tensor:
    """Encode unsigned 10-bit XYZ coordinates into the official 30-bit Z-order code."""

    if (
        not isinstance(coordinates, torch.Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or coordinates.is_floating_point()
    ):
        raise ValueError("coordinates must have integer shape (num_voxels, 3)")
    if bool((coordinates < 0).any()) or bool((coordinates >= 1024).any()):
        raise ValueError("30-bit Morton encoding supports coordinates in [0, 1023]")
    values = coordinates.to(dtype=torch.int64)
    code = torch.zeros(values.shape[0], dtype=torch.int64, device=values.device)
    for bit in range(10):
        code |= ((values[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((values[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((values[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


def morton_decode_3d(code: torch.Tensor) -> torch.Tensor:
    """Decode official 30-bit Z-order codes into XYZ coordinates."""

    if not isinstance(code, torch.Tensor) or code.ndim != 1 or code.is_floating_point():
        raise ValueError("code must be a rank-one integer tensor")
    values = code.to(dtype=torch.int64)
    if bool((values < 0).any()) or bool((values >= 1 << 30).any()):
        raise ValueError("Morton codes must be unsigned 30-bit values")
    coordinates = torch.zeros(values.shape[0], 3, dtype=torch.int64, device=values.device)
    for bit in range(10):
        coordinates[:, 0] |= ((values >> (3 * bit)) & 1) << bit
        coordinates[:, 1] |= ((values >> (3 * bit + 1)) & 1) << bit
        coordinates[:, 2] |= ((values >> (3 * bit + 2)) & 1) << bit
    return coordinates


def _lexicographic_coordinate_order(coordinates: torch.Tensor) -> torch.Tensor:
    order = torch.arange(coordinates.shape[0], device=coordinates.device)
    for axis in (2, 1, 0):
        order = order[torch.argsort(coordinates[order, axis], stable=True)]
    return order


def _scalar_channel(value: torch.Tensor, count: int, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value[:, None]
    if value.shape != (count, 1):
        raise ValueError(f"{name} must have shape ({count}, 1)")
    return value


def _pack_unit(value: torch.Tensor, name: str) -> torch.Tensor:
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite floating-point tensor")
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return torch.round(value * 255).to(dtype=torch.uint8)


def _unpack_unit(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype is not torch.uint8:
        raise ValueError(f"packed official {name} must use torch.uint8")
    return value.to(dtype=torch.float32) / 255


def _resolution_tuple(resolution: int | Sequence[int] | torch.Tensor) -> tuple[int, int, int]:
    values = torch.as_tensor(resolution)
    if values.ndim == 0:
        values = values.repeat(3)
    if values.shape != (3,) or values.is_floating_point() or bool((values <= 0).any()):
        raise ValueError("resolution must be a positive integer or three positive integers")
    return tuple(int(item) for item in values.tolist())


def _serialization_resolution(asset: OVoxelAsset, *, format_name: str) -> tuple[int, int, int]:
    if bool((asset.active_coordinates < 0).any()):
        raise ValueError(f"{format_name} coordinates must be non-negative")
    if bool((asset.active_coordinates > torch.iinfo(torch.uint16).max).any()):
        raise ValueError(f"{format_name} coordinates must fit in uint16")
    resolution = asset.metadata.get("resolution")
    if resolution is None:
        resolution = (asset.active_coordinates.amax(dim=0) + 1).tolist()
    resolution_values = _resolution_tuple(resolution)
    if bool((asset.active_coordinates >= asset.active_coordinates.new_tensor(resolution_values)).any()):
        raise ValueError(f"{format_name} coordinates must be strictly below the declared resolution")
    return resolution_values


def _unpack_intersection_flags(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value[:, None]
    if value.shape == (count, 1):
        if value.is_floating_point():
            raise ValueError("packed intersected flags must use an integer dtype")
        if bool(((value < 0) | (value > 7)).any()):
            raise ValueError("packed intersected flags must contain only the low three axis bits")
        return torch.cat(
            [
                value & 1,
                (value >> 1) & 1,
                (value >> 2) & 1,
            ],
            dim=1,
        ).bool()
    if value.shape == (count, 3):
        if bool(((value != 0) & (value != 1)).any()):
            raise ValueError("unpacked intersected flags must contain only zero or one")
        return value.bool()
    raise ValueError("intersected must have shape (num_voxels, 1) or (num_voxels, 3)")


def ovoxel_asset_from_official(
    coordinates: torch.Tensor,
    attributes: Mapping[str, torch.Tensor],
    *,
    resolution: int | Sequence[int] | None = None,
    aabb: torch.Tensor | Sequence[Sequence[float]] = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    packed: bool | None = None,
) -> OVoxelAsset:
    """Map official O-Voxel tensors into the package-owned lossless schema."""

    if (
        not isinstance(coordinates, torch.Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or coordinates.is_floating_point()
        or coordinates.shape[0] == 0
    ):
        raise ValueError("coordinates must have non-empty integer shape (num_voxels, 3)")
    if bool((coordinates < 0).any()):
        raise ValueError("coordinates must be non-negative")
    values = dict(attributes)
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in values.items()):
        raise TypeError("attributes must map string names to tensors")
    count = coordinates.shape[0]
    if any(value.ndim == 0 or value.shape[0] != count for value in values.values()):
        raise ValueError("every official attribute must align with coordinates")
    dual_key = "dual_vertices" if "dual_vertices" in values else "vertices"
    if dual_key not in values or "intersected" not in values:
        raise ValueError("official O-Voxel data requires dual_vertices (or vertices) and intersected")

    def unit(name: str, *, default: torch.Tensor | None = None) -> torch.Tensor | None:
        value = values.get(name, default)
        if value is None:
            return None
        value_is_packed = packed is not False and value.dtype is torch.uint8
        if value_is_packed:
            return _unpack_unit(value, name)
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"unpacked official {name} must be a finite floating-point tensor")
        return value.to(dtype=torch.float32)

    dual_vertices = unit(dual_key)
    assert dual_vertices is not None
    if dual_vertices.shape != (count, 3):
        raise ValueError("dual_vertices must have shape (num_voxels, 3)")
    intersected_bits = _unpack_intersection_flags(values["intersected"], count)

    default_dtype = torch.uint8 if packed is True else torch.float32
    default_base = torch.zeros(count, 3, dtype=default_dtype, device=coordinates.device)
    default_metallic = torch.zeros(count, 1, dtype=default_dtype, device=coordinates.device)
    default_roughness = torch.full(
        (count, 1),
        128 if packed is True else 0.5,
        dtype=default_dtype,
        device=coordinates.device,
    )
    base_color = unit("base_color", default=default_base)
    metallic = unit("metallic", default=default_metallic)
    roughness = unit("roughness", default=default_roughness)
    assert base_color is not None and metallic is not None and roughness is not None
    metallic = _scalar_channel(metallic, count, "metallic")
    roughness = _scalar_channel(roughness, count, "roughness")
    opacity = unit("alpha")
    if opacity is not None:
        opacity = _scalar_channel(opacity, count, "alpha")
    emissive = unit("emissive")
    normals = unit("normal")
    if normals is not None:
        # Official tensors store normals in [0, 1] (packed or not); OVoxelAsset stores signed normals.
        normals = normals * 2 - 1
    split_weights = values.get("split_weight", values.get("split_weights"))
    if split_weights is not None:
        split_weights = _scalar_channel(split_weights, count, "split_weight")
        if packed is not False and split_weights.dtype is torch.uint8:
            # Legacy/native VXZ data stores uint8 split weights; new writes keep them floating point.
            split_weights = _unpack_unit(split_weights, "split_weight")
        elif not split_weights.is_floating_point() or not bool(torch.isfinite(split_weights).all()):
            raise ValueError("split_weight must be a finite floating-point tensor")
        if bool((split_weights < 0).any()):
            raise ValueError("split_weight values must be non-negative")

    if resolution is None:
        inferred = coordinates.amax(dim=0).to(dtype=torch.int64) + 1
        resolution_values = tuple(int(item) for item in inferred.tolist())
        resolution_inferred = True
    else:
        resolution_values = _resolution_tuple(resolution)
        resolution_inferred = False
    resolution_tensor = coordinates.new_tensor(resolution_values)
    if bool((coordinates >= resolution_tensor).any()):
        raise ValueError("coordinates fall outside resolution")
    bounds = torch.as_tensor(aabb, device=base_color.device, dtype=base_color.dtype)
    metadata = {
        "family": "trellis2",
        "representation": "o_voxel",
        "resolution": list(resolution_values),
        "aabb": bounds.detach().cpu().tolist(),
        "resolution_inferred": resolution_inferred,
        "official_packed_input": all(
            value.dtype is torch.uint8
            for name, value in values.items()
            if name not in {"split_weight", "split_weights"}
        ),
        "dual_vertex_semantics": "fractional_cell_offset",
    }
    return OVoxelAsset(
        active_coordinates=coordinates,
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        dual_grid_vertex_offsets=dual_vertices,
        intersection_data=intersected_bits,
        split_weights=split_weights,
        opacity=opacity,
        normals=normals,
        emissive=emissive,
        grid_transform=ovoxel_grid_transform(
            resolution_values,
            bounds,
            device=base_color.device,
            dtype=base_color.dtype,
        ),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        metadata=metadata,
    )


def official_tensors_from_ovoxel_asset(
    asset: OVoxelAsset,
    *,
    packed: bool = True,
    morton_order: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Map an O-Voxel asset to official aligned tensors and optional uint8 packing."""

    if type(asset) is not OVoxelAsset:
        raise TypeError("asset must be an exact OVoxelAsset")
    asset.validate(expensive=True)
    if asset.coordinate_system is not CoordinateSystem.RIGHT_HANDED_Z_UP:
        raise ValueError("official O-Voxel tensors require the native right-handed Z-up coordinate system")
    if asset.dual_grid_vertex_offsets is None or asset.dual_grid_vertex_offsets.ndim != 2:
        raise ValueError("official O-Voxel mapping requires one fractional dual vertex per active voxel")
    if asset.dual_grid_vertex_offsets.shape != asset.active_coordinates.shape:
        raise ValueError("dual-grid vertices must align one-to-one with active coordinates")
    if asset.intersection_data is None:
        raise ValueError("official O-Voxel mapping requires intersection flags")
    flags = _unpack_intersection_flags(asset.intersection_data, asset.active_coordinates.shape[0])
    bits = flags.to(dtype=torch.uint8)
    intersected = bits[:, 0:1] + 2 * bits[:, 1:2] + 4 * bits[:, 2:3]

    attributes: dict[str, torch.Tensor] = {
        "dual_vertices": asset.dual_grid_vertex_offsets,
        "intersected": intersected,
        "base_color": asset.base_color,
        "metallic": asset.metallic.reshape(-1, 1),
        "roughness": asset.roughness.reshape(-1, 1),
    }
    if asset.opacity is not None:
        attributes["alpha"] = asset.opacity.reshape(-1, 1)
    if asset.normals is not None:
        attributes["normal"] = asset.normals * 0.5 + 0.5
    if asset.emissive is not None:
        attributes["emissive"] = asset.emissive
    if asset.split_weights is not None:
        attributes["split_weight"] = asset.split_weights.reshape(-1, 1)
    # NumPy has no bfloat16; float32 holds every bfloat16 value exactly, so a bf16 pipeline output still serializes.
    attributes = {
        name: value.to(dtype=torch.float32) if value.dtype == torch.bfloat16 else value
        for name, value in attributes.items()
    }
    if packed:
        packed_attributes = {}
        for name, value in attributes.items():
            if name == "intersected":
                packed_attributes[name] = value.to(dtype=torch.uint8)
            elif name == "dual_vertices":
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError("packed official dual_vertices must be finite and floating point")
                if bool(((value < -0.5) | (value > 1.5)).any()):
                    raise ValueError("packed official dual_vertices must lie in [-0.5, 1.5]")
                if bool(((value < 0) | (value > 1)).any()):
                    if value.dtype not in (torch.float16, torch.float32):
                        raise ValueError(
                            "out-of-cell packed official dual_vertices must use torch.float16 or torch.float32"
                        )
                    packed_attributes[name] = value
                else:
                    packed_attributes[name] = _pack_unit(value, name)
            elif name == "split_weight":
                if value.dtype not in (torch.float16, torch.float32) or not bool(torch.isfinite(value).all()):
                    raise ValueError("packed official split_weight must use finite torch.float16 or torch.float32")
                if bool((value < 0).any()):
                    raise ValueError("packed official split_weight must be non-negative")
                packed_attributes[name] = value
            else:
                packed_attributes[name] = _pack_unit(value, name)
        attributes = packed_attributes
    elif any(not value.is_floating_point() for name, value in attributes.items() if name != "intersected"):
        raise ValueError("unpacked official attributes must be floating-point tensors")

    coordinates = asset.active_coordinates
    if morton_order:
        order = torch.argsort(morton_encode_3d(coordinates), stable=True)
        coordinates = coordinates[order]
        attributes = {name: value[order] for name, value in attributes.items()}
    return coordinates, attributes


def write_ovoxel_npz(
    file: str | PathLike[str] | BinaryIO,
    asset: OVoxelAsset,
    *,
    compressed: bool = True,
    packed: bool = True,
    morton_order: bool | None = None,
) -> None:
    """Write an NPZ holding only fields the official reader accepts as attributes.

    Rows are sorted lexicographically by default, which supports any uint16 coordinate; ``morton_order=True`` uses
    the official 30-bit order instead and limits coordinates to 1023. Resolution and AABB are not stored, so pass
    them to :func:`read_ovoxel_npz` when they cannot be inferred.
    """

    resolution = _serialization_resolution(asset, format_name="official O-Voxel NPZ")
    if morton_order is not None and not isinstance(morton_order, bool):
        raise TypeError("morton_order must be a bool or None")
    coordinates, attributes = official_tensors_from_ovoxel_asset(
        asset,
        packed=packed,
        morton_order=bool(morton_order),
    )
    if morton_order is None:
        order = _lexicographic_coordinate_order(coordinates)
        coordinates = coordinates[order]
        attributes = {name: value[order] for name, value in attributes.items()}
    if "split_weight" in attributes and attributes["split_weight"].dtype not in (torch.float16, torch.float32):
        raise ValueError("O-Voxel NPZ split_weight must use torch.float16 or torch.float32")
    aabb = asset.metadata.get("aabb", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    expected_grid_transform = ovoxel_grid_transform(
        resolution,
        aabb,
        device=asset.device,
        dtype=asset.grid_transform.dtype,
    )
    if not torch.allclose(asset.grid_transform, expected_grid_transform):
        raise ValueError("asset grid_transform does not match its native O-Voxel resolution and aabb metadata")
    arrays: dict[str, np.ndarray] = {"coord": coordinates.detach().cpu().numpy().astype(np.uint16)}
    arrays.update({name: value.detach().cpu().numpy() for name, value in attributes.items()})
    writer = np.savez_compressed if compressed else np.savez
    writer(file, **arrays)


def read_ovoxel_npz(
    file: str | PathLike[str] | BinaryIO,
    *,
    resolution: int | Sequence[int] | None = None,
    aabb: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> OVoxelAsset:
    """Read O-Voxel NPZ data without importing the compiled runtime."""

    with np.load(file, allow_pickle=False) as data:
        if "coord" not in data:
            raise ValueError("O-Voxel NPZ is missing coord")
        coordinate_array = np.array(data["coord"], copy=True)
        coordinates = torch.from_numpy(coordinate_array.astype(np.int32))
        attributes = {
            name: torch.from_numpy(np.array(data[name], copy=True))
            for name in data.files
            if name != "coord" and not name.startswith(_LEGACY_OVOXEL_METADATA_PREFIX)
        }
        stored_resolution = data.get(f"{_LEGACY_OVOXEL_METADATA_PREFIX}resolution")
        stored_aabb = data.get(f"{_LEGACY_OVOXEL_METADATA_PREFIX}aabb")
        stored_packed = data.get(f"{_LEGACY_OVOXEL_METADATA_PREFIX}packed")
        if resolution is None and stored_resolution is not None:
            resolution = tuple(int(item) for item in stored_resolution.tolist())
        if aabb is None and stored_aabb is not None:
            aabb = stored_aabb.tolist()
        packed = bool(stored_packed[0]) if stored_packed is not None else None
    return ovoxel_asset_from_official(
        coordinates,
        attributes,
        resolution=resolution,
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)) if aabb is None else aabb,
        packed=packed,
    )


_EDGE_NEIGHBOR_OFFSETS = torch.tensor(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ]
)
_QUAD_SPLITS = torch.tensor([[0, 1, 2, 0, 2, 3], [0, 1, 3, 3, 1, 2]])


def _split_alignment(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    # Upstream scores a split from the first four entries of its six-index triangle list; kept for parity.
    corners = vertices[triangles[:, :4]]
    first = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1)
    second = torch.cross(corners[:, 2] - corners[:, 1], corners[:, 3] - corners[:, 1], dim=1)
    return (first * second).sum(dim=1).abs()


def ovoxel_to_mesh(asset: OVoxelAsset) -> MeshAsset:
    """Extract an O-Voxel's dual-grid surface as a triangle mesh coloured by its base colour and opacity.

    A plain-PyTorch port of upstream ``o_voxel.convert.flexible_dual_grid_to_mesh``: one vertex per active voxel,
    and one quad, split into two triangles, around every crossed voxel edge whose four voxels are all active.
    """

    resolution, aabb = asset.metadata.get("resolution"), asset.metadata.get("aabb")
    if resolution is None or aabb is None:
        raise ValueError("mesh extraction requires explicit resolution and aabb metadata")
    if asset.dual_grid_vertex_offsets is None or asset.intersection_data is None:
        raise ValueError("mesh extraction requires dual-grid vertices and intersection flags")
    coordinates = asset.active_coordinates
    device = coordinates.device
    grid_size = _resolution_tuple(resolution)
    aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    voxel_size = (aabb[1] - aabb[0]) / torch.tensor(grid_size, dtype=torch.int32, device=device)
    vertices = (coordinates.float() + asset.dual_grid_vertex_offsets.float()) * voxel_size + aabb[0]

    flags = _unpack_intersection_flags(asset.intersection_data, coordinates.shape[0])
    edges = coordinates[:, None, None, :] + _EDGE_NEIGHBOR_OFFSETS.to(device)
    quads = VoxelIndex(coordinates, grid_size).lookup(edges[flags])
    quads = quads[(quads >= 0).all(dim=1)]
    first_split, second_split = (quads[:, split] for split in _QUAD_SPLITS.to(device))
    if asset.split_weights is None:
        use_first = _split_alignment(vertices, first_split) > _split_alignment(vertices, second_split)
    else:
        weights = asset.split_weights.float().reshape(-1)[quads]
        use_first = weights[:, 0] * weights[:, 2] > weights[:, 1] * weights[:, 3]
    faces = torch.where(use_first[:, None], first_split, second_split).reshape(-1, 3)
    if faces.numel() == 0:
        raise ValueError("the O-Voxel has no surface: no crossed voxel edge has all four of its voxels active")

    colors = asset.base_color
    if asset.opacity is not None:
        colors = torch.cat([colors, asset.opacity.reshape(-1, 1)], dim=1)
    return MeshAsset(
        vertices=vertices,
        faces=faces,
        transform=asset.transform.to(device=device, dtype=vertices.dtype),
        coordinate_system=asset.coordinate_system,
        colors=colors.to(vertices.dtype),
        metadata={
            "source": "o_voxel",
            "reference_revision": OVOXEL_REFERENCE_REVISION,
            "resolution": list(grid_size),
            "aabb": aabb.tolist(),
        },
    )


__all__ = [
    "OVOXEL_REFERENCE_REVISION",
    "morton_decode_3d",
    "morton_encode_3d",
    "official_tensors_from_ovoxel_asset",
    "ovoxel_to_mesh",
    "ovoxel_grid_transform",
    "ovoxel_asset_from_official",
    "read_ovoxel_npz",
    "write_ovoxel_npz",
]
