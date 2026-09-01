# Portions of this file are derived from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file contains a clean package adapter; the compiled O-Voxel source is not vendored.

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from os import PathLike, fspath
from types import MappingProxyType, ModuleType
from typing import Any, BinaryIO

import numpy as np
import torch

from ..objects import CoordinateSystem, MeshAsset, OVoxelAsset
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability

OVOXEL_REFERENCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
OVOXEL_METADATA_PREFIX = "__diffusers_3d_ovoxel_"
_OVOXEL_NPZ_SCHEMA_VERSION = 2
_COORDINATE_ORDERS = frozenset({"input", "lexicographic_xyz", "morton_30bit"})
_ATTRIBUTE_LAYOUTS = {
    "alpha": "voxel_scalar",
    "base_color": "voxel_rgb",
    "dual_vertices": "voxel_xyz_offset",
    "emissive": "voxel_rgb",
    "intersected": "voxel_xyz_bitfield",
    "metallic": "voxel_scalar",
    "normal": "voxel_xyz_unit_encoded",
    "roughness": "voxel_scalar",
    "split_weight": "voxel_scalar_nonnegative",
}


class OVoxelRuntimeUnavailableError(RuntimeError):
    """Raised when an operation requires the separately built O-Voxel runtime."""


class OVoxelCapability(str, Enum):
    """Independently discoverable O-Voxel adapter features."""

    SCHEMA_PACK = "schema_pack"
    NPZ_CODEC = "npz_codec"
    NATIVE_CONVERSION = "native_conversion"
    NATIVE_CODEC = "native_codec"
    NATIVE_RENDERING = "native_rendering"


_NATIVE_CAPABILITY_MEMBERS = {
    OVoxelCapability.NATIVE_CONVERSION: ("convert.flexible_dual_grid_to_mesh",),
    OVoxelCapability.NATIVE_CODEC: ("io.read_vxz", "io.write_vxz"),
    OVoxelCapability.NATIVE_RENDERING: ("rasterize.VoxelRenderer",),
}


def ovoxel_grid_transform(
    resolution: int | Sequence[int],
    aabb: torch.Tensor | Sequence[Sequence[float]] = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map native O-Voxel grid corners to its right-handed Z-up object AABB."""

    grid_size = torch.as_tensor(resolution, device=device)
    if grid_size.ndim == 0:
        grid_size = grid_size.repeat(3)
    if grid_size.shape != (3,) or grid_size.is_floating_point() or bool((grid_size <= 0).any()):
        raise ValueError("resolution must be a positive integer or three positive integers")
    bounds = torch.as_tensor(aabb, device=device, dtype=dtype)
    if bounds.shape != (2, 3) or not bool(torch.isfinite(bounds).all()) or bool((bounds[1] <= bounds[0]).any()):
        raise ValueError("aabb must contain finite increasing lower and upper 3D bounds")
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[range(3), range(3)] = (bounds[1] - bounds[0]) / grid_size.to(dtype=dtype)
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


def _validate_recorded_coordinate_order(coordinates: torch.Tensor, coordinate_order: str) -> None:
    if coordinate_order == "input":
        return
    if coordinate_order == "morton_30bit":
        keys = morton_encode_3d(coordinates)
        ordered = bool((keys[1:] >= keys[:-1]).all())
    else:
        order = _lexicographic_coordinate_order(coordinates)
        ordered = torch.equal(order, torch.arange(coordinates.shape[0], device=coordinates.device))
    if not ordered:
        raise ValueError(f"O-Voxel NPZ coordinates do not match recorded {coordinate_order!r} ordering")


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


def _validate_serializable_coordinates(
    coordinates: torch.Tensor,
    resolution: tuple[int, int, int],
    *,
    format_name: str,
) -> None:
    if bool((coordinates < 0).any()):
        raise ValueError(f"{format_name} coordinates must be non-negative")
    if bool((coordinates > torch.iinfo(torch.uint16).max).any()):
        raise ValueError(f"{format_name} coordinates must fit in uint16")
    resolution_tensor = coordinates.new_tensor(resolution)
    if bool((coordinates >= resolution_tensor).any()):
        raise ValueError(f"{format_name} coordinates must be strictly below the declared resolution")


def _serialization_resolution(asset: OVoxelAsset, *, format_name: str) -> tuple[int, int, int]:
    if bool((asset.active_coordinates < 0).any()):
        raise ValueError(f"{format_name} coordinates must be non-negative")
    if bool((asset.active_coordinates > torch.iinfo(torch.uint16).max).any()):
        raise ValueError(f"{format_name} coordinates must fit in uint16")
    resolution = asset.metadata.get("resolution")
    if resolution is None:
        resolution = (asset.active_coordinates.amax(dim=0) + 1).tolist()
    resolution_values = _resolution_tuple(resolution)
    _validate_serializable_coordinates(asset.active_coordinates, resolution_values, format_name=format_name)
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
    if packed is None:
        packed = values[dual_key].dtype is torch.uint8

    def unit(name: str, *, default: torch.Tensor | None = None) -> torch.Tensor | None:
        value = values.get(name, default)
        if value is None:
            return None
        return _unpack_unit(value, name) if packed else value.to(dtype=torch.float32)

    dual_vertices = unit(dual_key)
    assert dual_vertices is not None
    if dual_vertices.shape != (count, 3):
        raise ValueError("dual_vertices must have shape (num_voxels, 3)")
    intersected_bits = _unpack_intersection_flags(values["intersected"], count)

    default_base = torch.zeros(count, 3, dtype=torch.uint8 if packed else torch.float32, device=coordinates.device)
    default_metallic = torch.zeros(count, 1, dtype=torch.uint8 if packed else torch.float32, device=coordinates.device)
    default_roughness = torch.full(
        (count, 1),
        128 if packed else 0.5,
        dtype=torch.uint8 if packed else torch.float32,
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
        # Official tensors store normals in the same [0, 1] channel domain
        # before and after uint8 packing. OVoxelAsset stores signed normals.
        normals = normals * 2 - 1
    split_weights = values.get("split_weight", values.get("split_weights"))
    if split_weights is not None:
        split_weights = _scalar_channel(split_weights, count, "split_weight")
        if packed and split_weights.dtype is torch.uint8:
            # Read legacy/native VXZ data according to the only value domain
            # that uint8 can represent. New lossless writes keep this channel
            # floating point.
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
        "official_packed_input": bool(packed),
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
    flags = asset.intersection_data
    if flags.ndim == 1:
        flags = flags[:, None]
    if flags.shape[1] == 1:
        intersected = flags.to(dtype=torch.uint8)
    elif flags.shape[1] == 3:
        bits = flags.to(dtype=torch.uint8)
        if bool((bits > 1).any()):
            raise ValueError("unpacked intersection flags must contain only zero or one")
        intersected = bits[:, 0:1] + 2 * bits[:, 1:2] + 4 * bits[:, 2:3]
    else:
        raise ValueError("intersection flags must have one packed or three unpacked channels")

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
    if packed:
        packed_attributes = {}
        for name, value in attributes.items():
            if name == "intersected":
                packed_attributes[name] = value.to(dtype=torch.uint8)
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
    """Write an official-compatible NPZ plus reserved lossless grid metadata.

    The default deterministic lexicographic order supports the full uint16
    coordinate domain. Passing ``morton_order=True`` explicitly requests the
    official 30-bit order and therefore limits every coordinate to 1023.
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
        coordinate_order = "lexicographic_xyz"
    else:
        coordinate_order = "morton_30bit" if morton_order else "input"
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
    attribute_metadata = {
        "attributes": {
            name: {
                "dtype": str(value.dtype),
                "encoding": (
                    "xyz_bitfield_uint8"
                    if name == "intersected"
                    else "nonnegative_float"
                    if name == "split_weight"
                    else "unit_uint8"
                    if packed
                    else "unit_float"
                ),
                "layout": _ATTRIBUTE_LAYOUTS.get(name, "voxel_channels"),
                "shape": list(value.shape[1:]),
            }
            for name, value in sorted(arrays.items())
            if name != "coord"
        },
        "coordinate_dtype": "uint16",
        "coordinate_layout": "voxel_xyz",
        "coordinate_order": coordinate_order,
        "morton_order": coordinate_order == "morton_30bit",
        "schema_version": _OVOXEL_NPZ_SCHEMA_VERSION,
    }
    arrays[f"{OVOXEL_METADATA_PREFIX}resolution"] = np.asarray(resolution, dtype=np.uint32)
    arrays[f"{OVOXEL_METADATA_PREFIX}aabb"] = np.asarray(aabb, dtype=np.float32)
    arrays[f"{OVOXEL_METADATA_PREFIX}packed"] = np.asarray([int(packed)], dtype=np.uint8)
    arrays[f"{OVOXEL_METADATA_PREFIX}layout"] = np.asarray(
        json.dumps(attribute_metadata, separators=(",", ":"), sort_keys=True)
    )
    writer = np.savez_compressed if compressed else np.savez
    writer(file, **arrays)


def read_ovoxel_npz(
    file: str | PathLike[str] | BinaryIO,
    *,
    resolution: int | Sequence[int] | None = None,
    aabb: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> OVoxelAsset:
    """Read O-Voxel NPZ data without importing the compiled runtime."""

    coordinate_order = None
    with np.load(file, allow_pickle=False) as data:
        if "coord" not in data:
            raise ValueError("O-Voxel NPZ is missing coord")
        coordinate_array = np.array(data["coord"], copy=True)
        coordinates = torch.from_numpy(coordinate_array.astype(np.int32))
        attributes = {
            name: torch.from_numpy(np.array(data[name], copy=True))
            for name in data.files
            if name != "coord" and not name.startswith(OVOXEL_METADATA_PREFIX)
        }
        stored_resolution = data.get(f"{OVOXEL_METADATA_PREFIX}resolution")
        stored_aabb = data.get(f"{OVOXEL_METADATA_PREFIX}aabb")
        stored_packed = data.get(f"{OVOXEL_METADATA_PREFIX}packed")
        stored_layout = data.get(f"{OVOXEL_METADATA_PREFIX}layout")
        packed = bool(stored_packed[0]) if stored_packed is not None else None
        if stored_layout is not None:
            try:
                layout_metadata = json.loads(str(stored_layout.item()))
                expected_attributes = layout_metadata["attributes"]
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("O-Voxel NPZ has invalid dtype/layout metadata") from error
            if (
                layout_metadata.get("schema_version") != _OVOXEL_NPZ_SCHEMA_VERSION
                or layout_metadata.get("coordinate_dtype") != "uint16"
                or layout_metadata.get("coordinate_layout") != "voxel_xyz"
                or layout_metadata.get("coordinate_order") not in _COORDINATE_ORDERS
                or layout_metadata.get("morton_order")
                is not (layout_metadata.get("coordinate_order") == "morton_30bit")
                or coordinate_array.dtype != np.uint16
                or not isinstance(expected_attributes, dict)
                or set(expected_attributes) != set(attributes)
            ):
                raise ValueError("O-Voxel NPZ dtype/layout metadata does not match its arrays")
            coordinate_order = layout_metadata["coordinate_order"]
            _validate_recorded_coordinate_order(coordinates, coordinate_order)
            for name, value in attributes.items():
                description = expected_attributes[name]
                expected_encoding = (
                    "xyz_bitfield_uint8"
                    if name == "intersected"
                    else "nonnegative_float"
                    if name == "split_weight"
                    else "unit_uint8"
                    if packed
                    else "unit_float"
                )
                if (
                    not isinstance(description, dict)
                    or description.get("dtype") != str(value.numpy().dtype)
                    or description.get("encoding") != expected_encoding
                    or description.get("shape") != list(value.shape[1:])
                    or description.get("layout") != _ATTRIBUTE_LAYOUTS.get(name, "voxel_channels")
                ):
                    raise ValueError(f"O-Voxel NPZ dtype/layout metadata does not match attribute {name!r}")
        if resolution is None and stored_resolution is not None:
            resolution = tuple(int(item) for item in stored_resolution.tolist())
        if aabb is None and stored_aabb is not None:
            aabb = stored_aabb.tolist()
    asset = ovoxel_asset_from_official(
        coordinates,
        attributes,
        resolution=resolution,
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)) if aabb is None else aabb,
        packed=packed,
    )
    if coordinate_order is not None:
        asset.metadata["coordinate_order"] = coordinate_order
    return asset


class OVoxelBackend:
    """Portable schema/NPZ adapter with explicitly lazy native O-Voxel operations."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        accept_nvdiffrast_research_license: bool = False,
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        if not isinstance(accept_nvdiffrast_research_license, bool):
            raise TypeError("accept_nvdiffrast_research_license must be a bool")
        self.device = torch.device(device)
        self.dtype = dtype
        self.accept_nvdiffrast_research_license = accept_nvdiffrast_research_license
        self.registry = registry
        self._runtime: ModuleType | None = None

    @property
    def capabilities(self) -> Mapping[OVoxelCapability, bool]:
        capabilities = {
            OVoxelCapability.SCHEMA_PACK: True,
            OVoxelCapability.NPZ_CODEC: True,
        }
        capabilities.update(
            {
                capability: self._supports_native_members(members)
                for capability, members in _NATIVE_CAPABILITY_MEMBERS.items()
            }
        )
        return MappingProxyType(capabilities)

    def supports(self, capability: OVoxelCapability | str) -> bool:
        return self.capabilities[OVoxelCapability(capability)]

    @staticmethod
    def from_official(
        coordinates: torch.Tensor,
        attributes: Mapping[str, torch.Tensor],
        **kwargs: Any,
    ) -> OVoxelAsset:
        return ovoxel_asset_from_official(coordinates, attributes, **kwargs)

    @staticmethod
    def to_official(
        asset: OVoxelAsset,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return official_tensors_from_ovoxel_asset(asset, **kwargs)

    @staticmethod
    def read_npz(file: str | PathLike[str] | BinaryIO, **kwargs: Any) -> OVoxelAsset:
        return read_ovoxel_npz(file, **kwargs)

    @staticmethod
    def write_npz(file: str | PathLike[str] | BinaryIO, asset: OVoxelAsset, **kwargs: Any) -> None:
        write_ovoxel_npz(file, asset, **kwargs)

    @staticmethod
    def _runtime_member(runtime: ModuleType, path: str) -> Any:
        value: Any = runtime
        for name in path.split("."):
            value = getattr(value, name, None)
            if value is None:
                break
        return value

    def _supports_native_members(self, required_members: Sequence[str]) -> bool:
        if not self.accept_nvdiffrast_research_license:
            return False
        try:
            self.registry.select(
                BackendCapability.NATIVE_REPRESENTATION,
                name="o_voxel",
                device=self.device,
                dtype=self.dtype,
                differentiable=False,
            )
            if self._runtime is None:
                self._runtime = importlib.import_module("o_voxel")
        except Exception:
            return False
        return all(callable(self._runtime_member(self._runtime, member)) for member in required_members)

    def _load_runtime(
        self,
        operation: str,
        *,
        capability: OVoxelCapability,
        required_members: Sequence[str],
    ) -> ModuleType:
        if not self.accept_nvdiffrast_research_license:
            raise OVoxelRuntimeUnavailableError(
                f"{operation} requires accept_nvdiffrast_research_license=True because pinned O-Voxel "
                "eagerly imports its nvdiffrast-dependent postprocess module; pure schema and .npz support "
                "do not import it"
            )
        try:
            self.registry.select(
                BackendCapability.NATIVE_REPRESENTATION,
                name="o_voxel",
                device=self.device,
                dtype=self.dtype,
                differentiable=False,
            )
        except Exception as error:
            raise OVoxelRuntimeUnavailableError(
                f"{operation} requires the separately compiled o_voxel runtime; pure support is limited to "
                "schema packing and .npz"
            ) from error
        if self._runtime is None:
            try:
                self._runtime = importlib.import_module("o_voxel")
            except (ImportError, OSError, RuntimeError) as error:
                raise OVoxelRuntimeUnavailableError(
                    f"{operation} requires the separately compiled o_voxel runtime"
                ) from error
        missing = tuple(
            member for member in required_members if not callable(self._runtime_member(self._runtime, member))
        )
        if missing:
            raise OVoxelRuntimeUnavailableError(
                f"{operation} requires O-Voxel capability {capability.value!r}, but the selected runtime "
                f"is missing: {', '.join(missing)}"
            )
        return self._runtime

    @staticmethod
    def _native_file(file: str | PathLike[str] | BinaryIO) -> str | BinaryIO:
        return fspath(file) if isinstance(file, (str, PathLike)) else file

    def read_vxz(
        self,
        file: str | PathLike[str] | BinaryIO,
        *,
        resolution: int | Sequence[int] | None = None,
        aabb: torch.Tensor | Sequence[Sequence[float]] = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        num_threads: int = -1,
    ) -> OVoxelAsset:
        runtime = self._load_runtime(
            ".vxz reading",
            capability=OVoxelCapability.NATIVE_CODEC,
            required_members=("io.read_vxz",),
        )
        if resolution is None:
            raise ValueError(
                ".vxz does not encode the source grid resolution; pass resolution explicitly to preserve its transform"
            )
        if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads == 0:
            raise ValueError("num_threads must be -1 or a non-zero integer")
        coordinates, attributes = runtime.io.read_vxz(
            self._native_file(file),
            num_threads=num_threads,
        )
        return self.from_official(
            coordinates,
            attributes,
            resolution=resolution,
            aabb=aabb,
            packed=True,
        )

    def write_vxz(
        self,
        file: str | PathLike[str] | BinaryIO,
        asset: OVoxelAsset,
        **kwargs: Any,
    ) -> None:
        _serialization_resolution(asset, format_name="native O-Voxel VXZ")
        runtime = self._load_runtime(
            ".vxz writing",
            capability=OVoxelCapability.NATIVE_CODEC,
            required_members=("io.write_vxz",),
        )
        if asset.split_weights is not None:
            raise ValueError(
                "The pinned VXZ v0 runtime accepts only uint8 attributes and has no verified lossless "
                "split_weights encoding; use write_npz() instead"
            )
        # The official runtime performs chunk-local ordering internally. Global
        # Morton sorting here would reject valid uint16-resolution assets.
        coordinates, attributes = self.to_official(asset, packed=True, morton_order=False)
        runtime.io.write_vxz(
            self._native_file(file),
            coordinates.to(dtype=torch.int32),
            attributes,
            **kwargs,
        )

    def to_mesh(self, asset: OVoxelAsset, *, train: bool = False) -> MeshAsset:
        runtime = self._load_runtime(
            "O-Voxel mesh conversion",
            capability=OVoxelCapability.NATIVE_CONVERSION,
            required_members=("convert.flexible_dual_grid_to_mesh",),
        )
        coordinates, attributes = self.to_official(asset, packed=False, morton_order=False)
        resolution = asset.metadata.get("resolution")
        aabb = asset.metadata.get("aabb")
        if resolution is None or aabb is None:
            raise ValueError("native mesh conversion requires explicit resolution and aabb metadata")
        intersected = _unpack_intersection_flags(asset.intersection_data, asset.active_coordinates.shape[0])
        vertices, faces = runtime.convert.flexible_dual_grid_to_mesh(
            coordinates.to(device=self.device, dtype=torch.int32),
            attributes["dual_vertices"].to(self.device),
            intersected.to(self.device),
            None if asset.split_weights is None else asset.split_weights.to(self.device),
            grid_size=resolution,
            aabb=aabb,
            train=train,
        )
        return MeshAsset(
            vertices=vertices,
            faces=faces.to(dtype=torch.int64),
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
            metadata={
                "source": "o_voxel",
                "reference_revision": OVOXEL_REFERENCE_REVISION,
                "resolution": list(_resolution_tuple(resolution)),
                "aabb": aabb,
            },
        )

    def render_voxels(
        self,
        asset: OVoxelAsset,
        *,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: int,
        attribute: str = "base_color",
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if not isinstance(extrinsics, torch.Tensor) or extrinsics.shape != (4, 4):
            raise ValueError("extrinsics must have shape (4, 4)")
        if not isinstance(intrinsics, torch.Tensor) or intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        runtime = self._load_runtime(
            "O-Voxel rendering",
            capability=OVoxelCapability.NATIVE_RENDERING,
            required_members=("rasterize.VoxelRenderer",),
        )
        values = {
            "base_color": asset.base_color,
            "metallic": asset.metallic,
            "roughness": asset.roughness,
            "alpha": asset.opacity,
            "normal": asset.normals,
            "emissive": asset.emissive,
        }.get(attribute)
        if values is None:
            raise ValueError(f"attribute {attribute!r} is unavailable")
        resolution = asset.metadata.get("resolution")
        try:
            resolution_values = _resolution_tuple(resolution)
        except (TypeError, ValueError) as error:
            raise ValueError("native voxel rendering requires explicit cubic resolution metadata") from error
        if len(set(resolution_values)) != 1:
            raise ValueError("native voxel rendering currently requires explicit cubic resolution metadata")
        voxel_sizes = asset.grid_transform.diagonal()[:3]
        if not torch.allclose(voxel_sizes, voxel_sizes[:1].expand_as(voxel_sizes)):
            raise ValueError("native voxel rendering requires equal world-space voxel sizes on every axis")
        renderer = runtime.rasterize.VoxelRenderer({"resolution": image_size})
        position = torch.cat(
            [
                asset.active_coordinates.to(dtype=asset.base_color.dtype),
                torch.ones(asset.active_coordinates.shape[0], 1, device=asset.device, dtype=asset.base_color.dtype),
            ],
            dim=1,
        )
        position = (asset.grid_transform @ position.T).T[:, :3]
        result = renderer.render(
            position=position.to(self.device),
            attrs=values.to(self.device),
            voxel_size=float(voxel_sizes[0].item()),
            extrinsics=extrinsics.to(self.device),
            intrinsics=intrinsics.to(self.device),
        )
        return {"attr": result.attr, "depth": result.depth, "alpha": result.alpha}


__all__ = [
    "OVOXEL_METADATA_PREFIX",
    "OVOXEL_REFERENCE_REVISION",
    "OVoxelBackend",
    "OVoxelCapability",
    "OVoxelRuntimeUnavailableError",
    "morton_decode_3d",
    "morton_encode_3d",
    "official_tensors_from_ovoxel_asset",
    "ovoxel_grid_transform",
    "ovoxel_asset_from_official",
    "read_ovoxel_npz",
    "write_ovoxel_npz",
]
