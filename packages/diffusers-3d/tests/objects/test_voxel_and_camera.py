import pytest
import torch

from diffusers_3d import (
    CameraRig,
    Object3DKind,
    Object3DValidationError,
    OVoxelAsset,
    SparseVoxelAsset,
    TensorDTypeError,
    TensorShapeError,
)


def test_sparse_voxel_channels_and_to(sparse_voxel):
    assert sparse_voxel.kind is Object3DKind.SPARSE_VOXEL
    assert sparse_voxel.features.shape[0] == sparse_voxel.coordinates.shape[0]
    moved = sparse_voxel.to(dtype=torch.float64)
    assert moved.features.dtype is torch.float64
    assert moved.extras["occupancy"].dtype is torch.float64
    assert moved.coordinates.dtype is torch.int32
    assert moved.semantic_labels.dtype is torch.int64


def test_sparse_voxel_accepts_grid_transform():
    voxel = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 0, 0]], dtype=torch.int64),
        features=torch.ones(1, 2),
        grid_transform=torch.eye(4),
    )
    assert voxel.voxel_size is None
    assert voxel.grid_transform.shape == (4, 4)


def test_sparse_voxel_requires_one_grid_metric():
    arguments = {
        "coordinates": torch.tensor([[0, 0, 0]], dtype=torch.int64),
        "features": torch.ones(1, 2),
    }
    with pytest.raises(Object3DValidationError, match="exactly one"):
        SparseVoxelAsset(**arguments)
    with pytest.raises(Object3DValidationError, match="exactly one"):
        SparseVoxelAsset(**arguments, voxel_size=1.0, grid_transform=torch.eye(4))


def test_sparse_voxel_validates_coordinate_and_channel_counts():
    with pytest.raises(TensorDTypeError):
        SparseVoxelAsset(torch.zeros(1, 3), torch.ones(1, 2), voxel_size=1.0)
    with pytest.raises(TensorShapeError, match="features"):
        SparseVoxelAsset(
            torch.tensor([[0, 0, 0], [1, 0, 0]]),
            torch.ones(1, 2),
            voxel_size=1.0,
        )
    with pytest.raises(TensorShapeError, match="semantic_labels"):
        SparseVoxelAsset(
            torch.tensor([[0, 0, 0], [1, 0, 0]]),
            torch.ones(2, 2),
            voxel_size=1.0,
            semantic_labels=torch.ones(1, dtype=torch.int64),
        )


def test_sparse_voxel_expensive_validation_rejects_duplicates():
    voxel = SparseVoxelAsset(
        torch.tensor([[0, 0, 0], [0, 0, 0]]),
        torch.ones(2, 1),
        voxel_size=torch.tensor([0.1, 0.1, 0.2]),
    )
    voxel.validate()
    with pytest.raises(Object3DValidationError, match="duplicate"):
        voxel.validate(expensive=True)


def test_o_voxel_preserves_dual_grid_and_pbr_channels(o_voxel):
    assert o_voxel.kind is Object3DKind.O_VOXEL
    assert o_voxel.dual_grid_vertex_offsets.shape == (4, 3)
    assert o_voxel.dual_grid_topology.dtype is torch.int64
    assert o_voxel.base_color.shape == (2, 3)
    o_voxel.validate(expensive=True)

    moved = o_voxel.to(dtype=torch.float64)
    assert moved.base_color.dtype is torch.float64
    assert moved.dual_grid_vertex_offsets.dtype is torch.float64
    assert moved.active_coordinates.dtype is torch.int32
    assert moved.dual_grid_topology.dtype is torch.int64


def test_o_voxel_accepts_intersection_representation():
    voxel = OVoxelAsset(
        active_coordinates=torch.tensor([[0, 0, 0]], dtype=torch.int64),
        base_color=torch.ones(1, 4),
        metallic=torch.zeros(1),
        roughness=torch.ones(1, 1),
        intersection_data=torch.tensor([[[0.0, 0.5], [1.0, 0.5]]]),
    )
    assert voxel.intersection_data.shape == (1, 2, 2)
    assert voxel.dual_grid_topology is None


@pytest.mark.parametrize("channel", ["metallic", "roughness"])
def test_o_voxel_rejects_missing_mandatory_material_channels(channel):
    arguments = {
        "active_coordinates": torch.tensor([[0, 0, 0]], dtype=torch.int64),
        "base_color": torch.ones(1, 3),
        "metallic": torch.zeros(1),
        "roughness": torch.ones(1),
        "intersection_data": torch.ones(1, 2),
    }
    arguments[channel] = None

    with pytest.raises(Object3DValidationError, match=f"{channel} must be a tensor"):
        OVoxelAsset(**arguments)


def test_o_voxel_requires_complete_geometry_schema():
    arguments = {
        "active_coordinates": torch.tensor([[0, 0, 0]], dtype=torch.int64),
        "base_color": torch.ones(1, 3),
        "metallic": torch.zeros(1),
        "roughness": torch.ones(1),
    }
    with pytest.raises(Object3DValidationError, match="must be provided"):
        OVoxelAsset(**arguments)
    with pytest.raises(Object3DValidationError, match="provided together"):
        OVoxelAsset(**arguments, dual_grid_vertex_offsets=torch.zeros(1, 3))


def test_o_voxel_validates_topology_and_pbr_semantics(o_voxel):
    o_voxel.dual_grid_topology[0, 2] = 4
    with pytest.raises(Object3DValidationError, match="vertex index"):
        o_voxel.validate(expensive=True)

    with pytest.raises(Object3DValidationError, match=r"\[0, 1\]"):
        OVoxelAsset(
            active_coordinates=torch.tensor([[0, 0, 0]]),
            base_color=torch.ones(1, 3),
            metallic=torch.tensor([1.1]),
            roughness=torch.ones(1),
            intersection_data=torch.ones(1, 2),
        )
    with pytest.raises(TensorShapeError, match="base_color"):
        OVoxelAsset(
            active_coordinates=torch.tensor([[0, 0, 0]]),
            base_color=torch.ones(1, 2),
            metallic=torch.zeros(1),
            roughness=torch.ones(1),
            intersection_data=torch.ones(1, 2),
        )


def make_camera_rig() -> CameraRig:
    world_to_camera = torch.eye(4).repeat(2, 1, 1)
    intrinsics = torch.tensor(
        [
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            [[120.0, 0.0, 32.0], [0.0, 110.0, 24.0], [0.0, 0.0, 1.0]],
        ]
    )
    return CameraRig(world_to_camera, intrinsics, torch.tensor([[48, 64], [48, 64]]))


def test_camera_rig_validation_and_to():
    cameras = make_camera_rig()
    assert cameras.world_to_camera.shape == (2, 4, 4)
    moved = cameras.to(dtype=torch.float64)
    assert moved.intrinsics.dtype is torch.float64
    assert moved.image_sizes.dtype is torch.int64


def test_camera_rig_rejects_invalid_counts_intrinsics_and_sizes():
    cameras = make_camera_rig()
    with pytest.raises(TensorShapeError, match="same camera count"):
        CameraRig(cameras.world_to_camera, cameras.intrinsics[:1], cameras.image_sizes)

    intrinsics = cameras.intrinsics.clone()
    intrinsics[0, 0, 0] = 0
    with pytest.raises(Object3DValidationError, match="focal lengths"):
        CameraRig(cameras.world_to_camera, intrinsics, cameras.image_sizes)

    sizes = cameras.image_sizes.clone()
    sizes[0, 0] = 0
    with pytest.raises(Object3DValidationError, match="positive"):
        CameraRig(cameras.world_to_camera, cameras.intrinsics, sizes)

    with pytest.raises(TensorDTypeError):
        CameraRig(cameras.world_to_camera, cameras.intrinsics, cameras.image_sizes.float())
