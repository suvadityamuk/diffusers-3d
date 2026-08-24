import pytest
import torch

from diffusers_3d import (
    GaussianSplatAsset,
    MeshAsset,
    OVoxelAsset,
    PBRMaterial,
    SparseVoxelAsset,
)


@pytest.fixture
def material() -> PBRMaterial:
    return PBRMaterial(
        base_color=torch.tensor([0.2, 0.4, 0.6]),
        metallic=torch.tensor(0.1),
        roughness=torch.tensor(0.7),
    )


@pytest.fixture
def mesh(material: PBRMaterial) -> MeshAsset:
    return MeshAsset(
        vertices=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        normals=torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone(),
        colors=torch.ones(3, 4),
        uvs=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        face_material_ids=torch.tensor([0], dtype=torch.int32),
        materials=(material,),
        extras={"vertex_confidence": torch.ones(3, 1)},
        metadata={"name": "triangle", "tags": ["test", 3]},
    )


@pytest.fixture
def gaussian() -> GaussianSplatAsset:
    return GaussianSplatAsset(
        means=torch.zeros(2, 3),
        log_scales=torch.full((2, 3), -2.0),
        quaternions_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1).clone(),
        opacity_logits=torch.zeros(2, 1),
        sh_coefficients=torch.zeros(2, 4, 3),
        active_sh_degree=1,
        extras={"labels": torch.tensor([1, 2], dtype=torch.int64)},
    )


@pytest.fixture
def sparse_voxel() -> SparseVoxelAsset:
    return SparseVoxelAsset(
        coordinates=torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
        features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        voxel_size=0.05,
        semantic_labels=torch.tensor([2, 4], dtype=torch.int64),
        extras={"occupancy": torch.ones(2, 1)},
    )


@pytest.fixture
def o_voxel() -> OVoxelAsset:
    return OVoxelAsset(
        active_coordinates=torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
        base_color=torch.tensor([[0.2, 0.4, 0.6], [0.6, 0.4, 0.2]]),
        metallic=torch.zeros(2, 1),
        roughness=torch.full((2,), 0.5),
        dual_grid_vertex_offsets=torch.tensor(
            [
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
                [0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.0],
            ]
        ),
        dual_grid_topology=torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64),
        opacity=torch.ones(2, 1),
        normals=torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1).clone(),
        extras={"cell_features": torch.ones(2, 2)},
    )
