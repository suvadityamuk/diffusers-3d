import pytest
import torch
from diffusers.utils import BaseOutput

from diffusers_3d import (
    CoordinateSystem,
    MeshAsset,
    MetadataValidationError,
    Object3D,
    Object3DKind,
    Object3DValidationError,
    PBRMaterial,
    TensorDeviceError,
    TensorDTypeError,
    TensorShapeError,
)
from diffusers_3d.objects import MeshAsset as ObjectsMeshAsset


def test_public_exports_and_structural_protocol(mesh):
    assert ObjectsMeshAsset is MeshAsset
    assert isinstance(mesh, BaseOutput)
    assert isinstance(mesh, Object3D)
    assert mesh.kind is Object3DKind.MESH
    assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Y_UP
    assert mesh.object_to_world is mesh.transform
    assert mesh.device == torch.device("cpu")


def test_mesh_base_output_dict_and_tuple_access(mesh):
    assert mesh["vertices"] is mesh.vertices
    assert mesh[0] is mesh.vertices
    assert mesh.to_tuple()[0] is mesh.vertices
    assert dict(mesh)["faces"] is mesh.faces
    assert tuple(mesh.materials) == mesh.materials
    assert "normals" in mesh


def test_mesh_normalizes_coordinate_metadata_and_sequences(material):
    mesh = MeshAsset(
        vertices=torch.zeros(3, 3),
        faces=torch.tensor([[0, 1, 2]]),
        coordinate_system="right_handed_z_up",
        materials=[material],
        metadata={"nested": {"values": [1, True, None]}},
    )
    assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    assert isinstance(mesh.materials, tuple)
    assert mesh.metadata == {"nested": {"values": [1, True, None]}}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("vertices", torch.zeros(3, 2), TensorShapeError),
        ("vertices", torch.ones(3, 3, dtype=torch.int64), TensorDTypeError),
        ("faces", torch.tensor([0, 1, 2]), TensorShapeError),
        ("faces", torch.zeros(1, 3), TensorDTypeError),
        ("normals", torch.zeros(2, 3), TensorShapeError),
        ("colors", torch.zeros(3, 2), TensorShapeError),
        ("uvs", torch.zeros(2, 2), TensorShapeError),
        ("face_material_ids", torch.zeros(2, dtype=torch.int64), TensorShapeError),
    ],
)
def test_mesh_rejects_invalid_shapes_and_dtypes(field, value, error, material):
    arguments = {
        "vertices": torch.zeros(3, 3),
        "faces": torch.tensor([[0, 1, 2]]),
        "normals": torch.zeros(3, 3),
        "colors": torch.zeros(3, 3),
        "uvs": torch.zeros(3, 2),
        "face_material_ids": torch.zeros(1, dtype=torch.int64),
        "materials": (material,),
    }
    arguments[field] = value
    with pytest.raises(error):
        MeshAsset(**arguments)


def test_mesh_expensive_validation_checks_indices(mesh):
    mesh.faces[0, 2] = 10
    mesh.validate()
    with pytest.raises(Object3DValidationError, match="vertex index"):
        mesh.validate(expensive=True)

    mesh.faces[0, 2] = 2
    mesh.face_material_ids[0] = 1
    with pytest.raises(Object3DValidationError, match="material index"):
        mesh.validate(expensive=True)


def test_mesh_rejects_invalid_transform_and_mixed_devices():
    singular = torch.eye(4)
    singular[2, 2] = 0
    with pytest.raises(Object3DValidationError, match="invertible"):
        MeshAsset(torch.zeros(3, 3), torch.tensor([[0, 1, 2]]), transform=singular)

    with pytest.raises(TensorDeviceError):
        MeshAsset(
            torch.zeros(3, 3),
            torch.tensor([[0, 1, 2]]),
            extras={"indices": torch.empty(3, 1, dtype=torch.int64, device="meta")},
        )


def test_metadata_must_be_json_safe():
    with pytest.raises(MetadataValidationError):
        PBRMaterial(torch.ones(3), metadata={"bad": (1, 2)})
    with pytest.raises(MetadataValidationError):
        PBRMaterial(torch.ones(3), metadata={"bad": float("nan")})
    with pytest.raises(MetadataValidationError):
        PBRMaterial(torch.ones(3), metadata={1: "not a string key"})
    material = PBRMaterial(torch.ones(3))
    material.metadata["bad"] = torch.tensor(1)
    with pytest.raises(MetadataValidationError):
        material.validate()


def test_pbr_channel_semantics_and_ranges():
    material = PBRMaterial(
        base_color=torch.ones(4, 5, 3),
        metallic=torch.zeros(4, 5, 1),
        roughness=torch.tensor(0.5),
        normal=torch.tensor([0.0, 0.0, 1.0]),
        emissive=torch.zeros(4, 5, 3),
        opacity=torch.ones(1),
    )
    assert material.base_color.shape == (4, 5, 3)

    with pytest.raises(TensorShapeError, match="spatial dimensions"):
        PBRMaterial(torch.ones(4, 5, 3), metallic=torch.zeros(4, 1))
    with pytest.raises(Object3DValidationError, match=r"\[0, 1\]"):
        PBRMaterial(torch.tensor([1.1, 0.0, 0.0]))


def test_to_preserves_indices_nested_channels_and_gradients():
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    base_color = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
    weights = torch.ones(3, 1, requires_grad=True)
    mesh = MeshAsset(
        vertices=vertices,
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        face_material_ids=torch.tensor([0], dtype=torch.int32),
        materials=(PBRMaterial(base_color),),
        extras={"weights": weights, "labels": torch.ones(3, dtype=torch.int16)},
    )

    moved = mesh.to(dtype=torch.float64, non_blocking=True)
    assert type(moved) is MeshAsset
    assert moved is not mesh
    assert moved.vertices.dtype is torch.float64
    assert moved.faces.dtype is torch.int64
    assert moved.face_material_ids.dtype is torch.int32
    assert moved.extras["labels"].dtype is torch.int16
    assert moved.materials[0].base_color.dtype is torch.float64
    assert mesh.vertices.dtype is torch.float32

    (moved.vertices.sum() + moved.materials[0].base_color.sum() + moved.extras["weights"].sum()).backward()
    assert vertices.grad is not None
    assert base_color.grad is not None
    assert weights.grad is not None


def test_base_output_is_a_pytree_node(mesh):
    leaves, tree_spec = torch.utils._pytree.tree_flatten(mesh)
    assert not torch.utils._pytree.tree_is_leaf(mesh)
    assert any(leaf is mesh.vertices for leaf in leaves)
    rebuilt = torch.utils._pytree.tree_unflatten(leaves, tree_spec)
    assert isinstance(rebuilt, MeshAsset)
    assert torch.equal(rebuilt.faces, mesh.faces)
