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


def test_mesh_to_coordinate_system_matches_trellis_z_up_to_y_up_and_flips_winding_on_reflection():
    from diffusers_3d import changes_handedness, coordinate_change_matrix

    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    normals = torch.nn.functional.normalize(vertices + 0.5, dim=-1)
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    mesh = MeshAsset(
        vertices=vertices,
        faces=faces,
        normals=normals,
        transform=transform,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        metadata={"family": "trellis"},
    )

    y_up = mesh.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP)
    # TRELLIS ``to_glb``: vertices @ [[1, 0, 0], [0, 0, -1], [0, 1, 0]] -> (x, z, -y).
    torch.testing.assert_close(y_up.vertices, vertices @ torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]))
    torch.testing.assert_close(y_up.normals, normals @ torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]))
    torch.testing.assert_close(y_up.transform[:3, 3], torch.tensor([1.0, 3.0, -2.0]))
    assert torch.equal(y_up.faces, faces)
    assert y_up.coordinate_system is CoordinateSystem.RIGHT_HANDED_Y_UP
    assert y_up.metadata == mesh.metadata
    assert mesh.to_coordinate_system("right_handed_z_up") is mesh
    round_trip = y_up.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Z_UP)
    torch.testing.assert_close(round_trip.vertices, vertices)
    torch.testing.assert_close(round_trip.transform, transform)

    # World-space points agree whichever frame the object is expressed in.
    matrix = coordinate_change_matrix(CoordinateSystem.RIGHT_HANDED_Z_UP, CoordinateSystem.RIGHT_HANDED_Y_UP)
    homogeneous = torch.cat([vertices, torch.ones(4, 1)], dim=1)
    world_z_up = (homogeneous @ transform.T)[:, :3]
    world_y_up = (torch.cat([y_up.vertices, torch.ones(4, 1)], dim=1) @ y_up.transform.T)[:, :3]
    torch.testing.assert_close(world_y_up, world_z_up @ matrix.T)

    left = y_up.to_coordinate_system(CoordinateSystem.LEFT_HANDED_Y_UP)
    assert changes_handedness(CoordinateSystem.RIGHT_HANDED_Y_UP, CoordinateSystem.LEFT_HANDED_Y_UP)
    torch.testing.assert_close(left.vertices, y_up.vertices * torch.tensor([1.0, 1.0, -1.0]))
    assert torch.equal(left.faces, faces[:, [0, 2, 1]])
    for system in CoordinateSystem:
        matrix = coordinate_change_matrix(CoordinateSystem.RIGHT_HANDED_Y_UP, system, dtype=torch.float64)
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(3, dtype=torch.float64))
        assert abs(float(torch.det(matrix))) == 1.0
