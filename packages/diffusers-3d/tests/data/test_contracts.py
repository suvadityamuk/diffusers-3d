from dataclasses import replace

import pytest
import torch

import diffusers_3d
from diffusers_3d import (
    CameraRig,
    ImageCondition,
    MeshAsset,
    MultiViewCondition,
    Object3DDataset,
    Object3DExample,
    Object3DValidationError,
    PackedMeshBatch,
    PBRMaterial,
    TensorDTypeError,
    TensorShapeError,
    TextCondition,
)


def make_camera(count: int, height: int = 8, width: int = 6) -> CameraRig:
    return CameraRig(
        world_to_camera=torch.eye(4).expand(count, -1, -1).clone(),
        intrinsics=torch.tensor([[4.0, 0.0, 3.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]).expand(count, -1, -1).clone(),
        image_sizes=torch.tensor([[height, width]], dtype=torch.int64).expand(count, -1).clone(),
    )


def make_mesh(offset: float = 0.0, *, with_channels: bool = True) -> MeshAsset:
    vertices = torch.tensor(
        [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]],
        requires_grad=True,
    )
    arguments = {}
    if with_channels:
        arguments = {
            "normals": torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone(),
            "colors": torch.ones(3, 4),
            "uvs": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "face_material_ids": torch.tensor([0], dtype=torch.int32),
            "materials": (PBRMaterial(torch.tensor([0.2, 0.4, 0.6], requires_grad=True)),),
            "extras": {
                "weights": torch.ones(3, 1, requires_grad=True),
                "labels": torch.tensor([1, 2, 3], dtype=torch.int16),
            },
            "metadata": {"name": f"triangle-{offset}"},
        }
    return MeshAsset(
        vertices=vertices,
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        **arguments,
    )


def test_conditions_and_runtime_dataset_contract():
    text = TextCondition("a chair", negative_text="")
    image = ImageCondition(torch.zeros(3, 8, 6), camera=make_camera(1), mask=torch.ones(1, 8, 6))
    views = MultiViewCondition(
        torch.zeros(2, 3, 8, 6),
        cameras=make_camera(2),
        masks=torch.ones(2, 1, 8, 6),
    )
    mesh = make_mesh()
    examples = [
        Object3DExample(mesh, text, "text"),
        Object3DExample(mesh, image, "image"),
        Object3DExample(mesh, views, "views"),
    ]

    class Dataset:
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            return examples[index]

    assert isinstance(Dataset(), Object3DDataset)
    assert diffusers_3d.TextCondition is TextCondition
    assert diffusers_3d.Object3DExample is Object3DExample


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: TextCondition(" "),
        lambda: ImageCondition(torch.zeros(2, 8, 6)),
        lambda: ImageCondition(torch.zeros(3, 8, 6), camera=make_camera(2)),
        lambda: ImageCondition(torch.zeros(3, 8, 6), mask=torch.full((1, 8, 6), 2.0)),
        lambda: MultiViewCondition(torch.zeros(2, 3, 8, 6), cameras=make_camera(1)),
        lambda: MultiViewCondition(torch.zeros(2, 3, 8, 6), cameras=make_camera(2, height=7)),
    ],
)
def test_conditions_reject_invalid_values(constructor):
    with pytest.raises(Object3DValidationError):
        constructor()


def test_condition_to_preserves_integer_dtypes_and_gradients():
    images = torch.ones(2, 3, 8, 6, requires_grad=True)
    condition = MultiViewCondition(images, make_camera(2))
    moved = condition.to(dtype=torch.float64)

    assert moved.images.dtype is torch.float64
    assert moved.cameras.image_sizes.dtype is torch.int64
    moved.images.sum().backward()
    assert images.grad is not None


def test_packed_mesh_batch_round_trip_is_lossless_and_differentiable():
    meshes = (make_mesh(0.0), make_mesh(2.0, with_channels=False))
    packed = PackedMeshBatch.pack(meshes)
    unpacked = packed.unpack()

    assert packed.batch_size == 2
    assert torch.equal(packed.vertex_offsets, torch.tensor([0, 3, 6]))
    assert torch.equal(packed.face_offsets, torch.tensor([0, 1, 2]))
    for original, restored in zip(meshes, unpacked):
        for name in ("vertices", "faces", "transform", "normals", "colors", "uvs", "face_material_ids"):
            original_value = getattr(original, name)
            restored_value = getattr(restored, name)
            if original_value is None:
                assert restored_value is None
            else:
                torch.testing.assert_close(restored_value, original_value)
        assert restored.coordinate_system is original.coordinate_system
        assert restored.metadata == original.metadata
        assert restored.extras.keys() == original.extras.keys()
        assert len(restored.materials) == len(original.materials)

    moved = packed.to(dtype=torch.float64)
    assert moved.vertices.dtype is torch.float64
    assert moved.faces.dtype is torch.int64
    assert moved.vertex_offsets.dtype is torch.int64
    assert moved.extras[0]["labels"].dtype is torch.int16
    loss = moved.unpack()[0].vertices.sum() + moved.extras[0]["weights"].sum()
    loss.backward()
    assert meshes[0].vertices.grad is not None
    assert meshes[0].extras["weights"].grad is not None


def test_packed_mesh_batch_rejects_invalid_inputs():
    with pytest.raises(Object3DValidationError, match="at least one"):
        PackedMeshBatch.pack(())
    with pytest.raises(TensorDTypeError):
        PackedMeshBatch.pack((make_mesh(), make_mesh().to(dtype=torch.float64)))

    packed = PackedMeshBatch.pack((make_mesh(),))
    with pytest.raises(TensorShapeError, match="start at zero"):
        replace(packed, vertex_offsets=torch.tensor([1, 3], dtype=torch.int64))
    with pytest.raises(TensorDTypeError):
        replace(packed, face_offsets=packed.face_offsets.to(torch.int32))
