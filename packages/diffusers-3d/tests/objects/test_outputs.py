import pytest
import torch

from diffusers_3d import (
    CoordinateSystem,
    Latent3DOutput,
    Object3D,
    Object3DKind,
    Object3DPipelineOutput,
    Object3DValidationError,
    TensorShapeError,
)


class StructuralMesh:
    def __init__(self, vertices: torch.Tensor):
        self.vertices = vertices
        self.kind = Object3DKind.MESH
        self.coordinate_system = CoordinateSystem.RIGHT_HANDED_Y_UP
        self.object_to_world = torch.eye(4, device=vertices.device, dtype=vertices.dtype)

    @property
    def device(self) -> torch.device:
        return self.vertices.device

    def tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (("vertices", self.vertices), ("object_to_world", self.object_to_world))

    def validate(self, expensive: bool = False) -> None:
        del expensive
        if self.vertices.shape[-1] != 3:
            raise ValueError("vertices must end in three coordinates")

    def to(self, device=None, dtype=None, non_blocking=False):
        return StructuralMesh(self.vertices.to(device=device, dtype=dtype, non_blocking=non_blocking))


def test_latent_output_validation_access_and_to():
    source = torch.ones(1, 4, requires_grad=True)
    output = Latent3DOutput(source, metadata={"stage": "geometry"})
    assert output["latents"] is source
    assert output[0] is source

    moved = output.to(dtype=torch.float64)
    assert isinstance(moved, Latent3DOutput)
    assert moved.latents.dtype is torch.float64
    moved.latents.sum().backward()
    assert source.grad is not None

    with pytest.raises(TensorShapeError):
        Latent3DOutput(torch.ones(4))


def test_pipeline_output_has_stable_primary_objects_semantics(mesh):
    output = Object3DPipelineOutput(objects=[mesh])
    assert isinstance(output.objects, tuple)
    assert output[0] is output.objects
    assert output["objects"] is output.objects
    assert output.to_tuple() == (output.objects,)
    assert tuple(output.keys()) == ("objects",)

    latents = Latent3DOutput(torch.ones(1, 4))
    output_with_optional = Object3DPipelineOutput(objects=(mesh,), latents=latents, previews=["preview"])
    assert tuple(output_with_optional.keys()) == ("objects", "latents", "previews")
    assert output_with_optional[0] is output_with_optional.objects
    assert output_with_optional.previews == ("preview",)


def test_pipeline_output_requires_nonempty_structural_objects():
    with pytest.raises(Object3DValidationError, match="at least one"):
        Object3DPipelineOutput(objects=())
    with pytest.raises(Object3DValidationError, match="protocol"):
        Object3DPipelineOutput(objects=(torch.zeros(1),))

    structural = StructuralMesh(torch.zeros(3, 3))
    assert isinstance(structural, Object3D)
    output = Object3DPipelineOutput(objects=(structural,))
    assert output.objects == (structural,)


def test_pipeline_output_rejects_invalid_structural_attributes():
    structural = StructuralMesh(torch.zeros(3, 3))
    structural.kind = "mesh"
    with pytest.raises(Object3DValidationError, match="Object3DKind"):
        Object3DPipelineOutput(objects=(structural,))

    with pytest.raises(Object3DValidationError, match="previews"):
        Object3DPipelineOutput(objects=(StructuralMesh(torch.zeros(3, 3)),), previews="bad")


def test_pipeline_output_accepts_raw_latents_and_validates_them(mesh):
    output = Object3DPipelineOutput(objects=(mesh,), latents=torch.ones(1, 4))
    assert output.latents.shape == (1, 4)
    with pytest.raises(TensorShapeError):
        Object3DPipelineOutput(objects=(mesh,), latents=torch.ones(4))
    with pytest.raises(Object3DValidationError, match="latents"):
        Object3DPipelineOutput(objects=(mesh,), latents={"bad": "type"})


def test_pipeline_to_moves_nested_outputs_and_preserves_indices(mesh):
    output = Object3DPipelineOutput(
        objects=(mesh,),
        latents=Latent3DOutput(torch.ones(1, 4)),
        previews=(torch.ones(1),),
    )
    moved = output.to(dtype=torch.float64)
    assert isinstance(moved, Object3DPipelineOutput)
    assert moved.objects[0].vertices.dtype is torch.float64
    assert moved.objects[0].faces.dtype is mesh.faces.dtype
    assert moved.latents.latents.dtype is torch.float64
    assert moved.previews[0].dtype is torch.float64


def test_pipeline_output_pytree_flattening(mesh):
    output = Object3DPipelineOutput(objects=(mesh,), latents=torch.ones(1, 4))
    leaves, tree_spec = torch.utils._pytree.tree_flatten(output)
    assert not torch.utils._pytree.tree_is_leaf(output)
    assert any(leaf is mesh.vertices for leaf in leaves)
    assert any(leaf is output.latents for leaf in leaves)
    rebuilt = torch.utils._pytree.tree_unflatten(leaves, tree_spec)
    assert isinstance(rebuilt, Object3DPipelineOutput)
    assert rebuilt.objects[0].kind is Object3DKind.MESH
