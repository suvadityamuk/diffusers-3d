from __future__ import annotations

import pytest
import torch
from diffusers.configuration_utils import register_to_config

from diffusers_3d import (
    ContributionStatus,
    MeshAsset,
    Object3DKind,
    Object3DModel,
    Object3DPipeline,
    ReviewStatus,
)


class TinyObject3DModel(Object3DModel):
    family_id = "tiny-family"
    component_role = "denoiser"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


class UnreviewedObject3DModel(Object3DModel):
    family_id = "unreviewed-family"
    component_role = "denoiser"
    supported_object_kinds = (Object3DKind.MESH,)

    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


class OtherTinyObject3DModel(Object3DModel):
    family_id = "tiny-family"
    component_role = "denoiser"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


class TinyObject3DModelSubclass(TinyObject3DModel):
    pass


class TinyObject3DPipeline(Object3DPipeline):
    family_id = "tiny-family"
    task_ids = ("text-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("triangle-mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("trimesh",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    def __init__(self) -> None:
        super().__init__()
        self.register_to_config()


class OtherTinyObject3DPipeline(Object3DPipeline):
    family_id = "tiny-family"
    task_ids = ("text-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("triangle-mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("trimesh",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    def __init__(self) -> None:
        super().__init__()
        self.register_to_config()


class TinyObject3DPipelineSubclass(TinyObject3DPipeline):
    pass


class DispatchObject3DPipeline(Object3DPipeline):
    family_id = "dispatch-family"
    task_ids = ("image-to-3d", "text-to-3d")
    output_object_types = (MeshAsset,)
    output_representations = ("triangle-mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    load_calls: list[tuple[object, dict[str, object]]] = []

    def __init__(self) -> None:
        super().__init__()
        self.register_to_config()
        self.loaded_from: object | None = None
        self.loaded_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cls.load_calls.append((pretrained_model_name_or_path, kwargs))
        pipeline = cls()
        pipeline.loaded_from = pretrained_model_name_or_path
        pipeline.loaded_kwargs = kwargs
        return pipeline


@pytest.fixture
def tiny_model_class():
    return TinyObject3DModel


@pytest.fixture
def unreviewed_model_class():
    return UnreviewedObject3DModel


@pytest.fixture
def other_tiny_model_class():
    return OtherTinyObject3DModel


@pytest.fixture
def tiny_model_subclass():
    return TinyObject3DModelSubclass


@pytest.fixture
def tiny_pipeline_class():
    return TinyObject3DPipeline


@pytest.fixture
def other_tiny_pipeline_class():
    return OtherTinyObject3DPipeline


@pytest.fixture
def tiny_pipeline_subclass():
    return TinyObject3DPipelineSubclass


@pytest.fixture
def dispatch_pipeline_class():
    DispatchObject3DPipeline.load_calls.clear()
    return DispatchObject3DPipeline
