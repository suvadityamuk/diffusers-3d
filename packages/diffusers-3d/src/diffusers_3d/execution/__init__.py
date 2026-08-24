"""Nominal model and pipeline contracts with exact config-first loading."""

from .auto_pipeline import (
    IMAGE_TO_3D_TASK,
    TEXT_TO_3D_TASK,
    AutoPipelineFor3D,
    AutoPipelineForImageTo3D,
    AutoPipelineForTextTo3D,
)
from .exceptions import (
    Object3DExecutionError,
    Object3DLoadingError,
    Object3DMetadataError,
    Object3DRegistrationError,
    Object3DSchemaError,
    Object3DTaskError,
)
from .metadata import (
    OBJECT3D_API_VERSION,
    OBJECT3D_MODEL_INDEX_NAME,
    OBJECT3D_SCHEMA_VERSION,
    ContributionStatus,
    Object3DModelIndex,
    Object3DModelMetadata,
    ReviewStatus,
)
from .models import Object3DModel
from .pipelines import ModularObject3DPipeline, Object3DPipeline
from .registry import (
    Object3DModelRegistration,
    Object3DModelRegistry,
    Object3DPipelineRegistration,
    Object3DPipelineRegistry,
)

__all__ = [
    "IMAGE_TO_3D_TASK",
    "OBJECT3D_API_VERSION",
    "OBJECT3D_MODEL_INDEX_NAME",
    "OBJECT3D_SCHEMA_VERSION",
    "TEXT_TO_3D_TASK",
    "AutoPipelineFor3D",
    "AutoPipelineForImageTo3D",
    "AutoPipelineForTextTo3D",
    "ContributionStatus",
    "ModularObject3DPipeline",
    "Object3DExecutionError",
    "Object3DLoadingError",
    "Object3DMetadataError",
    "Object3DModel",
    "Object3DModelIndex",
    "Object3DModelMetadata",
    "Object3DModelRegistration",
    "Object3DModelRegistry",
    "Object3DPipeline",
    "Object3DPipelineRegistration",
    "Object3DPipelineRegistry",
    "Object3DRegistrationError",
    "Object3DSchemaError",
    "Object3DTaskError",
    "ReviewStatus",
]
