"""Exact recipe registration, policy-gated preparation, and bounded training."""

from .exceptions import (
    Object3DTrainingError,
    TrainableParameterError,
    TrainingCheckpointError,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingManifestError,
    TrainingManifestMismatchError,
    TrainingPolicyError,
    TrainingRegistrationError,
    TrainingTargetError,
)
from .manifest import (
    TRAINING_MANIFEST_NAME,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_MANIFEST_VERSION,
    TrainingManifest3D,
    trainable_parameter_hash,
)
from .recipe import TRAINING_ADAPTER_NAME, TrainingRecipe3D
from .registry import TrainingRecipeRegistration, TrainingRecipeRegistry, create_training_recipe_registry
from .trainer import Object3DTrainer
from .types import (
    ComponentPolicy,
    FineTuneKind,
    FineTuneStrategy3D,
    FullFineTune,
    LoRAFineTune,
    MetricValue,
    TrainingConfig3D,
    TrainingStep3DOutput,
)

__all__ = [
    "TRAINING_ADAPTER_NAME",
    "TRAINING_MANIFEST_NAME",
    "TRAINING_MANIFEST_SCHEMA",
    "TRAINING_MANIFEST_VERSION",
    "ComponentPolicy",
    "FineTuneKind",
    "FineTuneStrategy3D",
    "FullFineTune",
    "LoRAFineTune",
    "MetricValue",
    "Object3DTrainer",
    "Object3DTrainingError",
    "TrainableParameterError",
    "TrainingCheckpointError",
    "TrainingConfig3D",
    "TrainingConfigurationError",
    "TrainingDependencyError",
    "TrainingManifest3D",
    "TrainingManifestError",
    "TrainingManifestMismatchError",
    "TrainingPolicyError",
    "TrainingRecipe3D",
    "TrainingRecipeRegistration",
    "TrainingRecipeRegistry",
    "TrainingRegistrationError",
    "TrainingStep3DOutput",
    "TrainingTargetError",
    "create_training_recipe_registry",
    "trainable_parameter_hash",
]
