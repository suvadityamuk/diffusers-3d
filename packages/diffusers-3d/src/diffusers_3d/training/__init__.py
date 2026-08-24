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
from .trainer import (
    ACCELERATOR_STATE_DIRECTORY,
    TRAINER_STATE_NAME,
    TRAINER_STATE_SCHEMA_VERSION,
    Object3DTrainer,
)
from .types import (
    ComponentPolicy,
    FineTuneKind,
    FineTuneStrategy3D,
    FrozenComponentPolicy,
    FullFineTune,
    LoRAFineTune,
    MetricValue,
    TrainingConfig3D,
    TrainingStep3DOutput,
    TrainingSummary3D,
)

__all__ = [
    "ACCELERATOR_STATE_DIRECTORY",
    "TRAINING_ADAPTER_NAME",
    "TRAINING_MANIFEST_NAME",
    "TRAINING_MANIFEST_SCHEMA",
    "TRAINING_MANIFEST_VERSION",
    "ComponentPolicy",
    "FineTuneKind",
    "FineTuneStrategy3D",
    "FrozenComponentPolicy",
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
    "TrainingSummary3D",
    "TrainingStep3DOutput",
    "TrainingTargetError",
    "TRAINER_STATE_NAME",
    "TRAINER_STATE_SCHEMA_VERSION",
    "create_training_recipe_registry",
    "trainable_parameter_hash",
]
