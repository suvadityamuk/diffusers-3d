class Object3DTrainingError(RuntimeError):
    """Base exception for recipe-gated object-3D training."""


class TrainingConfigurationError(Object3DTrainingError, ValueError):
    """Raised when a training configuration or strategy is invalid."""


class TrainingRegistrationError(Object3DTrainingError):
    """Raised when a recipe does not have an exact reviewed registration."""


class TrainingTargetError(Object3DTrainingError, TypeError):
    """Raised when a recipe target or one of its components is not exact."""


class TrainingPolicyError(Object3DTrainingError, ValueError):
    """Raised when a requested fine-tuning strategy violates a component policy."""


class TrainableParameterError(Object3DTrainingError):
    """Raised when actual trainable parameters differ from the approved set."""


class TrainingDependencyError(Object3DTrainingError, ImportError):
    """Raised when a lazily required training dependency is unavailable."""


class TrainingCheckpointError(Object3DTrainingError):
    """Raised when exact training checkpoint state cannot be saved or loaded."""


class TrainingManifestError(TrainingCheckpointError, ValueError):
    """Raised when a training manifest is malformed."""


class TrainingManifestMismatchError(TrainingManifestError):
    """Raised when a resume manifest does not exactly match the current run."""


__all__ = [
    "Object3DTrainingError",
    "TrainableParameterError",
    "TrainingCheckpointError",
    "TrainingConfigurationError",
    "TrainingDependencyError",
    "TrainingManifestError",
    "TrainingManifestMismatchError",
    "TrainingPolicyError",
    "TrainingRegistrationError",
    "TrainingTargetError",
]
