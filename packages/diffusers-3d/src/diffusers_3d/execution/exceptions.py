class Object3DExecutionError(RuntimeError):
    """Base class for object-3D execution contract failures."""


class Object3DMetadataError(Object3DExecutionError):
    """Raised when object-3D metadata is missing or malformed."""


class Object3DSchemaError(Object3DMetadataError):
    """Raised when object-3D metadata uses an unsupported schema."""


class Object3DRegistrationError(Object3DExecutionError):
    """Raised when exact reviewed registration cannot be established."""


class Object3DTaskError(Object3DExecutionError):
    """Raised when a requested object-3D task is missing or ambiguous."""


class Object3DLoadingError(Object3DExecutionError):
    """Raised when metadata or a reviewed pipeline cannot be loaded."""


__all__ = [
    "Object3DExecutionError",
    "Object3DLoadingError",
    "Object3DMetadataError",
    "Object3DRegistrationError",
    "Object3DSchemaError",
    "Object3DTaskError",
]
