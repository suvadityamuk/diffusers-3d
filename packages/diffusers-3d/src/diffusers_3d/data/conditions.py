from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from ..objects import CameraRig, Object3D
from ..objects._validation import (
    Object3DValidationError,
    TensorShapeError,
    validate_shared_device,
    validate_tensor,
)
from ..objects.base import TensorDataMixin


@dataclass(frozen=True, slots=True)
class TextCondition(TensorDataMixin):
    """A single validated text condition."""

    text: str
    negative_text: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise Object3DValidationError("text must be a non-empty string")
        if self.negative_text is not None and not isinstance(self.negative_text, str):
            raise Object3DValidationError("negative_text must be a string or None")


@dataclass(frozen=True, slots=True)
class ImageCondition(TensorDataMixin):
    """A channel-first image with an optional matching camera."""

    image: torch.Tensor
    camera: CameraRig | None = None
    mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("image", self.image, rank=3, floating=True)
        channels, height, width = self.image.shape
        if channels not in (1, 3, 4) or height == 0 or width == 0:
            raise TensorShapeError("image must have shape (1|3|4, height, width) with non-zero spatial dimensions")
        if self.mask is not None:
            validate_tensor("mask", self.mask, rank=3, trailing_shape=(height, width), floating=True)
            if self.mask.shape[0] != 1:
                raise TensorShapeError("mask must have shape (1, height, width)")
            if bool(((self.mask < 0) | (self.mask > 1)).any()):
                raise Object3DValidationError("mask values must be in [0, 1]")
        if self.camera is not None:
            if type(self.camera) is not CameraRig:
                raise Object3DValidationError("camera must be an exact CameraRig")
            self.camera.validate()
            if self.camera.world_to_camera.shape[0] != 1:
                raise TensorShapeError("an ImageCondition camera must contain exactly one camera")
            expected_size = self.camera.image_sizes.new_tensor([height, width])
            if not torch.equal(self.camera.image_sizes[0], expected_size):
                raise TensorShapeError("camera image size must match the condition image")
        validate_shared_device(self.tensor_items())


@dataclass(frozen=True, slots=True)
class MultiViewCondition(TensorDataMixin):
    """A stack of channel-first images and one exact camera per view."""

    images: torch.Tensor
    cameras: CameraRig
    masks: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        view_count, channels, height, width = self.images.shape
        if view_count == 0 or channels not in (1, 3, 4) or height == 0 or width == 0:
            raise TensorShapeError("images must have shape (views, 1|3|4, height, width) with non-zero dimensions")
        if type(self.cameras) is not CameraRig:
            raise Object3DValidationError("cameras must be an exact CameraRig")
        self.cameras.validate()
        if self.cameras.world_to_camera.shape[0] != view_count:
            raise TensorShapeError("cameras must contain exactly one camera per image")
        expected_sizes = self.cameras.image_sizes.new_tensor([height, width]).expand(view_count, -1)
        if not torch.equal(self.cameras.image_sizes, expected_sizes):
            raise TensorShapeError("all camera image sizes must match the condition images")
        if self.masks is not None:
            validate_tensor("masks", self.masks, rank=4, trailing_shape=(1, height, width), floating=True)
            if self.masks.shape[0] != view_count:
                raise TensorShapeError("masks must have shape (views, 1, height, width)")
            if bool(((self.masks < 0) | (self.masks > 1)).any()):
                raise Object3DValidationError("masks values must be in [0, 1]")
        validate_shared_device(self.tensor_items())


Object3DCondition: TypeAlias = TextCondition | ImageCondition | MultiViewCondition


@dataclass(frozen=True, slots=True)
class Object3DExample(TensorDataMixin):
    """One typed condition and its object-native training target."""

    target: Object3D
    condition: Object3DCondition
    example_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, expensive: bool = False) -> None:
        if not isinstance(self.target, Object3D):
            raise Object3DValidationError("target must implement the Object3D protocol")
        if type(self.condition) not in (TextCondition, ImageCondition, MultiViewCondition):
            raise Object3DValidationError("condition must be a TextCondition, ImageCondition, or MultiViewCondition")
        if self.example_id is not None and (not isinstance(self.example_id, str) or not self.example_id):
            raise Object3DValidationError("example_id must be a non-empty string or None")
        self.target.validate(expensive=expensive)
        self.condition.validate()
        if isinstance(self.condition, (ImageCondition, MultiViewCondition)):
            cameras = self.condition.camera if isinstance(self.condition, ImageCondition) else self.condition.cameras
            if cameras is not None and cameras.coordinate_system is not self.target.coordinate_system:
                raise Object3DValidationError("condition cameras and target must use the same coordinate system")
        validate_shared_device(self.tensor_items())


__all__ = [
    "ImageCondition",
    "MultiViewCondition",
    "Object3DCondition",
    "Object3DExample",
    "TextCondition",
]
