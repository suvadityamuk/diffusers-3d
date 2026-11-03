"""Typed conditions and image preprocessing for 3D pipelines."""

from .background import (
    BackgroundRemoval,
    BackgroundRemover,
    RembgBackgroundRemover,
    downscale_longest_side,
    has_ambiguous_alpha,
    lacks_alpha,
    lacks_alpha_channel,
    remove_background,
    resize_shortest_side,
)
from .birefnet import RMBG2_REPO, RMBG2_REVISION, BiRefNet, BiRefNetBackgroundRemover
from .conditions import (
    ImageCondition,
    ImageInput,
    MultiViewCondition,
    Object3DCondition,
    TextCondition,
    as_image_conditions,
)
from .image_processing import preprocess_image_condition, validate_image_condition_pixels
from .rmbg import RMBG14_REPO, RMBG14_REVISION, BriaRMBG, RMBG14BackgroundRemover

__all__ = [
    "RMBG14_REPO",
    "RMBG14_REVISION",
    "RMBG2_REPO",
    "RMBG2_REVISION",
    "BriaRMBG",
    "RMBG14BackgroundRemover",
    "BackgroundRemoval",
    "BackgroundRemover",
    "BiRefNet",
    "BiRefNetBackgroundRemover",
    "RembgBackgroundRemover",
    "downscale_longest_side",
    "has_ambiguous_alpha",
    "lacks_alpha",
    "lacks_alpha_channel",
    "remove_background",
    "resize_shortest_side",
    "ImageCondition",
    "ImageInput",
    "MultiViewCondition",
    "Object3DCondition",
    "TextCondition",
    "as_image_conditions",
    "preprocess_image_condition",
    "validate_image_condition_pixels",
]
