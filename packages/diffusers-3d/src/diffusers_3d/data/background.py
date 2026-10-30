from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from .conditions import ImageCondition, _image_tensor

BackgroundRemover = Callable[[Image.Image], Image.Image]


class RembgBackgroundRemover:
    """Background removal with `rembg <https://github.com/danielgatis/rembg>`_: ``rembg.remove(image, session=...)``.

    ``rembg`` is MIT-licensed and the ``u2net`` weights are Apache-2.0; they download on first use. The session is
    always named: recent rembg releases default ``rembg.new_session()`` to ``bria-rmbg`` (non-commercial RMBG-2.0
    weights), where the rembg 2.0.65 upstream pinned defaulted to ``u2net``. ``remove_kwargs`` go to ``rembg.remove``.
    """

    def __init__(self, model_name: str = "u2net", **remove_kwargs: Any) -> None:
        import rembg

        self._remove = rembg.remove
        self._session = rembg.new_session(model_name)
        self._remove_kwargs = remove_kwargs

    def __call__(self, image: Image.Image) -> Image.Image:
        return self._remove(image, session=self._session, **self._remove_kwargs)


def lacks_alpha(condition: ImageCondition) -> bool:
    """No mask, and no alpha channel or a fully opaque one: upstream's ``mode != "RGBA" or alpha.min() == 255``.

    As upstream, the alpha of images loaded from another PIL mode (LA, or a palette with transparency) is ignored.
    """

    if condition.mask is not None:
        return False
    if condition.source_mode is not None and condition.source_mode != "RGBA":
        return True
    return condition.image.shape[0] != 4 or not bool(condition.image[3].lt(254.5 / 255).any())


def lacks_alpha_channel(condition: ImageCondition) -> bool:
    """No mask and no alpha channel (upstream ``mode == "RGB"``); a fully opaque alpha does not trigger removal.

    Upstream Step1X-3D texturing converts greyscale images to RGBA first, so it never cuts them out; here they count as
    RGB.
    """

    return condition.mask is None and condition.image.shape[0] != 4


def has_ambiguous_alpha(condition: ImageCondition) -> bool:
    """No mask, and no alpha channel or one failing TripoSG's ``is_valid_alpha``.

    That check bins the 8-bit alpha into 20 bins and needs at least 1% of the pixels in each end bin (alpha <= 12 and
    alpha >= 244).
    """

    if condition.mask is not None:
        return False
    if condition.image.shape[0] != 4:
        return True
    alpha = condition.image[3].mul(255).round()
    minimum_count = alpha.numel() * 0.01
    return bool((alpha <= 12).sum() < minimum_count or (alpha >= 244).sum() < minimum_count)


def downscale_longest_side(image: Image.Image, size: int = 1024) -> Image.Image:
    """TRELLIS, TRELLIS.2, and Pixal3D: LANCZOS-downscale so the longer side is at most ``size`` (truncating)."""

    scale = min(1, size / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def resize_shortest_side(image: Image.Image, size: int = 1024) -> Image.Image:
    """IDOL and TripoSplat: LANCZOS-resize so the shorter side is ``size`` (rounding), up or down."""

    scale = size / min(image.size)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


@dataclass(frozen=True)
class BackgroundRemoval:
    """How a pipeline cuts out the object of inputs without usable alpha, following its upstream.

    Args:
        remover: Builds the remover on first use: a ``PIL RGB -> PIL RGBA`` callable that keeps the size. It may have
            a ``to(device)`` method, which is called with the pipeline's execution device before each use.
        trigger: Whether a condition needs removal.
        resize: Upstream's resize of the 8-bit RGB image before the remover; later steps see its output size.
        resize_in_source_mode: Whether upstream resizes the image in the mode it was loaded in before converting it
            to RGB, as TRELLIS.2, Pixal3D, IDOL, and TripoSplat do (see :func:`remove_background`).
        automatic: Whether removal runs without :meth:`~diffusers_3d.Object3DPipeline.enable_background_removal`.
    """

    remover: Callable[[], BackgroundRemover] = RembgBackgroundRemover
    trigger: Callable[[ImageCondition], bool] = lacks_alpha
    resize: Callable[[Image.Image], Image.Image] | None = None
    resize_in_source_mode: bool = False
    automatic: bool = True


def _source_mode_image(rgba: np.ndarray, mode: str | None) -> Image.Image:
    """The 8-bit image in the PIL mode it was loaded from, where Pillow resizes that mode differently from RGB.

    Pillow resizes palette and bilevel images with NEAREST, which commutes with the palette lookup, and LA images
    with premultiplied alpha.
    """

    if mode in ("P", "1"):
        colors, indices = np.unique(rgba[..., :3].reshape(-1, 3), axis=0, return_inverse=True)
        image = Image.fromarray(indices.reshape(rgba.shape[:2]).astype(np.uint8), "P")
        image.putpalette(colors.astype(np.uint8).tobytes())
        return image
    if mode == "LA" and rgba.shape[-1] == 4:
        return Image.fromarray(rgba[..., [0, 3]], "LA")
    return Image.fromarray(rgba[..., :3], "RGB")


def remove_background(
    condition: ImageCondition,
    remover: BackgroundRemover,
    *,
    trigger: Callable[[ImageCondition], bool] = lacks_alpha,
    resize: Callable[[Image.Image], Image.Image] | None = None,
    resize_in_source_mode: bool = False,
) -> ImageCondition:
    """Run ``remover`` on the 8-bit RGB of a condition that ``trigger`` selects, after ``resize`` if given.

    With ``resize_in_source_mode``, palette and LA images are resized in that mode and converted to RGB afterwards,
    as upstreams that resize the loaded PIL image do. Returns an RGBA condition whose alpha marks the object; other
    conditions are returned unchanged.
    """

    if not trigger(condition):
        return condition
    channels = condition.image.shape[0]
    pixels = condition.image[[0, 0, 0] if channels == 1 else [0, 1, 2]]
    if channels == 4:
        pixels = torch.cat([pixels, condition.image[3:]])
    pixels = pixels.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(np.ascontiguousarray(pixels[..., :3]), "RGB")
    if resize is not None:
        if resize_in_source_mode:
            image = resize(_source_mode_image(pixels, condition.source_mode)).convert("RGB")
        else:
            image = resize(image)
        if condition.camera is not None and image.size != tuple(reversed(condition.image.shape[1:])):
            raise ValueError("background removal resizes this image, which would invalidate its camera")
    result = remover(image)
    if not isinstance(result, Image.Image) or result.size != image.size:
        raise ValueError("the background remover must return an image of the input size")
    rgba = _image_tensor(result.convert("RGBA") if result.mode != "RGBA" else result)
    return ImageCondition(image=rgba.to(condition.image.device), camera=condition.camera)


__all__ = [
    "BackgroundRemoval",
    "BackgroundRemover",
    "RembgBackgroundRemover",
    "downscale_longest_side",
    "has_ambiguous_alpha",
    "lacks_alpha",
    "lacks_alpha_channel",
    "remove_background",
    "resize_shortest_side",
]
