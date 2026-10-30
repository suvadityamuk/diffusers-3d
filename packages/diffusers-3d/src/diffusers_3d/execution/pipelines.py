from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import huggingface_hub
import torch
from diffusers import DiffusionPipeline

from ..data import ImageCondition, ImageInput, as_image_conditions
from ..data._hub import require_hub_access
from ..data.background import BackgroundRemoval, BackgroundRemover, remove_background

_HUB_KWARGS = ("cache_dir", "force_download", "proxies", "token", "local_files_only", "revision")


@dataclass(frozen=True)
class ExternalComponent:
    """A third-party component that loads from its original source and is never converted or redistributed.

    ``repository`` is a Hub repository, whose access is checked first, or a download URL; ``revision`` pins it (a
    commit, or the file's SHA-256 for a URL). ``load(repository, revision=..., dtype=..., **hub_kwargs)`` returns the
    component, converting it if the original is not in a loadable layout; ``hub_kwargs`` are ``token``,
    ``cache_dir``, ``local_files_only``, ``force_download`` and ``proxies`` as given to ``from_pretrained``.
    """

    repository: str
    revision: str
    load: Callable[..., Any]

    def __call__(self, name: str, pipeline: str, dtype: torch.dtype | None, hub_kwargs: dict[str, Any]) -> Any:
        if not hub_kwargs.get("local_files_only") and not self.repository.startswith(("https://", "http://")):
            require_hub_access(
                self.repository,
                used_for=f"{pipeline}'s {name!r}",
                alternative=f"pass {name}=... to from_pretrained yourself",
                token=hub_kwargs.get("token"),
            )
        return self.load(self.repository, revision=self.revision, dtype=dtype, **hub_kwargs)


def _component_dtype(dtype: Any, name: str) -> torch.dtype | None:
    if isinstance(dtype, dict):
        return dtype.get(name, dtype.get("default"))
    return dtype


class Object3DPipeline(DiffusionPipeline):
    """Base class for diffusers-3d pipelines; loading, saving, offloading, and ``.to()`` are stock Diffusers.

    ``background_removal`` declares how the family's upstream cuts out inputs without usable alpha: which remover, on
    which inputs, after which resize, and whether it runs by default. The remover is built on first use, kept in
    ``background_remover``, and never saved with the pipeline.
    """

    background_removal: BackgroundRemoval = BackgroundRemoval(automatic=False)
    # Third-party components (for example gated image encoders) that load from their own repositories.
    external_components: ClassVar[dict[str, ExternalComponent]] = {}
    background_remover: BackgroundRemover | None = None
    _background_removal_enabled: bool | None = None

    def enable_background_removal(self, remover: BackgroundRemover | None = None) -> None:
        """Remove backgrounds where the family's trigger fires, with ``remover`` (a ``PIL RGB -> RGBA PIL`` callable of
        the same size) or the family's remover."""

        self.background_remover = remover
        self._background_removal_enabled = True

    def disable_background_removal(self) -> None:
        self._background_removal_enabled = False

    def image_conditions(self, image: ImageInput | list[ImageInput]) -> list[ImageCondition]:
        """:func:`~diffusers_3d.data.as_image_conditions`, then background removal if it is on."""

        conditions = as_image_conditions(image)
        removal = self.background_removal
        enabled = self._background_removal_enabled
        if not (removal.automatic if enabled is None else enabled):
            return conditions

        def remover(image):
            if self.background_remover is None:
                self.background_remover = removal.remover()
            if callable(getattr(self.background_remover, "to", None)):
                self.background_remover.to(self._execution_device)
            return self.background_remover(image)

        return [
            remove_background(
                condition,
                remover,
                trigger=removal.trigger,
                resize=removal.resize,
                resize_in_source_mode=removal.resize_in_source_mode,
            )
            for condition in conditions
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any) -> Any:
        """``DiffusionPipeline.from_pretrained`` that also accepts Hub repo ids.

        Diffusers only downloads components whose classes come from Diffusers, Transformers, or code in the repo, so a
        Hub id is first downloaded here: ``model_index.json`` plus every component folder not passed as an argument.
        :attr:`external_components` that ``model_index.json`` names and that are not passed as arguments load from their
        original sources.
        """

        if "torch_dtype" in kwargs:
            kwargs.setdefault("dtype", kwargs.pop("torch_dtype"))
        hub_kwargs = {key: kwargs[key] for key in _HUB_KWARGS if key in kwargs}
        config = cls.load_config(pretrained_model_name_or_path, **hub_kwargs)
        external_kwargs = {key: value for key, value in hub_kwargs.items() if key != "revision"}
        for name, component in cls.external_components.items():
            value = config.get(name)
            if name not in kwargs and isinstance(value, list) and len(value) == 2 and value[0] is not None:
                dtype = _component_dtype(kwargs.get("dtype"), name)
                kwargs[name] = component(name, cls.__name__, dtype, external_kwargs)
        if not os.path.isdir(pretrained_model_name_or_path):
            hub_kwargs = {key: kwargs.pop(key) for key in _HUB_KWARGS if key in kwargs}
            folders = [
                name
                for name, value in config.items()
                if isinstance(value, list) and len(value) == 2 and value[0] is not None and name not in kwargs
            ]
            pretrained_model_name_or_path = huggingface_hub.snapshot_download(
                str(pretrained_model_name_or_path),
                allow_patterns=[cls.config_name, *(f"{folder}/*" for folder in folders)],
                ignore_patterns=["*.bin", "*.ckpt", "*.pt", "*.pth", "*.msgpack", "*.h5"],
                **hub_kwargs,
            )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str | os.PathLike[str], **kwargs: Any) -> None:
        """``DiffusionPipeline.save_pretrained`` without the :attr:`external_components`, which are never redistributed;
        ``model_index.json`` still names them so loading fetches them from their repositories."""

        self._saving_without_externals = True
        try:
            super().save_pretrained(save_directory, **kwargs)
        finally:
            del self._saving_without_externals
        for name in self.external_components:
            shutil.rmtree(Path(save_directory) / name, ignore_errors=True)

    @classmethod
    def _get_signature_keys(cls, obj):
        # Diffusers saves only the components this returns, then writes the full ``model_index.json``.
        expected, optional = super()._get_signature_keys(obj)
        if getattr(obj, "_saving_without_externals", False):
            expected = {name for name in expected if name not in cls.external_components}
        return expected, optional


__all__ = ["ExternalComponent", "Object3DPipeline"]
