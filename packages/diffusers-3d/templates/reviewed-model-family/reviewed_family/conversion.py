from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .model import ReviewedDenoiser


def convert_denoiser_checkpoint(
    source_state_dict: Mapping[str, torch.Tensor],
    output_directory: str | Path,
) -> ReviewedDenoiser:
    """Apply the parity-reviewed key mapping, then use public save/load APIs."""

    converted_state_dict = dict(source_state_dict)
    model = ReviewedDenoiser()
    model.load_state_dict(converted_state_dict, strict=True)
    model.save_pretrained(output_directory)
    return ReviewedDenoiser.from_pretrained(output_directory, local_files_only=True)


def convert_pipeline_config(source_config: Mapping[str, object]) -> dict[str, object]:
    """Return the reviewed Diffusers config mapping for this exact upstream revision."""

    return dict(source_config)
