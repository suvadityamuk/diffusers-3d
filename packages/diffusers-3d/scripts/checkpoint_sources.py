"""Fetch the released TRELLIS / TRELLIS.2 checkpoints and convert them into diffusers-3d pipeline folders.

Shared by ``gpu_smoke.py`` and ``publish_hub_checkpoints.py``. Conversions are cached under ``work / "converted"``.
"""

from __future__ import annotations

import json
from pathlib import Path

RELEASES = {
    "trellis-image": {
        "source": "microsoft/TRELLIS-image-large",
        "conditioner": "facebook/dinov2-with-registers-large",
        "conditioner_patterns": None,
    },
    "trellis-text": {
        "source": "microsoft/TRELLIS-text-large",
        "conditioner": "openai/clip-vit-large-patch14",
        "conditioner_patterns": ["*.json", "*.txt", "model.safetensors"],
    },
    "trellis2": {
        "source": "microsoft/TRELLIS.2-4B",
        "conditioner": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "conditioner_patterns": None,
    },
}


def download(repo_id: str, work: Path, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, local_dir=work / repo_id.replace("/", "__"), allow_patterns=allow_patterns))


def materialize_shared_components(source: Path) -> None:
    """Fetch components a release references from another Hub repo (``owner/repo/ckpts/name``) into ``source``.

    Both converters resolve references relative to the source folder, so placing the pair at
    ``source/owner/repo/ckpts/name.*`` makes the released ``pipeline.json`` convertible unchanged.
    """

    from huggingface_hub import snapshot_download

    pipeline = json.loads((source / "pipeline.json").read_text())
    for reference in pipeline["args"]["models"].values():
        parts = reference.split("/")
        if len(parts) < 3 or (source / reference).with_suffix(".json").is_file():
            continue
        repo_id, member = "/".join(parts[:2]), "/".join(parts[2:])
        snapshot_download(repo_id, local_dir=source / parts[0] / parts[1], allow_patterns=[f"{member}.*"])


def convert_release(name: str, work: Path) -> Path:
    """Download and convert one of :data:`RELEASES`; returns the converted pipeline folder."""

    release = RELEASES[name]
    target = work / "converted" / name
    if (target / "model_index.json").is_file():
        return target
    source = download(release["source"], work)
    materialize_shared_components(source)
    conditioner = download(release["conditioner"], work, release["conditioner_patterns"])
    if name == "trellis2":
        from diffusers_3d.families.trellis2.conversion import convert_trellis2_checkpoint

        convert_trellis2_checkpoint(source, target, conditioner_path=conditioner)
    else:
        from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint

        convert_trellis_checkpoint(source, target, conditioner_path=conditioner)
    return target


__all__ = ["RELEASES", "convert_release", "download", "materialize_shared_components"]
