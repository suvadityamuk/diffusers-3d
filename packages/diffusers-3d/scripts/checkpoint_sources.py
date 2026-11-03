"""Fetch released 3D checkpoints and convert them into diffusers-3d pipeline folders.

Shared by ``gpu_smoke.py`` and ``publish_hub_checkpoints.py``. Conversions are cached under ``work / "converted"``.
Only each model's own weights are converted: third-party components a pipeline uses unchanged (DINOv3, DINOv2,
NAF, MoGe, ...) are never downloaded here, since the pipelines load them from their original sources.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SAM3D_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"


def download(repo_id: str, work: Path, allow_patterns: list[str] | None = None, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = work / repo_id.replace("/", "__")
    return Path(snapshot_download(repo_id, revision=revision, local_dir=local_dir, allow_patterns=allow_patterns))


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


def _convert_trellis(name: str, work: Path, target: Path) -> None:
    release = RELEASES[name]
    source = download(release["source"], work)
    if name == "trellis2":
        # TRELLIS.2's cross-repository sparse-structure decoder loads from TRELLIS-image-large; it is not fetched.
        from diffusers_3d.families.trellis2.conversion import convert_trellis2_checkpoint

        convert_trellis2_checkpoint(source, target)
    else:
        from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint

        materialize_shared_components(source)
        encoder = release["encoder"] and download(release["encoder"], work, release["encoder_patterns"])
        convert_trellis_checkpoint(source, target, encoder_path=encoder)


def _convert_triposg(work: Path, target: Path) -> None:
    from diffusers_3d.families.triposg.conversion import convert_triposg_checkpoint

    convert_triposg_checkpoint(RELEASES["triposg"]["source"], target)


def _convert_step1x3d(work: Path, target: Path) -> None:
    from diffusers_3d.families.step1x3d.conversion import STEP1X3D_CHECKPOINT_REVISION, convert_step1x3d_checkpoint

    folder = "Step1X-3D-Geometry-1300m"
    source = download(RELEASES["step1x3d"]["source"], work, [f"{folder}/*"], STEP1X3D_CHECKPOINT_REVISION)
    convert_step1x3d_checkpoint(source / folder, target)


def _convert_hunyuan3d(work: Path, target: Path) -> None:
    from diffusers_3d.families.hunyuan3d.conversion import convert_hunyuan3d_checkpoint

    convert_hunyuan3d_checkpoint(RELEASES["hunyuan3d"]["source"], target)


def _convert_step1x3d_texture(work: Path, target: Path) -> None:
    from diffusers_3d.families.step1x3d.conversion import STEP1X3D_CHECKPOINT_REVISION
    from diffusers_3d.families.step1x3d.texture_conversion import convert_step1x3d_texture_checkpoint

    folder = "Step1X-3D-Texture"
    source = download(RELEASES["step1x3d-texture"]["source"], work, [f"{folder}/*"], STEP1X3D_CHECKPOINT_REVISION)
    convert_step1x3d_texture_checkpoint(source / folder, target)


def _convert_hunyuan3d_paint(work: Path, target: Path) -> None:
    from diffusers_3d.families.hunyuan3d.paint_conversion import convert_hunyuan3d_paint_checkpoint

    convert_hunyuan3d_paint_checkpoint(RELEASES["hunyuan3d-paint"]["source"], target)


def _convert_sam3d(work: Path, target: Path) -> None:
    from diffusers_3d.families.sam3d_objects.conversion import convert_sam3d_checkpoint

    source = download(RELEASES["sam3d-objects"]["source"], work, ["checkpoints/*"], SAM3D_REVISION)
    convert_sam3d_checkpoint(source / "checkpoints", target)


def _convert_pixal3d(work: Path, target: Path) -> None:
    from diffusers_3d.families.pixal3d.conversion import PIXAL3D_WEIGHTS_REVISION, convert_pixal3d_checkpoint

    source = download(RELEASES["pixal3d"]["source"], work, revision=PIXAL3D_WEIGHTS_REVISION)
    convert_pixal3d_checkpoint(source, target)


RELEASES = {
    "trellis-image": {
        "source": "microsoft/TRELLIS-image-large",
        "encoder": None,  # the converter's pinned TRELLIS_ENCODER_CHECKPOINTS entry
        "encoder_patterns": None,
        "convert": lambda work, target: _convert_trellis("trellis-image", work, target),
    },
    "trellis-text": {
        "source": "microsoft/TRELLIS-text-large",
        "encoder": None,  # the converter's pinned TRELLIS_ENCODER_CHECKPOINTS entry
        "encoder_patterns": ["*.json", "*.txt", "model.safetensors"],
        "convert": lambda work, target: _convert_trellis("trellis-text", work, target),
    },
    "trellis2": {
        "source": "microsoft/TRELLIS.2-4B",
        "convert": lambda work, target: _convert_trellis("trellis2", work, target),
    },
    "triposg": {"source": "VAST-AI/TripoSG", "convert": _convert_triposg},
    "step1x3d": {"source": "stepfun-ai/Step1X-3D", "convert": _convert_step1x3d},
    "step1x3d-texture": {"source": "stepfun-ai/Step1X-3D", "convert": _convert_step1x3d_texture},
    "hunyuan3d": {"source": "tencent/Hunyuan3D-2.1", "convert": _convert_hunyuan3d},
    "hunyuan3d-paint": {"source": "tencent/Hunyuan3D-2.1", "convert": _convert_hunyuan3d_paint},
    "sam3d-objects": {"source": "facebook/sam-3d-objects", "convert": _convert_sam3d},
    "pixal3d": {"source": "TencentARC/Pixal3D", "convert": _convert_pixal3d},
}


def external_components(folder: Path) -> dict[str, str]:
    """The third-party components the pipeline in ``folder`` loads from their own repositories: name -> repository."""

    import diffusers_3d

    index_path = folder / "model_index.json"
    if not index_path.is_file():
        return {}
    pipeline = getattr(diffusers_3d, json.loads(index_path.read_text())["_class_name"])
    return {name: component.repository for name, component in pipeline.external_components.items()}


def convert_release(name: str, work: Path) -> Path:
    """Download and convert one of :data:`RELEASES`; returns the converted pipeline folder."""

    target = work / "converted" / name
    stale = [target / folder for folder in ("conditioner", *external_components(target))]
    if (target / "model_index.json").is_file() and not any(path.exists() for path in stale):
        return target
    shutil.rmtree(target, ignore_errors=True)
    RELEASES[name]["convert"](work, target)
    return target


__all__ = ["RELEASES", "convert_release", "download", "external_components", "materialize_shared_components"]
