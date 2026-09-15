"""Image to 3D with TRELLIS.2.

The whole thing is three calls::

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("/path/to/trellis2").to("cuda")
    output = pipeline(ImageCondition(image=rgba))
    ovoxel = output.objects[0]                      # OVoxelAsset: dual-grid shape + PBR channels

The rest of this file is the surrounding detail: reading an image into an ``ImageCondition``, the
sampler knobs, what comes back, and how to write each asset type to disk.

Run it::

    # against a converted checkpoint (see ``diffusers-3d-convert-trellis2``)
    python -m diffusers_3d.families.trellis2.examples.image_to_3d \
        --model /path/to/trellis2 --image chair.png --output out/

    # offline, on CPU, with random tiny weights; exercises every call but produces no real geometry
    python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/
    python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --sparse-only --output out/

TRELLIS.2 is image-conditioned; there is no text-to-3D variant. A full checkpoint runs three
stages: sparse structure (which voxels are occupied), shape and texture SLAT flows on those voxels,
and the O-Voxel decoders that turn the SLATs into a dual-grid surface with PBR channels. ``formats``
picks which of those intermediate and final assets come back.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    ImageCondition,
    MeshAsset,
    Object3D,
    Object3DPipelineOutput,
    OVoxelAsset,
    SparseVoxelAsset,
    Trellis2ImageTo3DPipeline,
    write_ovoxel_npz,
)

SPARSE_ONLY_FORMATS = ("sparse_structure",)
ALL_STAGE_FORMATS = ("sparse_structure", "shape_slat", "texture_slat", "o_voxel")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------


def load_pipeline(model: str, *, device: str, dtype: torch.dtype) -> Trellis2ImageTo3DPipeline:
    """A converted checkpoint loads like any Diffusers pipeline.

    ``model`` is a local directory or a Hub repository ID. ``AutoPipelineForImageTo3D`` reads the
    ``object3d_model_index.json`` sidecar, verifies each component's class before downloading, and
    returns the concrete pipeline. If you already know the family, the concrete class works too::

        Trellis2ImageTo3DPipeline.from_pretrained(model)
    """

    pipeline = AutoPipelineForImageTo3D.from_pretrained(model)
    return pipeline.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# 2. Condition
# ---------------------------------------------------------------------------


def load_image_condition(path: str | Path) -> ImageCondition:
    """``ImageCondition.image`` is a float ``(C, H, W)`` tensor in ``[0, 1]`` with 1, 3, or 4 channels.

    With an alpha channel (or a separate ``mask=``) the pipeline crops and recenters the foreground
    the way the released TRELLIS.2 code does. Plain RGB is treated as already background-removed; the
    pipeline never runs a background remover for you.
    """

    with Image.open(path) as image:
        mode = "RGBA" if "A" in image.getbands() else "RGB"
        pixels = torch.from_numpy(np.asarray(image.convert(mode), dtype=np.float32))
    return ImageCondition(image=pixels.permute(2, 0, 1) / 255.0)


def synthetic_image_condition(size: int = 64) -> ImageCondition:
    """An RGBA gradient with a transparent border, used when no image is supplied."""

    ramp = torch.linspace(0.0, 1.0, size)
    rgb = torch.stack(
        [ramp[None, :].expand(size, size), ramp[:, None].expand(size, size), torch.full((size, size), 0.5)]
    )
    alpha = torch.zeros(1, size, size)
    alpha[:, size // 8 : -size // 8, size // 8 : -size // 8] = 1.0
    return ImageCondition(image=torch.cat([rgb, alpha]))


# ---------------------------------------------------------------------------
# 3. Generate
# ---------------------------------------------------------------------------


def generate(
    pipeline: Trellis2ImageTo3DPipeline,
    conditions: Sequence[ImageCondition],
    *,
    formats: Sequence[str] = ALL_STAGE_FORMATS,
    steps: int | None = None,
    guidance_strength: float | None = None,
    seed: int = 0,
) -> Object3DPipelineOutput:
    """Call the pipeline. A list of conditions is a batch; you get one object per image per format.

    Sampler settings are per-stage dicts. Anything you leave out falls back to the released defaults
    stored in ``pipeline.config`` (sparse structure: 12 steps, guidance 7.5, rescale 0.7, interval
    (0.6, 1.0), ``rescale_t`` 5.0). ``pipeline(condition)`` with no keyword arguments is a valid call
    and returns just the final O-Voxel; ``pipeline_type`` ("512", "1024", "1024_cascade", "1536_cascade")
    selects the released resolution preset and defaults to the one stored in the checkpoint.
    """

    sampler_params: dict[str, float | int] = {}
    if steps is not None:
        sampler_params["steps"] = steps
    if guidance_strength is not None:
        sampler_params["guidance_strength"] = guidance_strength

    return pipeline(
        list(conditions),
        formats=tuple(formats),
        sparse_structure_sampler_params=sampler_params,
        shape_slat_sampler_params=sampler_params,
        texture_slat_sampler_params=sampler_params,
        generator=torch.Generator(device=pipeline._execution_device).manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# 4. Use the result
# ---------------------------------------------------------------------------


def label(asset: Object3D) -> str:
    """Assets carry a JSON-safe ``metadata`` dict; ``representation`` and ``stage`` say where one came from."""

    representation = asset.metadata.get("representation") or type(asset).__name__
    stage = asset.metadata.get("stage")
    return f"{stage}_{representation}" if stage and stage not in representation else str(representation)


def describe(asset: Object3D) -> str:
    stage = label(asset)
    if isinstance(asset, SparseVoxelAsset):
        return (
            f"SparseVoxelAsset[{stage}] {asset.coordinates.shape[0]} voxels, "
            f"features {tuple(asset.features.shape)}, resolution {asset.metadata.get('resolution')}"
        )
    if isinstance(asset, OVoxelAsset):
        return (
            f"OVoxelAsset[{stage}] {asset.active_coordinates.shape[0]} active cells, "
            f"base_color {tuple(asset.base_color.shape)}, resolution {asset.metadata.get('resolution')}"
        )
    if isinstance(asset, MeshAsset):
        return f"MeshAsset {tuple(asset.vertices.shape)} vertices, {tuple(asset.faces.shape)} faces"
    return f"{type(asset).__name__}[{stage}]"


def save(asset: Object3D, path: Path) -> Path:
    """Serialization is per representation.

    Sparse voxels have no interchange format, so their tensors go through ``torch.save``. O-Voxels use
    the package's pure-NumPy ``.npz`` codec, which the official TRELLIS.2 tooling reads. Meshes export
    through the optional trimesh backend.
    """

    if isinstance(asset, OVoxelAsset):
        target = path.with_suffix(".npz")
        write_ovoxel_npz(target, asset)
        return target
    if isinstance(asset, MeshAsset):
        from diffusers_3d import TrimeshBackend

        target = path.with_suffix(".glb")
        TrimeshBackend().export_mesh(asset, target)
        return target
    target = path.with_suffix(".pt")
    torch.save(asset.to("cpu"), target)
    return target


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="Converted TRELLIS.2 checkpoint directory or Hub repository ID.")
    source.add_argument("--tiny", action="store_true", help="Use randomly initialised tiny CPU components.")
    parser.add_argument("--image", action="append", default=[], help="Input image; repeat for a batch.")
    parser.add_argument("--output", type=Path, default=Path("trellis2-output"), help="Directory for assets.")
    parser.add_argument("--steps", type=int, default=None, help="Sampler steps (default: released preset).")
    parser.add_argument("--guidance-strength", type=float, default=None, help="CFG strength (default: preset).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Stop after the sparse structure instead of running the SLAT and O-Voxel stages.",
    )
    args = parser.parse_args(argv)
    if not args.tiny and args.model is None:
        parser.error("pass --model to load a checkpoint or --tiny for the offline demo")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    formats = SPARSE_ONLY_FORMATS if args.sparse_only else ALL_STAGE_FORMATS
    if args.tiny:
        from .tiny_components import build_tiny_pipeline

        pipeline = build_tiny_pipeline(include_slat=not args.sparse_only)
        steps = args.steps if args.steps is not None else 2
    else:
        pipeline = load_pipeline(args.model, device=args.device, dtype=getattr(torch, args.dtype))
        steps = args.steps

    conditions = [load_image_condition(path) for path in args.image] or [synthetic_image_condition()]
    output = generate(
        pipeline,
        conditions,
        formats=formats,
        steps=steps,
        guidance_strength=args.guidance_strength,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(output.objects)} object(s) from {len(conditions)} image(s):")
    for index, asset in enumerate(output.objects):
        written = save(asset, args.output / f"{index:02d}-{label(asset)}")
        print(f"  {describe(asset)} -> {written}")
    if output.latents is not None:
        print(f"sparse-structure latents: {tuple(output.latents.latents.shape)}")


if __name__ == "__main__":
    main()
