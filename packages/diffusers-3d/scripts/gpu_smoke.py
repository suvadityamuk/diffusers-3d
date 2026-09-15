"""GPU smoke test: run both TRELLIS families on the released weights and exercise every compiled backend.

This is the manual/scheduled counterpart of the CPU test suite (see ``.github/workflows/diffusers_3d_gpu_smoke.yml``
for how it runs on Hugging Face Jobs). It converts the official checkpoints, generates from one image and one
prompt, pushes the outputs through gsplat, the O-Voxel runtime, CuMesh, the PBR facade, the texture baker, and the
radiance-field renderer, checks a few invariants (finite tensors, non-empty geometry, non-blank renders), and
writes ``report.json`` plus preview PNGs and GLBs to ``--output``. It exits non-zero when any stage fails.

    python scripts/gpu_smoke.py --work /tmp/smoke --output /tmp/smoke/out

Requires a CUDA device, ``HF_TOKEN`` with access to the gated DINOv3 weights, and the backends built by
``scripts/install_gpu_backends.sh`` (with ``ACCEPT_NVDIFFRAST_RESEARCH_LICENSE=1`` for the TRELLIS.2 stage).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch

EXAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/microsoft/TRELLIS/442aa1e1afb9014e80681d3bf604e8d728a86ee7/"
    "assets/example_image/T.png"
)
PROMPT = "a wooden rocking chair"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def timed(record: dict, name: str):
    """Context manager recording wall time and peak CUDA memory under ``record[name]``."""

    class _Timer:
        def __enter__(self):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            torch.cuda.synchronize()
            record[name] = {
                "seconds": round(time.perf_counter() - self.start, 3),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            }

    return _Timer()


def save_png(image: torch.Tensor, path: Path) -> None:
    """``(3, H, W)`` or ``(1, H, W)`` unit-range tensor -> PNG."""

    from PIL import Image

    array = (image.detach().float().clamp(0, 1).cpu() * 255).round().to(torch.uint8)
    array = array.expand(3, -1, -1) if array.shape[0] == 1 else array
    Image.fromarray(array.permute(1, 2, 0).numpy()).save(path)


def tile(images: torch.Tensor) -> torch.Tensor:
    """``(N, C, H, W)`` -> ``(C, H, N * W)``."""

    return torch.cat(list(images), dim=-1)


def load_image(source: str, work: Path):
    """RGBA file or URL -> ``ImageCondition``; the pipelines crop and recentre from the alpha channel."""

    import numpy as np
    from PIL import Image

    from diffusers_3d import ImageCondition

    if source.startswith(("http://", "https://")):
        import urllib.request

        target = work / "input.png"
        if not target.is_file():
            urllib.request.urlretrieve(source, target)
        source = str(target)
    rgba = np.array(Image.open(source).convert("RGBA"), dtype=np.float32) / 255.0
    return ImageCondition(image=torch.from_numpy(rgba).permute(2, 0, 1).contiguous())


def turntable(count: int, coordinate_system, *, image_size: int, device):
    from diffusers_3d.backends.texture_baking import sphere_hammersley_cameras

    return sphere_hammersley_cameras(count, image_size=image_size, coordinate_system=coordinate_system, device=device)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ------------------------------------------------------------------------------------------------ conversion


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


def convert_trellis2(work: Path) -> Path:
    from diffusers_3d.families.trellis2.conversion import convert_trellis2_checkpoint

    target = work / "converted" / "trellis2"
    if not (target / "model_index.json").is_file():
        source = download("microsoft/TRELLIS.2-4B", work)
        materialize_shared_components(source)
        conditioner = download("facebook/dinov3-vitl16-pretrain-lvd1689m", work)
        convert_trellis2_checkpoint(source, target, conditioner_path=conditioner)
    return target


def convert_trellis(work: Path) -> Path:
    from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint

    target = work / "converted" / "trellis"
    if not (target / "model_index.json").is_file():
        source = download("microsoft/TRELLIS-image-large", work)
        conditioner = download("facebook/dinov2-with-registers-large", work)
        convert_trellis_checkpoint(source, target, conditioner_path=conditioner)
    return target


def convert_trellis_text(work: Path) -> Path:
    from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint

    target = work / "converted" / "trellis-text"
    if not (target / "model_index.json").is_file():
        source = download("microsoft/TRELLIS-text-large", work)
        materialize_shared_components(source)
        conditioner = download("openai/clip-vit-large-patch14", work, ["*.json", "*.txt", "model.safetensors"])
        convert_trellis_checkpoint(source, target, conditioner_path=conditioner)
    return target


# ------------------------------------------------------------------------------------------------ stages


def stage_status(report: dict, device) -> None:
    from diffusers_3d.backends import BACKEND_REGISTRY, discover_backend

    report["device"] = torch.cuda.get_device_name(device)
    report["torch"] = torch.__version__
    statuses = {}
    for name in ("o_voxel", "cumesh", "flex_gemm", "nvdiffrast", "gsplat", "trimesh", "xatlas", "utils3d"):
        status = discover_backend(BACKEND_REGISTRY.get(name))
        statuses[name] = {
            "installed": status.installed,
            "importable": status.importable,
            "provenance_verified": status.provenance_verified,
        }
        log(f"backend {name:10s} {statuses[name]}")
    report["backends"] = statuses


def stage_trellis2(work: Path, out: Path, report: dict, condition, *, dtype, device) -> None:
    from diffusers_3d import Trellis2ImageTo3DPipeline
    from diffusers_3d.backends import CuMeshBackend, OVoxelBackend, Trellis2PBRPostprocessFacade, TrimeshBackend

    record = report.setdefault("trellis2", {})
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(convert_trellis2(work), dtype=dtype).to(device)
    ovoxel_backend = OVoxelBackend(device=device, accept_nvdiffrast_research_license=True)
    cumesh = CuMeshBackend(device=device)
    trimesh_backend = TrimeshBackend()

    with timed(record, "generate_512"):
        output = pipeline(condition, pipeline_type="512", generator=torch.Generator(device).manual_seed(0))
    ovoxel = output[0][0].to(dtype=torch.float32)
    record["cells"] = int(ovoxel.active_coordinates.shape[0])
    check(record["cells"] > 1000, "TRELLIS.2 produced almost no O-Voxel cells")

    with timed(record, "to_mesh"):
        mesh = ovoxel_backend.to_mesh(ovoxel)
    record["mesh"] = {"vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0])}
    check(bool(torch.isfinite(mesh.vertices).all()) and mesh.faces.shape[0] > 0, "O-Voxel mesh is empty or NaN")
    trimesh_backend.export_mesh(mesh.to("cpu").to_coordinate_system("right_handed_y_up"), out / "t2_mesh.glb")

    with timed(record, "cumesh"):
        repaired = cumesh.process_geometry(mesh, operation="repair")
        simplified = cumesh.process_geometry(repaired, operation="simplify", parameters={"target_faces": 100_000})
    record["cumesh"] = {
        "repaired_faces": int(repaired.faces.shape[0]),
        "simplified_faces": int(simplified.faces.shape[0]),
    }
    check(0 < simplified.faces.shape[0] <= 100_000, "CuMesh simplification did not hit the target")

    cameras = turntable(4, ovoxel.coordinate_system, image_size=512, device=device)
    frames = []
    with timed(record, "render_voxels"):
        for index in range(4):
            rendered = ovoxel_backend.render_voxels(
                ovoxel,
                extrinsics=cameras.world_to_camera[index],
                intrinsics=cameras.intrinsics[index],
                image_size=512,
                attribute="base_color",
            )
            frames.append(rendered["attr"][:3])
    frames = torch.stack(frames)
    record["render_voxels"]["nonblack_fraction"] = round(float((frames.sum(dim=1) > 0).float().mean()), 4)
    check(record["render_voxels"]["nonblack_fraction"] > 0.05, "voxel renders are blank")
    save_png(tile(frames), out / "t2_voxel_render.png")

    with timed(record, "to_glb"):
        glb = pipeline.postprocess_ovoxel(
            ovoxel,
            output_format="glb",
            pbr_postprocess=Trellis2PBRPostprocessFacade(),
            postprocess_kwargs={
                "accept_nvdiffrast_research_license": True,
                "device": device,
                "decimation_target": 300_000,
                "texture_size": 2048,
            },
        )
    glb.export(str(out / "t2_pbr.glb"))
    material = glb.visual.material
    record["to_glb"].update(
        {
            "vertices": int(glb.vertices.shape[0]),
            "faces": int(glb.faces.shape[0]),
            "material": type(material).__name__,
            "base_color_texture": None if material.baseColorTexture is None else list(material.baseColorTexture.size),
        }
    )
    check(material.baseColorTexture is not None, "PBR GLB has no baked base colour texture")
    del pipeline
    torch.cuda.empty_cache()


def stage_trellis(work: Path, out: Path, report: dict, condition, *, dtype, device) -> None:
    from diffusers_3d import GaussianSplatAsset, MeshAsset, RadianceFieldAsset, TrellisImageTo3DPipeline
    from diffusers_3d.backends import GsplatBackend, TrellisGlbPostprocessFacade, TrimeshBackend
    from diffusers_3d.backends.radiance_field import render_radiance_field

    record = report.setdefault("trellis", {})
    pipeline = TrellisImageTo3DPipeline.from_pretrained(convert_trellis(work), dtype=dtype).to(device)
    with timed(record, "generate"):
        output = pipeline(
            condition,
            formats=("gaussian", "mesh", "radiance_field"),
            generator=torch.Generator(device).manual_seed(0),
        )
    gaussians = next(asset for asset in output[0] if type(asset) is GaussianSplatAsset).to(dtype=torch.float32)
    mesh = next(asset for asset in output[0] if type(asset) is MeshAsset).to(dtype=torch.float32)
    field = next(asset for asset in output[0] if type(asset) is RadianceFieldAsset).to(dtype=torch.float32)
    record["gaussians"] = int(gaussians.means.shape[0])
    record["mesh"] = {"vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0])}
    record["radiance_field"] = {"voxels": int(field.coordinates.shape[0]), "rank": field.rank, "dim": field.dim}
    check(record["gaussians"] > 10_000 and mesh.faces.shape[0] > 1000, "TRELLIS decoders produced tiny outputs")
    check(bool(torch.isfinite(mesh.vertices).all()) and bool(torch.isfinite(field.trivec).all()), "NaN outputs")

    cameras = turntable(4, gaussians.coordinate_system, image_size=512, device=device)
    gsplat = GsplatBackend(device=device, dtype=torch.float32)
    with timed(record, "gsplat"):
        rendered = gsplat.rasterize_gaussians(gaussians, cameras)
    composite = rendered["color"] * rendered["alpha"] + (1 - rendered["alpha"])
    record["gsplat"]["alpha_mean"] = round(float(rendered["alpha"].mean()), 4)
    check(0.02 < record["gsplat"]["alpha_mean"] < 0.9, "gsplat renders are blank or fully covered")
    save_png(tile(composite), out / "t1_gsplat.png")

    small = turntable(4, field.coordinate_system, image_size=256, device=device)
    with timed(record, "render_radiance_field"):
        volume = render_radiance_field(field, small, background=(1.0, 1.0, 1.0))
    record["render_radiance_field"]["alpha_mean"] = round(float(volume["alpha"].mean()), 4)
    check(0.02 < record["render_radiance_field"]["alpha_mean"] < 0.9, "radiance-field renders are blank")
    # The field and the splats decode the same latent, so their silhouettes should mostly agree.
    splat_alpha = torch.nn.functional.interpolate(rendered["alpha"], size=256, mode="bilinear") > 0.5
    field_alpha = volume["alpha"] > 0.5
    overlap = (splat_alpha & field_alpha).sum() / (splat_alpha | field_alpha).sum().clamp(min=1)
    record["render_radiance_field"]["silhouette_iou_vs_gsplat"] = round(float(overlap), 4)
    check(float(overlap) > 0.6, "radiance-field silhouette disagrees with the Gaussian one")
    save_png(tile(volume["color"]), out / "t1_radiance_field.png")

    exportable = replace(mesh.to("cpu").to_coordinate_system("right_handed_y_up"), extras={})
    TrimeshBackend().export_mesh(exportable, out / "t1_mesh_vertex_colors.glb")

    facade = TrellisGlbPostprocessFacade()
    with timed(record, "textured_mesh"):
        textured = facade.to_textured_mesh(mesh, gaussians, device=device, texture_size=1024, num_views=100)
    record["textured_mesh"].update(
        {
            "vertices": int(textured.vertices.shape[0]),
            "faces": int(textured.faces.shape[0]),
            "observed_texel_fraction": round(textured.metadata["observed_texel_fraction"], 4),
        }
    )
    check(textured.metadata["observed_texel_fraction"] > 0.5, "most texels were never observed while baking")
    TrimeshBackend().export_mesh(textured.to("cpu").to_coordinate_system("right_handed_y_up"), out / "t1_textured.glb")
    save_png(textured.materials[0].base_color.permute(2, 0, 1), out / "t1_texture.png")
    del pipeline
    torch.cuda.empty_cache()


def stage_text(work: Path, out: Path, report: dict, *, dtype, device) -> None:
    from diffusers_3d import TextCondition, TrellisTextTo3DPipeline
    from diffusers_3d.backends import GsplatBackend

    record = report.setdefault("text", {})
    pipeline = TrellisTextTo3DPipeline.from_pretrained(convert_trellis_text(work), dtype=dtype).to(device)
    with timed(record, "generate"):
        output = pipeline(
            TextCondition(text=PROMPT), formats=("gaussian",), generator=torch.Generator(device).manual_seed(0)
        )
    gaussians = output[0][0].to(dtype=torch.float32)
    record["gaussians"] = int(gaussians.means.shape[0])
    check(record["gaussians"] > 10_000, "text-to-3D produced almost no splats")
    cameras = turntable(4, gaussians.coordinate_system, image_size=512, device=device)
    rendered = GsplatBackend(device=device, dtype=torch.float32).rasterize_gaussians(gaussians, cameras)
    record["alpha_mean"] = round(float(rendered["alpha"].mean()), 4)
    check(0.02 < record["alpha_mean"] < 0.9, "text-to-3D renders are blank")
    save_png(tile(rendered["color"] * rendered["alpha"] + (1 - rendered["alpha"])), out / "t1txt_gsplat.png")
    del pipeline
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="cache for downloads and converted checkpoints")
    parser.add_argument("--output", type=Path, required=True, help="where report.json, PNGs, and GLBs go")
    parser.add_argument("--stages", default="status,trellis2,trellis,text", help="comma-separated subset")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--image", default=EXAMPLE_IMAGE_URL, help="RGBA image path or URL")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("gpu_smoke.py needs a CUDA device", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    report: dict = {"stages": stages, "dtype": args.dtype, "errors": {}}
    condition = load_image(args.image, args.work) if {"trellis2", "trellis"} & set(stages) else None

    runners = {
        "status": lambda: stage_status(report, device),
        "trellis2": lambda: stage_trellis2(args.work, args.output, report, condition, dtype=dtype, device=device),
        "trellis": lambda: stage_trellis(args.work, args.output, report, condition, dtype=dtype, device=device),
        "text": lambda: stage_text(args.work, args.output, report, dtype=dtype, device=device),
    }
    for stage in stages:
        if stage not in runners:
            print(f"unknown stage {stage!r}; choose from {sorted(runners)}", file=sys.stderr)
            return 2
        log(f"stage {stage}")
        try:
            runners[stage]()
        except Exception as error:  # noqa: BLE001 - every stage is reported, then the exit code says it failed
            report["errors"][stage] = repr(error)
            traceback.print_exc()
        log(f"stage {stage} {'FAILED' if stage in report['errors'] else 'ok'}: {json.dumps(report.get(stage, {}))}")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"report written to {args.output / 'report.json'}; errors: {sorted(report['errors']) or 'none'}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
