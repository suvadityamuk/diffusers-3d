"""Convert the released TRELLIS / TRELLIS.2 checkpoints and publish them as diffusers-3d pipelines on the Hub.

    python scripts/publish_hub_checkpoints.py --namespace <owner> --work /tmp/publish

Publishes three model repos under ``<owner>``:

* ``TRELLIS-image-large-diffusers-3d`` (MIT TRELLIS weights plus the Apache-2.0 DINOv2 conditioner),
* ``TRELLIS-text-large-diffusers-3d`` (MIT TRELLIS weights plus the MIT CLIP text conditioner),
* ``TRELLIS.2-4B-diffusers-3d`` (MIT TRELLIS.2 weights only: the DINOv3 conditioner is gated under Meta's own
  license and is not redistributed; users pass it to ``from_pretrained``).

Each repo gets a model card with provenance and loading instructions. Needs ``HF_TOKEN`` with write access to the
namespace and read access to the gated DINOv3 weights (used locally to run the TRELLIS.2 converter). Every release
needs about twice its size under ``--work`` (source plus conversion); on hosts with little scratch space run one
release per invocation with ``--releases``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_sources import RELEASES, convert_release  # noqa: E402

REPOSITORIES = {
    "trellis-image": "TRELLIS-image-large-diffusers-3d",
    "trellis-text": "TRELLIS-text-large-diffusers-3d",
    "trellis2": "TRELLIS.2-4B-diffusers-3d",
}
TRELLIS_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
TRELLIS2_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"


def _card_header(*, pipeline_tag: str, base_model: str, extra_tags: list[str]) -> str:
    tags = "\n".join(f"- {tag}" for tag in ["diffusers-3d", "trellis", "3d", *extra_tags])
    return f"""---
license: mit
base_model: {base_model}
pipeline_tag: {pipeline_tag}
tags:
{tags}
---
"""


def _install_note() -> str:
    return """## Install

```bash
pip install git+https://github.com/suvadityamuk/diffusers.git
pip install "git+https://github.com/suvadityamuk/diffusers.git#subdirectory=packages/diffusers-3d"
```

`diffusers-3d` runs every network in plain PyTorch on CPU or GPU. Rendering Gaussian splats needs the optional
`gsplat` backend; meshing and PBR export for TRELLIS.2 need the compiled O-Voxel runtime (see the package docs).
"""


def card_trellis_image(namespace: str, version: str) -> str:
    return (
        _card_header(
            pipeline_tag="image-to-3d", base_model="microsoft/TRELLIS-image-large", extra_tags=["image-to-3d"]
        )
        + f"""
# TRELLIS-image-large for diffusers-3d

[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) converted into a
[diffusers-3d](https://github.com/suvadityamuk/diffusers) pipeline: ordinary Diffusers component folders
(`config.json` + `safetensors`) plus the `object3d_model_index.json` sidecar that the package's auto-loader
validates before downloading anything. Nothing here requires remote code.

{_install_note()}
## Usage

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("{namespace}/TRELLIS-image-large-diffusers-3d", dtype=torch.bfloat16).to("cuda")
output = pipeline(ImageCondition(image=rgba), formats=("gaussian", "mesh", "radiance_field"))
gaussians, mesh, radiance_field = output.objects
```

`rgba` is a `(4, H, W)` tensor in `[0, 1]` whose alpha channel masks the object; the pipeline crops and recentres
it as the released code does. `formats` selects any of `sparse_structure`, `slat`, `gaussian`, `mesh`, and
`radiance_field`.

## Components

| Folder | Class | Released file |
|---|---|---|
| `conditioner` | `TrellisDinov2Conditioner` | `facebook/dinov2-with-registers-large` (DINOv2 ViT-L/14 with registers) |
| `sparse_structure_flow_model` | `TrellisSparseStructureFlowModel` | `ss_flow_img_dit_L_16l8_fp16` |
| `sparse_structure_decoder` | `TrellisSparseStructureDecoder` | `ss_dec_conv3d_16l8_fp16` |
| `slat_flow_model` | `TrellisSLatFlowModel` | `slat_flow_img_dit_L_64l8p2_fp16` |
| `gaussian_decoder` | `TrellisSLatGaussianDecoder` | `slat_dec_gs_swin8_B_64l8gs32_fp16` |
| `mesh_decoder` | `TrellisSLatMeshDecoder` | `slat_dec_mesh_swin8_B_64l8m256c_fp16` |
| `radiance_field_decoder` | `TrellisSLatRadianceFieldDecoder` | `slat_dec_rf_swin8_B_64l8r16_fp16` |

Both schedulers carry the released sampler settings (25 steps, guidance 5.0 over the 0.5–1.0 interval,
`rescale_t=3`). Weights are stored as released (float16 for the transformers); load with `dtype=` to pick the
compute precision.

## Provenance

Converted with `diffusers-3d-convert-trellis` from `diffusers-3d {version}` against TRELLIS revision
`{TRELLIS_REVISION}`. The conversion renames parameters into the package layout and reformats configs; it does not
change any weight value. Tiny-configuration parity tests against the pinned upstream code are part of the package
test suite.

## License and attribution

TRELLIS weights and architecture: MIT License, Copyright (c) Microsoft Corporation. The DINOv2 conditioner weights
are Apache-2.0, Copyright (c) Meta Platforms, Inc. This repository redistributes both under those terms; it is not
affiliated with or endorsed by Microsoft or Meta.
"""
    )


def card_trellis_text(namespace: str, version: str) -> str:
    return (
        _card_header(pipeline_tag="text-to-3d", base_model="microsoft/TRELLIS-text-large", extra_tags=["text-to-3d"])
        + f"""
# TRELLIS-text-large for diffusers-3d

[microsoft/TRELLIS-text-large](https://huggingface.co/microsoft/TRELLIS-text-large) converted into a
[diffusers-3d](https://github.com/suvadityamuk/diffusers) pipeline. The text release shares its decoders with
the image release; they are included here so the repository loads on its own.

{_install_note()}
## Usage

```python
import torch
from diffusers_3d import AutoPipelineForTextTo3D

pipeline = AutoPipelineForTextTo3D.from_pretrained("{namespace}/TRELLIS-text-large-diffusers-3d", dtype=torch.bfloat16).to("cuda")
output = pipeline("a wooden rocking chair", formats=("gaussian", "mesh"))
```

Prompts may also be `TextCondition(text=..., negative_text=...)` values. Defaults follow the released text
sampler (guidance 7.5 over the 0.5–0.95 interval).

## Components

| Folder | Class | Released file |
|---|---|---|
| `conditioner` | `TrellisClipTextConditioner` | `openai/clip-vit-large-patch14` text tower and tokenizer |
| `sparse_structure_flow_model` | `TrellisSparseStructureFlowModel` | `ss_flow_txt_dit_L_16l8_fp16` |
| `sparse_structure_decoder` | `TrellisSparseStructureDecoder` | `ss_dec_conv3d_16l8_fp16` (from TRELLIS-image-large) |
| `slat_flow_model` | `TrellisSLatFlowModel` | `slat_flow_txt_dit_L_64l8p2_fp16` |
| `gaussian_decoder` | `TrellisSLatGaussianDecoder` | `slat_dec_gs_swin8_B_64l8gs32_fp16` (from TRELLIS-image-large) |
| `mesh_decoder` | `TrellisSLatMeshDecoder` | `slat_dec_mesh_swin8_B_64l8m256c_fp16` (from TRELLIS-image-large) |
| `radiance_field_decoder` | `TrellisSLatRadianceFieldDecoder` | `slat_dec_rf_swin8_B_64l8r16_fp16` (from TRELLIS-image-large) |

## Provenance

Converted with `diffusers-3d-convert-trellis` from `diffusers-3d {version}` against TRELLIS revision
`{TRELLIS_REVISION}`. Weight values are unchanged.

## License and attribution

TRELLIS weights and architecture: MIT License, Copyright (c) Microsoft Corporation. The CLIP text encoder weights
are MIT, Copyright (c) OpenAI. Not affiliated with or endorsed by Microsoft or OpenAI.
"""
    )


def card_trellis2(namespace: str, version: str) -> str:
    return (
        _card_header(
            pipeline_tag="image-to-3d", base_model="microsoft/TRELLIS.2-4B", extra_tags=["image-to-3d", "pbr"]
        )
        + f"""
# TRELLIS.2-4B for diffusers-3d

[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) converted into a
[diffusers-3d](https://github.com/suvadityamuk/diffusers) pipeline.

**The DINOv3 image conditioner is not included.** Its weights are gated under Meta's DINOv3 License, so this
repository ships every TRELLIS.2 component except `conditioner/`; you accept the license on
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and
pass the conditioner in when loading.

{_install_note()}
## Usage

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition, Trellis2Dinov3Conditioner

conditioner = Trellis2Dinov3Conditioner.from_dinov3_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "{namespace}/TRELLIS.2-4B-diffusers-3d", conditioner=conditioner, dtype=torch.bfloat16
).to("cuda")
output = pipeline(ImageCondition(image=rgba), pipeline_type="512")
ovoxel = output.objects[0]  # OVoxelAsset: dual-grid surface with PBR channels
```

`pipeline_type` selects the released preset (`512`, `1024`, `1024_cascade`, `1536_cascade`). Meshing the O-Voxel
(`OVoxelBackend.to_mesh`) and baking the PBR GLB (`pipeline.postprocess_ovoxel(..., output_format="glb")`) need
the compiled O-Voxel runtime and its research-licensed nvdiffrast dependency; see the package docs.

## Components

| Folder | Class | Released file |
|---|---|---|
| `conditioner` | `Trellis2Dinov3Conditioner` | not included (gated `facebook/dinov3-vitl16-pretrain-lvd1689m`) |
| `sparse_structure_flow_model` | `Trellis2SparseStructureFlowModel` | `ss_flow_img_dit_1_3B_64_bf16` |
| `sparse_structure_decoder` | `Trellis2SparseStructureDecoder` | `ss_dec_conv3d_16l8_fp16` (from TRELLIS-image-large) |
| `shape_slat_flow_model` | `Trellis2SLatFlowModel` | `slat_flow_img2shape_dit_1_3B_512_bf16` |
| `shape_slat_flow_model_1024` | `Trellis2SLatFlowModel` | `slat_flow_img2shape_dit_1_3B_1024_bf16` |
| `shape_slat_decoder` | `Trellis2ShapeDualGridDecoder` | `shape_dec_next_dc_f16c32_fp16` |
| `texture_slat_flow_model` | `Trellis2SLatFlowModel` | `slat_flow_imgshape2tex_dit_1_3B_512_bf16` |
| `texture_slat_flow_model_1024` | `Trellis2SLatFlowModel` | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` |
| `pbr_decoder` | `Trellis2PBRSparseDecoder` | `tex_dec_next_dc_f16c32_fp16` |

## Provenance

Converted with `diffusers-3d-convert-trellis2` from `diffusers-3d {version}` against TRELLIS.2 revision
`{TRELLIS2_REVISION}`. Weight values are unchanged.

## License and attribution

TRELLIS.2 weights and architecture: MIT License, Copyright (c) Microsoft Corporation. The sparse-structure decoder
comes from TRELLIS (MIT). No DINOv3 weights are redistributed. Not affiliated with or endorsed by Microsoft or Meta.
"""
    )


CARDS = {"trellis-image": card_trellis_image, "trellis-text": card_trellis_text, "trellis2": card_trellis2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True, help="Hub user or organization to publish under")
    parser.add_argument("--work", type=Path, required=True, help="cache for downloads and converted checkpoints")
    parser.add_argument("--releases", default=",".join(REPOSITORIES), help="comma-separated subset of releases")
    parser.add_argument("--private", action="store_true", help="create the repositories as private")
    parser.add_argument("--dry-run", action="store_true", help="convert and write the cards without uploading")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    import diffusers_3d

    api = HfApi()
    for name in [item.strip() for item in args.releases.split(",") if item.strip()]:
        if name not in REPOSITORIES:
            print(f"unknown release {name!r}; choose from {sorted(REPOSITORIES)}", file=sys.stderr)
            return 2
        repo_id = f"{args.namespace}/{REPOSITORIES[name]}"
        print(f"== {name}: converting {RELEASES[name]['source']}", flush=True)
        folder = convert_release(name, args.work)
        (folder / "README.md").write_text(CARDS[name](args.namespace, diffusers_3d.__version__))
        # The TRELLIS.2 converter needs the gated DINOv3 conditioner to run, but its weights must not be republished.
        ignore = ["conditioner/*"] if name == "trellis2" else []
        if args.dry_run:
            print(f"   dry run: would upload {folder} to {repo_id} (ignoring {ignore})")
            continue
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            ignore_patterns=ignore + ["*.pyc"],
            commit_message=f"Convert {RELEASES[name]['source']} with diffusers-3d {diffusers_3d.__version__}",
        )
        print(f"   published https://huggingface.co/{repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
