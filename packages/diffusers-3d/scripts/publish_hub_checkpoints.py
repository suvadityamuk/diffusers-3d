"""Convert released 3D checkpoints and publish them as diffusers-3d pipelines on the Hub.

    python scripts/publish_hub_checkpoints.py --namespace <owner> --work /tmp/publish

Publishes one model repo per release under ``<owner>``:

* ``TRELLIS-image-large-diffusers-3d`` (MIT TRELLIS weights plus the Apache-2.0 DINOv2 image encoder),
* ``TRELLIS-text-large-diffusers-3d`` (MIT TRELLIS weights plus the MIT CLIP text encoder and tokenizer),
* ``TRELLIS.2-4B-diffusers-3d`` (MIT TRELLIS.2 weights only: the DINOv3 image encoder is gated under Meta's own
  license and is not redistributed; users pass it to ``from_pretrained``),
* ``TripoSG-diffusers-3d``, ``Step1X-3D-Geometry-1300m-diffusers-3d``, ``Hunyuan3D-2.1-diffusers-3d``,
  ``sam-3d-objects-diffusers-3d``, and ``Pixal3D-diffusers-3d``, each with its family's license and notice files.
  Pixal3D, like TRELLIS.2, leaves out the gated DINOv3 image encoder.
* ``Step1X-3D-Texture-diffusers-3d`` and ``Hunyuan3D-2.1-Paint-diffusers-3d``, the two mesh-texturing stages, with
  the license texts of the base models they redistribute.

Each repo gets a model card with provenance and loading instructions. Needs ``HF_TOKEN`` with write access to the
namespace and read access to the gated source weights (used locally to run the converters). Every release
needs about twice its size under ``--work`` (source plus conversion); on hosts with little scratch space run one
release per invocation with ``--releases``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_sources import RELEASES, convert_release  # noqa: E402

REPOSITORIES = {
    "trellis-image": "TRELLIS-image-large-diffusers-3d",
    "trellis-text": "TRELLIS-text-large-diffusers-3d",
    "trellis2": "TRELLIS.2-4B-diffusers-3d",
    "triposg": "TripoSG-diffusers-3d",
    "step1x3d": "Step1X-3D-Geometry-1300m-diffusers-3d",
    "hunyuan3d": "Hunyuan3D-2.1-diffusers-3d",
    "sam3d-objects": "sam-3d-objects-diffusers-3d",
    "pixal3d": "Pixal3D-diffusers-3d",
    "step1x3d-texture": "Step1X-3D-Texture-diffusers-3d",
    "hunyuan3d-paint": "Hunyuan3D-2.1-Paint-diffusers-3d",
}
# Families whose license and notice files are copied into the published repo.
FAMILY_PACKAGES = {
    "triposg": "triposg",
    "step1x3d": "step1x3d",
    "hunyuan3d": "hunyuan3d",
    "sam3d-objects": "sam3d_objects",
    "pixal3d": "pixal3d",
    "step1x3d-texture": "step1x3d",
    "hunyuan3d-paint": "hunyuan3d",
}
# License texts of redistributed third-party weights that the family directories do not carry. Stability AI no longer
# hosts Stable Diffusion 2.x, so the CreativeML Open RAIL++-M text comes from the SDXL release.
EXTRA_LICENSES = {
    "hunyuan3d-paint": {
        "LICENSE-OPENRAIL-PLUSPLUS-M.md": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/"
        "462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md",
        "LICENSE-REAL-ESRGAN": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/"
        "a4abfb2979a7bbff3f69f58f58ae324608821e27/LICENSE",
    },
}
# Gated encoders the converters need but that must not be republished.
WITHHELD = {"trellis2": ["image_encoder/*"], "pixal3d": ["image_encoder/*"]}
TRELLIS_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
TRELLIS2_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"


def _card_header(
    *,
    pipeline_tag: str,
    base_model: str,
    extra_tags: list[str],
    family: str = "trellis",
    license: str = "license: mit",
) -> str:
    tags = "\n".join(f"- {tag}" for tag in ["diffusers-3d", family, "3d", *extra_tags])
    return f"""---
{license}
base_model: {base_model}
pipeline_tag: {pipeline_tag}
tags:
{tags}
---
"""


def _install_note() -> str:
    return """## Install

```bash
pip install git+https://github.com/suvadityamuk/diffusers-3d.git
pip install "git+https://github.com/suvadityamuk/diffusers-3d.git#subdirectory=packages/diffusers-3d"
```

`diffusers-3d` runs every network in plain PyTorch on CPU or GPU. Rendering Gaussian splats needs the optional
`gsplat` backend; the PBR-textured mesh for TRELLIS.2 needs CuMesh (see the package docs).
"""


def card_trellis_image(namespace: str, version: str) -> str:
    return (
        _card_header(
            pipeline_tag="image-to-3d", base_model="microsoft/TRELLIS-image-large", extra_tags=["image-to-3d"]
        )
        + f"""
# TRELLIS-image-large for diffusers-3d

[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) converted into a
[diffusers-3d](https://github.com/suvadityamuk/diffusers-3d) pipeline: ordinary Diffusers component folders
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
| `image_encoder` | `transformers.Dinov2WithRegistersModel` | `facebook/dinov2-with-registers-large` (DINOv2 ViT-L/14 with registers) |
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

TRELLIS weights and architecture: MIT License, Copyright (c) Microsoft Corporation. The DINOv2 image encoder weights
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
[diffusers-3d](https://github.com/suvadityamuk/diffusers-3d) pipeline. The text release shares its decoders with
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
| `text_encoder` | `transformers.CLIPTextModel` | `openai/clip-vit-large-patch14` text tower |
| `tokenizer` | `transformers.CLIPTokenizer` | `openai/clip-vit-large-patch14` tokenizer |
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
[diffusers-3d](https://github.com/suvadityamuk/diffusers-3d) pipeline.

**The DINOv3 image encoder is not included.** Its weights are gated under Meta's DINOv3 License, so this
repository ships every TRELLIS.2 component except `image_encoder/`; you accept the license on
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and
pass the image encoder in when loading.

{_install_note()}
## Usage

```python
import torch
from transformers import DINOv3ViTModel

from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "{namespace}/TRELLIS.2-4B-diffusers-3d", image_encoder=image_encoder, dtype=torch.bfloat16
).to("cuda")
output = pipeline(ImageCondition(image=rgba), pipeline_type="512")
ovoxel = output.objects[0]  # OVoxelAsset: dual-grid surface with PBR channels
```

`pipeline_type` selects the released preset (`512`, `1024`, `1024_cascade`, `1536_cascade`). `formats=("mesh",)`
returns a vertex-coloured mesh on any device; baking the PBR-textured mesh
(`pipeline.postprocess_ovoxel(..., output_format="textured_mesh")`) needs CuMesh, which bundles cubvh under NVIDIA's
non-commercial license; see the package docs.

## Components

| Folder | Class | Released file |
|---|---|---|
| `image_encoder` | `transformers.DINOv3ViTModel` | not included (gated `facebook/dinov3-vitl16-pretrain-lvd1689m`) |
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


_RGBA_NOTE = (
    "`rgba` is a `(4, H, W)` tensor in `[0, 1]` whose alpha channel masks the object; the pipeline crops and pads it "
    "as the released code does."
)


def _components(folder: Path, withheld: tuple[str, ...] = ()) -> str:
    index = json.loads((folder / "model_index.json").read_text())
    rows = ["| Folder | Class |", "|---|---|"]
    for name, value in sorted(index.items()):
        if isinstance(value, list) and len(value) == 2 and value[0]:
            note = " (not included, pass it to `from_pretrained`)" if name in withheld else ""
            rows.append(f"| `{name}` | `{value[0]}.{value[1]}`{note} |")
    return "\n".join(rows)


def _family_card(
    *,
    namespace: str,
    version: str,
    folder: Path,
    repo: str,
    title: str,
    source: str,
    family: str,
    license: str,
    converter: str,
    usage: str,
    notes: str,
    license_text: str,
    withheld: tuple[str, ...] = (),
    extra_tags: tuple[str, ...] = ("image-to-3d",),
) -> str:
    return (
        _card_header(
            pipeline_tag="image-to-3d", base_model=source, extra_tags=list(extra_tags), family=family, license=license
        )
        + f"""
# {title} for diffusers-3d

[{source}](https://huggingface.co/{source}) converted into a [diffusers-3d](https://github.com/suvadityamuk/diffusers-3d)
pipeline: ordinary Diffusers and Transformers component folders plus the `object3d_model_index.json` sidecar that
the package's auto-loader validates before downloading anything. Nothing here requires remote code.

{_install_note()}
## Usage

```python
{usage.format(repo=f"{namespace}/{repo}")}
```

{notes}

## Components

{_components(folder, withheld)}

## Provenance

Converted with `{converter}` from `diffusers-3d {version}`; the package's family README lists the pinned upstream
revisions and how the conversion maps each tensor. Parity tests against the pinned upstream code are part of the
package test suite.

## License and attribution

{license_text}
"""
    )


def card_triposg(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["triposg"],
        title="TripoSG",
        source="VAST-AI/TripoSG",
        family="triposg",
        license="license: mit",
        converter="diffusers-3d-convert-triposg",
        usage="""import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("{repo}").to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]""",
        notes=_RGBA_NOTE + " The result is an untextured `MeshAsset` extracted with marching cubes.",
        license_text="""TripoSG weights and code: MIT License, Copyright (c) VAST-AI-Research. The DINOv2 image encoder
weights are Apache-2.0, Copyright (c) Meta Platforms, Inc. TripoSG's transformer is derived from Tencent HunyuanDiT,
whose Tencent Hunyuan Community License does not apply in the European Union, the United Kingdom, or South Korea;
you are responsible for making sure your use complies with it. See `LICENSE-MIT` and `NOTICE`. Not affiliated with
or endorsed by VAST, Tencent, or Meta.""",
    )


def card_step1x3d(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["step1x3d"],
        title="Step1X-3D Geometry 1300m",
        source="stepfun-ai/Step1X-3D",
        family="step1x3d",
        license="license: apache-2.0",
        converter="diffusers-3d-convert-step1x3d",
        usage="""import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("{repo}").to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]""",
        notes=_RGBA_NOTE + " This is the geometry stage (`Step1X-3D-Geometry-1300m`); the result is an untextured "
        "`MeshAsset`.",
        license_text="""Step1X-3D weights and code: Apache-2.0, Copyright (c) StepFun. The DINOv2 image encoder weights
are Apache-2.0, Copyright (c) Meta Platforms, Inc. See `LICENSE-APACHE-2.0` and `NOTICE`. Not affiliated with or
endorsed by StepFun or Meta.""",
    )


def card_hunyuan3d(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["hunyuan3d"],
        title="Hunyuan3D-2.1 (shape)",
        source="tencent/Hunyuan3D-2.1",
        family="hunyuan3d",
        license="license: other\nlicense_name: tencent-hunyuan-3d-2-1-community\nlicense_link: "
        f"https://huggingface.co/{namespace}/{REPOSITORIES['hunyuan3d']}/blob/main/LICENSE-TENCENT-HUNYUAN-3D-2.1",
        converter="diffusers-3d-convert-hunyuan3d",
        usage="""import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("{repo}", dtype=torch.float16).to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]""",
        notes=_RGBA_NOTE + " This is the shape stage (`hunyuan3d-dit-v2-1`); the result is an untextured `MeshAsset`.",
        license_text="""**Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement,
Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of "Tencent Hunyuan" are owned by Tencent or its
affiliate.**

This repository redistributes modified Tencent Hunyuan 3D 2.1 works: diffusers-3d converted the released weights
into Diffusers component folders (parameters renamed and the ShapeVAE point-cloud encoder dropped; weight values
unchanged). The license, including its Acceptable Use Policy and the use restrictions of its Sections 5(a) and
5(b), applies to everyone who uses these files; read `LICENSE-TENCENT-HUNYUAN-3D-2.1` and `NOTICE` before use.
The license **does not apply in the European Union, the United Kingdom, or South Korea**, and products above
1 million monthly active users need a separate license from Tencent. This repository and diffusers-3d are not
affiliated with, sponsored, or endorsed by Tencent.""",
    )


def card_sam3d(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["sam3d-objects"],
        title="SAM 3D Objects",
        source="facebook/sam-3d-objects",
        family="sam3d-objects",
        license="license: other\nlicense_name: sam-license\nlicense_link: "
        f"https://huggingface.co/{namespace}/{REPOSITORIES['sam3d-objects']}/blob/main/LICENSE-SAM",
        converter="diffusers-3d-convert-sam3d",
        usage="""import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("{repo}").to("cuda")
output = pipeline(ImageCondition(image=rgb, mask=mask), formats=("gaussian", "mesh"))
gaussians, mesh = output.objects""",
        notes="`rgb` is a `(3, H, W)` tensor in `[0, 1]` and `mask` a `(1, H, W)` tensor selecting the object. The "
        "object comes back placed in the input camera's frame. Pass `pointmap=` (camera-space points per pixel) to "
        "use depth; without it the pointmap tokens are zero, as in the model's training dropout.",
        license_text="""SAM 3D Objects weights and code: SAM License, Copyright (c) Meta Platforms, Inc. and affiliates.
This repository redistributes converted SAM Materials under that Agreement, a copy of which is in `LICENSE-SAM`;
by using these files you agree to it. It is not an open-source license: it forbids military, warfare, nuclear,
espionage, and weapons uses and use by parties subject to trade controls, and publications using the model must
acknowledge it. Not affiliated with or endorsed by Meta.""",
    )


def card_pixal3d(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["pixal3d"],
        title="Pixal3D",
        source="TencentARC/Pixal3D",
        family="pixal3d",
        license="license: mit",
        converter="diffusers-3d-convert-pixal3d",
        usage="""import torch
from transformers import DINOv3ViTModel

from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

image_encoder = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
pipeline = AutoPipelineForImageTo3D.from_pretrained(
    "{repo}", image_encoder=image_encoder, dtype=torch.bfloat16
).to("cuda")
output = pipeline(ImageCondition(image=rgba), camera_angle_x=0.35, formats=("o_voxel", "mesh"))""",
        notes="**The DINOv3 image encoder is not included.** Its weights are gated under Meta's DINOv3 License; "
        "accept it on [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-"
        "pretrain-lvd1689m) and pass the encoder in when loading. " + _RGBA_NOTE + " `camera_angle_x` is the input "
        "camera's horizontal field of view in radians; the shape is reconstructed in that camera's frame.",
        license_text="""Pixal3D weights and code: MIT License, Copyright (c) Tencent ARC Lab. The `feature_upsampler` is
valeo.ai's NAF (Neighborhood Attention Filtering) upsampler, converted from the release checkpoint at
https://github.com/valeoai/NAF/releases (NAF code is Apache-2.0; the checkpoint is published there without a
separate license). No DINOv3 weights are redistributed. See `LICENSE-MIT` and `NOTICE`. Not affiliated with or
endorsed by Tencent, valeo.ai, or Meta.""",
        withheld=("image_encoder",),
    )


_TEXTURE_NOTE = (
    "`mesh` is any `MeshAsset`, for example a shape pipeline's output, and `rgba` a `(4, H, W)` tensor in `[0, 1]` "
    "whose alpha masks the object. The result is a UV-textured `MeshAsset`; export it with "
    '`TrimeshBackend().export_mesh(textured.to("cpu"), "textured.glb")` (needs the `portable` extra).'
)


def card_step1x3d_texture(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["step1x3d-texture"],
        title="Step1X-3D Texture",
        source="stepfun-ai/Step1X-3D",
        family="step1x3d",
        license="license: openrail++",
        converter="diffusers-3d-convert-step1x3d-texture",
        usage="""import torch
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition

shape = AutoPipelineForImageTo3D.from_pretrained("{namespace}/Step1X-3D-Geometry-1300m-diffusers-3d").to("cuda")
mesh = shape(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

texture = AutoPipelineFor3D.from_pretrained("{repo}", dtype=torch.float16).to("cuda")
textured = texture(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]""".replace(
            "{namespace}", namespace
        ),
        notes=_TEXTURE_NOTE + " The pipeline generates six orthographic views with SDXL, Step1X-3D's multi-view "
        "adapter, and a T2I-Adapter fed position and normal renders, keeps them consistent through a shared latent "
        "texture, and bakes a base-colour texture in plain PyTorch. `output.previews` holds the six views.",
        license_text="""This repository combines three sets of weights, each under its own license:

- the SDXL base UNet, text encoders, and tokenizers from
  [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), under
  the CreativeML Open RAIL++-M License in `LICENSE-SDXL.md`, including its use-based restrictions, which apply to
  everyone who uses these files;
- the VAE from [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) (MIT);
- Step1X-3D's T2I-Adapter and multi-view adapter weights from `Step1X-3D-Texture` (Apache-2.0, Copyright (c)
  StepFun; see `LICENSE-APACHE-2.0` and `NOTICE`).

The texture synchronisation follows SyncMVD (MIT, `LICENSE-MIT`). Not affiliated with or endorsed by StepFun or
Stability AI.""",
        extra_tags=("image-to-3d", "texturing"),
    )


def card_hunyuan3d_paint(namespace: str, version: str, folder: Path) -> str:
    return _family_card(
        namespace=namespace,
        version=version,
        folder=folder,
        repo=REPOSITORIES["hunyuan3d-paint"],
        title="Hunyuan3D-2.1 Paint",
        source="tencent/Hunyuan3D-2.1",
        family="hunyuan3d",
        license="license: other\nlicense_name: tencent-hunyuan-3d-2-1-community\nlicense_link: "
        f"https://huggingface.co/{namespace}/{REPOSITORIES['hunyuan3d-paint']}/blob/main/LICENSE-TENCENT-HUNYUAN-3D-2.1",
        converter="diffusers-3d-convert-hunyuan3d-paint",
        usage="""import torch
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition

shape = AutoPipelineForImageTo3D.from_pretrained("{namespace}/Hunyuan3D-2.1-diffusers-3d", dtype=torch.float16).to("cuda")
mesh = shape(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

paint = AutoPipelineFor3D.from_pretrained("{repo}", dtype=torch.float16).to("cuda")
textured = paint(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]""".replace(
            "{namespace}", namespace
        ),
        notes=_TEXTURE_NOTE + " Paint simplifies the mesh to 40k faces (`max_faces=None` keeps it; simplifying needs "
        "`fast-simplification`), unwraps it with xatlas, generates albedo and metallic-roughness views, upscales "
        "them with Real-ESRGAN, and bakes a base-colour plus metallic-roughness `PBRMaterial` in plain PyTorch.",
        license_text="""**Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement,
Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of "Tencent Hunyuan" are owned by Tencent or its
affiliate.**

This repository redistributes modified Tencent Hunyuan 3D 2.1 works: diffusers-3d converted the released
`hunyuan3d-paintpbr-v2-1` weights into Diffusers component folders (parameters renamed into stock modules, the unused
CLIP encoders dropped; weight values unchanged). The license, including its Acceptable Use Policy and the use
restrictions of its Sections 5(a) and 5(b), applies to everyone who uses these files; read
`LICENSE-TENCENT-HUNYUAN-3D-2.1` and `NOTICE` before use. The license **does not apply in the European Union, the
United Kingdom, or South Korea**, and products above 1 million monthly active users need a separate license from
Tencent.

The Paint UNets and VAE derive from Stable Diffusion 2.1-base and are also subject to the CreativeML Open RAIL++-M
License and its use-based restrictions; `LICENSE-OPENRAIL-PLUSPLUS-M.md` is that license's text as published with
SDXL, since Stability AI no longer hosts Stable Diffusion 2.x. The image encoder is
[facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant) (Apache-2.0). The upscaler holds the
RealESRGAN_x4plus weights of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), BSD-3-Clause, Copyright (c) 2021
Xintao Wang (`LICENSE-REAL-ESRGAN`). This repository and diffusers-3d are not affiliated with, sponsored, or
endorsed by Tencent, Stability AI, Meta, or the Real-ESRGAN authors.""",
        extra_tags=("image-to-3d", "texturing", "pbr"),
    )


def _with_folder(card):
    return lambda namespace, version, folder: card(namespace, version)


CARDS = {
    "trellis-image": _with_folder(card_trellis_image),
    "trellis-text": _with_folder(card_trellis_text),
    "trellis2": _with_folder(card_trellis2),
    "triposg": card_triposg,
    "step1x3d": card_step1x3d,
    "hunyuan3d": card_hunyuan3d,
    "sam3d-objects": card_sam3d,
    "pixal3d": card_pixal3d,
    "step1x3d-texture": card_step1x3d_texture,
    "hunyuan3d-paint": card_hunyuan3d_paint,
}


def copy_license_files(name: str, folder: Path) -> None:
    if name not in FAMILY_PACKAGES:
        return
    package = Path(importlib.import_module(f"diffusers_3d.families.{FAMILY_PACKAGES[name]}").__file__).parent
    for path in [*package.glob("LICENSE*"), package / "NOTICE"]:
        shutil.copyfile(path, folder / path.name)
    for filename, url in EXTRA_LICENSES.get(name, {}).items():
        urllib.request.urlretrieve(url, folder / filename)


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
        (folder / "README.md").write_text(CARDS[name](args.namespace, diffusers_3d.__version__, folder))
        copy_license_files(name, folder)
        ignore = WITHHELD.get(name, []) + ["conditioner/*"]
        if args.dry_run:
            print(f"   dry run: would upload {folder} to {repo_id} (ignoring {ignore})")
            continue
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            ignore_patterns=ignore + ["*.pyc"],
            # Repos published before the stock Transformers encoders still hold a `conditioner/` folder.
            delete_patterns=["conditioner/*"],
            commit_message=f"Convert {RELEASES[name]['source']} with diffusers-3d {diffusers_3d.__version__}",
        )
        print(f"   published https://huggingface.co/{repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
