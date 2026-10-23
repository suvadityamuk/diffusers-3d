# TRELLIS image-to-3D and text-to-3D

Microsoft TRELLIS, ported from upstream revision `442aa1e1afb9014e80681d3bf604e8d728a86ee7`. A sparse-structure flow
picks occupied voxels, a structured-latent (SLAT) flow fills them, and decoders turn the SLAT into Gaussian splats, a
FlexiCubes mesh, or a radiance field. Everything runs in plain PyTorch on any device. MIT license.

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, AutoPipelineForTextTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/TRELLIS-image-large-diffusers-3d").to("cuda")
mesh = pipeline(ImageCondition(image=rgba), formats=("mesh",), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

text = AutoPipelineForTextTo3D.from_pretrained("suvadityamuk/TRELLIS-text-large-diffusers-3d").to("cuda")
gaussians = text("a wooden chair", formats=("gaussian",)).objects[0]
```

- `formats` picks from `sparse_structure`, `slat`, `gaussian`, `mesh`, and `radiance_field`. By default you get every
  loaded decoder's output, in the order gaussian, mesh, radiance field.
- Defaults: 25 steps per stage and `rescale_t=3`. The image pipeline uses guidance 5.0 over the interval (0.5, 1.0).
  The text pipeline uses 7.5 over (0.5, 0.95).
- Images need RGBA, or RGB with a `mask`. RGB without a mask is treated as an already-cropped full frame.
- Assets are in TRELLIS' right-handed Z-up frame.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2WithRegistersModel` | `facebook/dinov2-with-registers-large`; pre-norm tokens with an unparameterized layer norm, all-zero unconditional tokens |
| `text_encoder`, `tokenizer` | `transformers.CLIPTextModel`, `CLIPTokenizer` | `openai/clip-vit-large-patch14`; the empty prompt is the unconditional branch |
| `sparse_structure_flow_model` | `TrellisSparseStructureFlowModel` | dense transformer on Diffusers attention |
| `sparse_structure_decoder` | `TrellisSparseStructureDecoder` | Conv3D decoder; occupancy logits thresholded at zero |
| `slat_flow_model` | `TrellisSLatFlowModel` | sparse-conv U-Net around a dense-attention core |
| `gaussian_decoder` | `TrellisSLatGaussianDecoder` | swin sparse attention to Gaussian splats |
| `mesh_decoder` | `TrellisSLatMeshDecoder` | FlexiCubes mesh with vertex colours |
| `radiance_field_decoder` | `TrellisSLatRadianceFieldDecoder` | rank-16 Strivec radiance field |
| `*_scheduler` | `TrellisFlowEulerScheduler` | upstream's Euler flow sampler and `(1 + w) * cond - w * uncond` guidance |

## Differences from upstream

- `spconv` and `xformers` are replaced by plain PyTorch in [`sparse_ops.py`](sparse_ops.py). Pooling keeps upstream's
  `scatter_reduce(..., include_self=True)` mean, because the released weights were trained with it.
- FlexiCubes is a plain-PyTorch port of the Apache-2.0 fork TRELLIS pins ([`flexicubes.py`](flexicubes.py)). It builds
  the grid only around active voxels, so meshes match upstream up to vertex order.
- Radiance fields render with `diffusers_3d.backends.radiance_field.render_radiance_field`, a new pure-PyTorch volume
  renderer. Upstream's `diffoctreerast` is research-only licensed, so there is no pixel parity with it.
- `TrellisGlbPostprocessFacade.to_textured_mesh(mesh, gaussians)` replaces upstream `to_glb`. It uses CuMesh, xatlas,
  gsplat, and plain-PyTorch baking instead of nvdiffrast, which is research-only licensed. Hole filling and
  optimisation-based baking are not ported.
- No background removal. Upstream runs `rembg` on images without alpha.
- nvdiffrast, diffoctreerast, and mip-splatting are never imported unless you explicitly accept their research-only
  licenses.

## Conversion

```bash
diffusers-3d-convert-trellis source/ output/ --encoder-path facebook/dinov2-with-registers-large
```

`source/` holds the upstream `pipeline.json` and component `.json`/`.safetensors` pairs. Pass
`--encoder-path openai/clip-vit-large-patch14` for the text release. The converter writes every component and
`object3d_model_index.json`.

## Verification

Tiny-weight CPU tests compare every model and FlexiCubes against the pinned upstream code, with `spconv` and
`xformers` shimmed to dense PyTorch. Set `DIFFUSERS_3D_TRELLIS_REFERENCE_ROOT` to the upstream checkout to run them.
GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
