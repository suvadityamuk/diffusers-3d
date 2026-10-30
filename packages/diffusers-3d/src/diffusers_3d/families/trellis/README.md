# TRELLIS image-to-3D and text-to-3D

Microsoft TRELLIS, ported from upstream revision `442aa1e1afb9014e80681d3bf604e8d728a86ee7`. A sparse-structure flow
picks occupied voxels, a structured-latent (SLAT) flow fills them, and decoders turn the SLAT into Gaussian splats, a
FlexiCubes mesh, or a radiance field. The networks run in plain PyTorch on any device; rendering and GLB export use
upstream's optional CUDA libraries where it does. MIT license.

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
- DINOv2 comes from `facebook/dinov2-with-registers-large` instead of `torch.hub`. Its weights equal
  `dinov2_vitl14_reg` tensor for tensor, and the conditioning tokens match upstream exactly.
- Noise is drawn in float32 from `generator` in upstream's order, so the same seed gives the same noise, and a
  generator list draws per item. Without a generator the pipeline does what upstream's `run` does with its default
  seed: `torch.manual_seed(42)`, then the CPU global RNG.
- `encode_conditioning` bilinearly resizes tensors that are not 518×518; upstream would feed them at native size.
  `preprocess` always produces 518×518.
- `diffusers_3d.backends.radiance_field.render_radiance_field` renders radiance fields. CUDA assets go through
  upstream's diffoctreerast (an optional, research-only-licensed dependency) with upstream's exact arguments and
  `ssaa=4`, bitwise identical to upstream for the same seed. Other devices use a plain-PyTorch ray marcher with the
  same early termination, culling, per-pixel random phase and depth, within about 96-105 dB PSNR of diffoctreerast.
- `textured_mesh_from_splats(mesh, gaussians)` is upstream `to_glb`, run on the same optional CUDA libraries: pyvista
  decimation, hole filling with nvdiffrast visibility, an igraph min-cut and pymeshfix, xatlas, mip-splatting renders,
  an nvdiffrast optimisation bake, and OpenCV inpainting. Like upstream it picks views with NumPy's global RNG.
- `save` writes Gaussians and meshes the way upstream's `save_ply` and `to_glb` do, through
  `metadata["export_rotation"]`.
- Inputs without usable alpha are cut out by default with rembg (u2net) after shrinking to at most 1024 px, as
  upstream does; `pipeline.disable_background_removal()` turns it off.
- Not ported: multi-image conditioning (`run_multi_image`) and the text pipeline's `run_variant`.

## Conversion

```bash
diffusers-3d-convert-trellis source/ output/
```

`source/` holds the upstream `pipeline.json` and component `.json`/`.safetensors` pairs. The converter writes every
TRELLIS network. It does not write the encoders: `model_index.json` names them, and the pipeline loads them from their
own repositories at pinned revisions, `facebook/dinov2-with-registers-large` for the image release and
`openai/clip-vit-large-patch14` (with its tokenizer) for the text release. Pass `image_encoder=...` (or
`text_encoder=...`, `tokenizer=...`) to `from_pretrained` to use your own copy.

## Verification

Tiny-weight CPU tests compare every model and FlexiCubes against the pinned upstream code, with `spconv` and
`xformers` shimmed to dense PyTorch. Set `DIFFUSERS_3D_TRELLIS_REFERENCE_ROOT` to the upstream checkout to run them.
They also check that the seeded noise matches upstream's draws and that CLIP prompt encoding matches upstream.

The DINOv2 test checks the Hub encoder against the `torch.hub` model upstream loads. The weights match tensor for
tensor, and the conditioning tokens at 518×518 match exactly. It needs a `facebookresearch/dinov2` checkout
(`DIFFUSERS_3D_DINOV2_REFERENCE_ROOT`) and the downloaded `dinov2_vitl14_reg4_pretrain.pth`
(`DIFFUSERS_3D_DINOV2_REFERENCE_WEIGHTS`).
GPU runs are in [docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-MIT](LICENSE-MIT) and [NOTICE](NOTICE).
