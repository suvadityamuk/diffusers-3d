# Step1X-3D geometry and texture

StepFun Step1X-3D, ported from upstream revision `cb5ac944709c6c913109070c7b90c3447f57f3d4`. The geometry stage
(`Step1X-3D-Geometry-1300m`) turns one image into an untextured mesh. The texture stage paints a mesh from a
reference image with MV-Adapter on SDXL. Everything runs in plain PyTorch on any device. Apache-2.0 license, plus MIT
for the texture synchronisation code. The geometry stage's hierarchical grid decoder, which upstream copied from
Hunyuan3D-2.1, is under the [Tencent Hunyuan 3D 2.1 Community License](../hunyuan3d/LICENSE-TENCENT-HUNYUAN-3D-2.1).

```python
import torch
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition, TrimeshBackend

geometry = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Step1X-3D-Geometry-1300m-diffusers-3d").to("cuda")
mesh = geometry(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

# Upstream decimates to 200k faces before texturing.
mesh = mesh.to("cpu").simplify(200_000)
texture = AutoPipelineFor3D.from_pretrained("suvadityamuk/Step1X-3D-Texture-diffusers-3d", dtype=torch.float16).to("cuda")
textured = texture(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

- The foreground comes from alpha or a `mask`. A fully opaque image is used as a full-frame crop.
- Geometry defaults: 50 steps, guidance 7.5, bounds 1.05, octree resolution 384 on the released hierarchical grid.
  Meshes are right-handed Y-up. Marching cubes needs `pip install "diffusers-3d[portable]"`.
- Geometry meshes then get the released clean-up, in upstream's order and with its PLY round trips: trimesh merges
  vertices and fixes the winding, pymeshlab removes floaters (components under 0.1% of the faces) and decimates to
  `max_facenum=200000` faces, and trimesh smooth-shades the result. The pymeshlab steps need
  `pip install "diffusers-3d[meshlab]"` (GPL); `do_remove_floater=False, do_reduce_face=False` skips them, and
  `do_shade_smooth=False` keeps the merged mesh.
- The Tencent license does not apply in the EU, the UK, or South Korea; running the hierarchical decoder warns
  about it.
- Texture defaults follow `inference.py`: the mesh is read as its GLB export holds it, cleaned and decimated to
  `max_faces=50000` with pymeshlab (`remove_degenerate_face` and `reduce_face`; `None` skips both, and
  `pip install "diffusers-3d[meshlab]"` provides it), then 50 steps at guidance 3.0 over six 768 px views and a
  2048 px base-colour texture. The output is the mesh upstream exports: right-handed Y-up, scaled to `max |x| = 0.5`,
  with the final xatlas UVs and the 8-bit texture.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2WithRegistersModel` | final layer-normed class and patch tokens, registers dropped |
| `transformer` | `Step1X3DTransformer1DModel` | Flux-style: 12 dual-stream (stock `JointTransformerBlock`) + 24 single-stream blocks |
| `vae` | `Step1X3DShapeDecoder` | decoder half of the shape VAE |
| `scheduler` | `Step1X3DFlowMatchEulerScheduler` | stock `FlowMatchEulerDiscreteScheduler`, subclassed under the name the published repo records |

Texture stage:

| Component | Class | Notes |
| --- | --- | --- |
| `text_encoder`, `text_encoder_2`, `unet` | stock SDXL classes | load from `stabilityai/stable-diffusion-xl-base-1.0` |
| `vae` | `diffusers.AutoencoderKL` | loads from `madebyollin/sdxl-vae-fp16-fix` |
| `tokenizer`, `tokenizer_2` | `transformers.CLIPTokenizer` | SDXL's, saved with the pipeline |
| `adapter` | `diffusers.T2IAdapter` | conditions on position and normal renders |
| `mv_adapter` | `Step1X3DMultiViewAdapter` | one multi-view attention processor per UNet self-attention layer |
| `scheduler` | `diffusers.DDPMScheduler` | released shifted-SNR betas stored as `trained_betas` |

The release changes none of the SDXL networks (its weights file holds only the two adapters), so the converted
folder holds only the adapters, the tokenizers, and the scheduler; the SDXL networks and the VAE load from their
original repositories at pinned revisions (`texture_components.py`). Pass them to `from_pretrained` to use your own.

## Differences from upstream

Geometry:

- Inputs without usable alpha are cut out by default with rembg (u2net) at full size, as upstream does, and texture
  reference images without an alpha channel with BiRefNet (`ZhengPeng7/BiRefNet`, MIT), as upstream's texture stage
  does; `pipeline.disable_background_removal()` turns either off.
- A grid that never crosses `mc_level` raises instead of returning `None`.
- Not ported: the label-conditioned variant, the point encoder (training only), and DMC extraction (needs CUDA `diso`;
  not the default).

Texture:

- Exact against the pinned code: the jump-flooding fill (the CuPy Voronoi kernel ported to plain PyTorch; equal to
  the CuPy kernel on GPU), `meshVerticeInpaint` (NumPy 1.x arithmetic) and `cv2.inpaint`, the texture export's two
  8-bit truncations, the three xatlas unwraps (the bake, the texture synchronisation with `max_iterations=4`, and the
  output), and the RNG draw order: the latents, the reference posterior, and the scheduler noise come from
  `generator`; the noise texture, the background noise, and the colour posteriors from a generator seeded with
  `torch.initial_seed()`, which is the state of upstream's (never seeded) global RNG in a fresh process; the
  background colours from `random.Random(generator.initial_seed())`, where upstream uses Python's unseeded `random`. The VAE is upcast for the colour latents and the decode as upstream's `upcast_vae` does.
- Version- and platform-dependent: pymeshlab's quadric decimation differs between 2023.12.post3 (upstream's pin) and
  2025.7, and xatlas gives different atlases on arm64 than on x86-64.
- Rasterization: on CUDA the final bake runs `custom_rasterizer` and the control renders run nvdiffrast, as upstream
  does (both optional). The texture synchronisation still uses the package rasterizer where upstream uses pytorch3d:
  on a 27k-face mesh it picks a different face on 145 of 29k covered pixels at 96 px (3 coverage differences) and
  1,364 of 838k at 512 px (55). On CUDA the control images match upstream's within 1e-3 on all but 5 of 1.9
  million pixels (max 0.01; the view matrices differ by up to 1e-6); off CUDA they differ on 0.4% of the pixels.
- `cv2.inpaint` and the vertex inpainting run on the CPU.

## Conversion

```bash
diffusers-3d-convert-step1x3d Step1X-3D-Geometry-1300m/ output/
diffusers-3d-convert-step1x3d-texture Step1X-3D-Texture/ output/
```

The geometry converter splits the fused attention projections and drops the point encoder. It rejects missing or
unexpected keys. The texture converter writes the adapters from `step1x-3d-ig2v.safetensors`, the SDXL tokenizers,
and the scheduler; the SDXL networks and the VAE enter `model_index.json` only. It copies the SDXL license as
`LICENSE-SDXL.md`.

## Verification

Tiny-weight CPU tests check the transformer, VAE, the hierarchical grid decoder (float32 and float16), marching cubes
with the trimesh steps, the pymeshlab clean-up (floaters, decimation, smooth shading), image encoder, and a 4-step
guided pipeline with its clean-up bit for bit against the pinned upstream code.
Texture tests cover the attention processors, betas, renders, texture sync, and bake. Set
`DIFFUSERS_3D_STEP1X3D_REFERENCE_ROOT` to the upstream checkout to run them. GPU runs are in
[docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0), [LICENSE-MIT](LICENSE-MIT), and [NOTICE](NOTICE). The SDXL weights
carry the CreativeML Open RAIL++-M License.
