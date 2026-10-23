# Step1X-3D geometry and texture

StepFun Step1X-3D, ported from upstream revision `cb5ac944709c6c913109070c7b90c3447f57f3d4`. The geometry stage
(`Step1X-3D-Geometry-1300m`) turns one image into an untextured mesh. The texture stage paints a mesh from a
reference image with MV-Adapter on SDXL. Everything runs in plain PyTorch on any device. Apache-2.0 license, plus MIT
for the texture synchronisation code.

```python
import torch
from diffusers_3d import AutoPipelineFor3D, AutoPipelineForImageTo3D, ImageCondition, TrimeshBackend

geometry = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/Step1X-3D-Geometry-1300m-diffusers-3d").to("cuda")
mesh = geometry(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]

# Upstream decimates to 200k faces before texturing.
mesh = TrimeshBackend().process_geometry(mesh, operation="simplify_quadric_decimation", parameters={"face_count": 200_000})
texture = AutoPipelineFor3D.from_pretrained("suvadityamuk/Step1X-3D-Texture-diffusers-3d", dtype=torch.float16).to("cuda")
textured = texture(mesh, ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
```

- The foreground comes from alpha or a `mask`. A fully opaque image is used as a full-frame crop.
- Geometry defaults: 50 steps, guidance 7.5, bounds 1.05, octree resolution 384. Meshes are right-handed Y-up.
  Marching cubes needs scikit-image (`pip install "diffusers-3d[portable]"`).
- Texture defaults: 50 steps, guidance 3.0, six 768 px views, a 2048 px base-colour texture. The output keeps the
  input mesh's scale and frame, with xatlas UVs.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2WithRegistersModel` | final layer-normed class and patch tokens, registers dropped |
| `transformer` | `Step1X3DTransformer1DModel` | Flux-style: 12 dual-stream (stock `JointTransformerBlock`) + 24 single-stream blocks |
| `vae` | `Step1X3DShapeDecoder` | decoder half of the shape VAE |
| `scheduler` | `Step1X3DFlowMatchEulerScheduler` | stock `FlowMatchEulerDiscreteScheduler`, subclassed only for the registry |

Texture stage:

| Component | Class | Notes |
| --- | --- | --- |
| `vae`, `text_encoder`, `text_encoder_2`, `tokenizer`, `tokenizer_2`, `unet` | stock SDXL classes | SDXL base with the fp16-fix VAE |
| `adapter` | `diffusers.T2IAdapter` | conditions on position and normal renders |
| `mv_adapter` | `Step1X3DMultiViewAdapter` | one multi-view attention processor per UNet self-attention layer |
| `scheduler` | `diffusers.DDPMScheduler` | released shifted-SNR betas stored as `trained_betas` |

## Differences from upstream

Geometry:

- The occupancy grid is evaluated densely. Upstream's hierarchical decoder comes from Hunyuan3D under a non-commercial
  license header, so it is not ported. Values match where upstream evaluates; dense evaluation costs more queries.
- Query points stay in float32 under half precision, because half precision cannot resolve a 384³ grid. Upstream
  differs here, so low-precision runs are not identical.
- Meshes are returned raw. Upstream's floater removal and pymeshlab decimation (GPL) are not applied.
- No background removal. Upstream runs `rembg` on opaque images.
- Not ported: the label-conditioned variant, the point encoder (training only), and DMC extraction (needs CUDA `diso`).

Texture:

- The package rasterizer replaces nvdiffrast and pytorch3d. Baking averages pixels into texels directly.
- Uncovered texels are filled by jump flooding. Upstream's CuPy Voronoi kernel has no license, and `cv2.inpaint` is
  not used.
- One xatlas atlas serves the whole run; upstream unwraps three times.
- All noise comes from the pipeline `generator`. Upstream also draws from the global RNGs.

## Conversion

```bash
diffusers-3d-convert-step1x3d Step1X-3D-Geometry-1300m/ output/
diffusers-3d-convert-step1x3d-texture Step1X-3D-Texture/ output/
```

The geometry converter splits the fused attention projections and drops the point encoder. It rejects missing or
unexpected keys. The texture converter loads the SDXL base and fp16-fix VAE at pinned revisions, plus the adapters
from `step1x-3d-ig2v.safetensors`. It copies the SDXL license next to the weights as `LICENSE-SDXL.md`.

## Verification

Tiny-weight CPU tests check the transformer, VAE, marching cubes, image encoder, and a 4-step guided pipeline bit for
bit against the pinned upstream code. Texture tests cover the attention processors, betas, renders, texture sync, and
bake. Set `DIFFUSERS_3D_STEP1X3D_REFERENCE_ROOT` to the upstream checkout to run them. GPU runs are in
[docs/compatibility.md](../../../../docs/compatibility.md).

See [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0), [LICENSE-MIT](LICENSE-MIT), and [NOTICE](NOTICE). The SDXL weights
carry the CreativeML Open RAIL++-M License.
