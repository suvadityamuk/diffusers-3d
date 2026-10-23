# Diffusers 3D

`diffusers-3d` runs generative 3D models on [Diffusers](https://github.com/huggingface/diffusers). Pipelines are
standard `DiffusionPipeline`s, every network runs in plain PyTorch, and outputs are tensor-native meshes, Gaussian
splats, sparse voxels, and O-Voxels.

The package is pre-alpha. Supported families: TRELLIS, TRELLIS.2, TripoSG, Step1X-3D (geometry and texture),
Hunyuan3D-2.1 (shape and Paint), SAM 3D Objects, and Pixal3D. The
[repository README](https://github.com/suvadityamuk/diffusers-3d#supported-models) lists their Hub checkpoints.

## Install

```bash
pip install -e .                                  # Diffusers core, from the repository root
pip install -e "packages/diffusers-3d[portable]"  # this package with CPU mesh tools
```

Requires Python 3.10+, PyTorch 2.6+, Transformers 5.5+, and Accelerate 1.1+. Extras:

- `portable`: trimesh, scikit-image, xatlas, fast-simplification. Needed for marching cubes (TripoSG, Step1X-3D,
  Hunyuan3D-2.1), the texture stages, and GLB export.
- `gaussian`: gsplat, for rasterizing splats.
- `lora`: PEFT, for `load_lora_weights`.

Compiled backends (CuMesh, O-Voxel, FlexGEMM, nvdiffrast) are installed separately; see [docs/backends.md](docs/backends.md).

## Usage

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition, TrimeshBackend

rgba = ...  # (4, H, W) float tensor in [0, 1]; alpha marks the object
pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/TripoSG-diffusers-3d").to("cuda")
mesh = pipeline(ImageCondition(image=rgba), generator=torch.Generator("cuda").manual_seed(0)).objects[0]
TrimeshBackend().export_mesh(mesh.to("cpu"), "mesh.glb")
```

[docs/inference.md](docs/inference.md) covers every family, the texture stages, saving, LoRA, and conversion. The
[TRELLIS.2 example](src/diffusers_3d/families/trellis2/examples/image_to_3d.py) runs offline with tiny weights:

```bash
python -m diffusers_3d.families.trellis2.examples.image_to_3d --tiny --output out/
```

## Limitations

- No pipeline removes backgrounds. Pass RGBA or a mask.
- TRELLIS.2 and Pixal3D need the gated DINOv3 image encoder passed to `from_pretrained`; it is not redistributed.
- TripoSG, Step1X-3D, and Hunyuan3D-2.1 extract raw marching-cubes meshes. Upstream's CUDA-only fast decoders and
  pymeshlab post-processing are not ported, mostly for licensing reasons.
- The texture stages fill texels that no view sees with their own fill instead of `cv2.inpaint`, so those texels
  differ slightly from upstream. Step1X-3D's generated views do not match upstream's for a given seed.
- SAM 3D Objects needs a mask. MoGe depth estimation and layout post-optimization are not ported.
- PBR-textured mesh export for TRELLIS and TRELLIS.2 needs CuMesh, whose bundled cubvh is under an NVIDIA
  non-commercial license.

Each family README lists its exact differences from upstream.

## License

The distribution's license expression is
`Apache-2.0 AND MIT AND LicenseRef-SAM-License AND LicenseRef-Tencent-Hunyuan-3D-2.1-Community`. Package glue and
Step1X-3D-derived code are Apache-2.0. Ported TRELLIS, TRELLIS.2, TripoSG, and Pixal3D code, and Step1X-3D's
SyncMVD-derived texture sync, keep their MIT terms. SAM 3D Objects code is under Meta's SAM License. Hunyuan3D-2.1 code
is under the Tencent Hunyuan 3D 2.1 Community License, which does not apply in the EU, UK, or South Korea. Each term
applies only to its own files, and model weights keep their own licenses. Pipelines with restrictive terms warn when
constructed.

## More

- [Compatibility](docs/compatibility.md): supported versions and verified runs.
- [Testing](docs/testing.md): markers and commands.
- [Contributing](CONTRIBUTING.md): adding a model family.
