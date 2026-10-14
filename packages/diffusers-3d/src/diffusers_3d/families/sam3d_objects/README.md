# SAM 3D Objects image-to-3D

This family integrates Meta's [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) at revision
`f91db411c50efee93d8db7aeb323885650f6f722` (weights: `facebook/sam-3d-objects`, gated). It reconstructs the object
selected by a mask as Gaussian splats and/or a FlexiCubes mesh, placed in the input camera's frame.

> **License.** The SAM 3D Objects code and weights are released under the [SAM License](LICENSE-SAM), which is not an
> open-source license. It grants a royalty-free right to use, copy, modify, and redistribute (including
> commercially), but forbids military, warfare, nuclear, espionage, and weapons uses and use by parties subject to
> trade controls, requires the license to accompany any redistribution and an acknowledgement in publications, and can
> be terminated or modified by Meta. The ported code in this directory is a derivative work under the same terms.
> Constructing `Sam3DObjectsPipeline` emits a `UserWarning` saying so.

## Architecture and what is reused

SAM 3D Objects is TRELLIS with a new geometry stage and new conditioning:

| Stage | Class | Source |
|---|---|---|
| Image and mask encoder | `transformers` `Dinov2WithRegistersModel` (`image_encoder`) | stock DINOv2-L/14 with registers, run on the object crop and full frame of the image and of the mask; the released checkpoint keeps four identical copies, all equal to `facebook/dinov2-with-registers-large` |
| Image, mask, and pointmap conditioning | `Sam3DConditionEmbedder` | new: Diffusers SwiGLU `FeedForward` projections of the DINOv2 tokens and a pointmap window encoder built on a Diffusers `BasicTransformerBlock` |
| Shape + layout flow (16^3 latent, rotation, translation, scale) | `Sam3DSparseStructureFlowModel` | new mixture-of-transformers wiring of the TRELLIS attention, feed-forward, norm, and timestep modules |
| Occupancy decoder | `TrellisSparseStructureDecoder` | TRELLIS, unchanged |
| Structured-latent flow | `TrellisSLatFlowModel` | TRELLIS, unchanged |
| Gaussian (32 per voxel) and mesh decoders | `TrellisSLatGaussianDecoder`, `TrellisSLatMeshDecoder` | TRELLIS, unchanged (softplus scale activation) |
| Both samplers | `TrellisFlowEulerScheduler` | TRELLIS, run on `1 - t` (SAM 3D integrates noise at `t=0` to data at `t=1`) |

The released defaults are reproduced: 25 sparse-structure steps with `rescale_t=3` and guidance 7 applied to every
modality, 25 structured-latent steps with `rescale_t=1` and guidance 1, guidance in the noisier half of each
trajectory, zeroed conditioning as the unconditional branch, decoder occupancy pruned to its surface voxels (and
halved above 42,000 voxels), and the `ScaleShiftInvariant` layout decoding.

## Usage

```python
import torch
from diffusers_3d import AutoPipelineForImageTo3D, ImageCondition

pipeline = AutoPipelineForImageTo3D.from_pretrained("suvadityamuk/sam-3d-objects-diffusers-3d").to("cuda")
condition = ImageCondition(image=rgb, mask=object_mask)  # (3, H, W) in [0, 1] and (1, H, W)
output = pipeline(condition, pointmap=points, generator=torch.Generator("cuda").manual_seed(42))
gaussians, mesh = output.objects
```

- **Mask.** Required on every `ImageCondition`; the pipeline never segments. Like upstream, the RGB image is not
  masked: the crop is centred on the mask and the mask is encoded separately.
- **Pointmap.** Optional `(3, H, W)` camera-space points in OpenCV axes (x right, y down, z forward; non-finite where
  invalid), for example MoGe's `points` as upstream computes them. MoGe is a separate model that is not part of the
  released checkpoint, so it is not bundled. Without a pointmap the pointmap tokens are zeroed (the modality dropout
  the model was trained with); the shape is still generated but the layout is scale-free.
- **Layout.** Every asset's `transform` places it in the PyTorch3D camera frame of the input image (x left, y up,
  z forward), in the pointmap's units. The transform is expressed in the asset's Z-up convention, so after
  `asset.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP)` world coordinates are exactly camera coordinates.
  `metadata["layout"]` repeats upstream's values: `scale * local @ rotation + translation` with `rotation` the matrix of
  upstream's quaternion, plus `metric` (whether a pointmap was given). Bake or drop the transform before rendering
  with `GsplatBackend`, which requires identity transforms.

## Conversion

```bash
diffusers-3d-convert-sam3d /path/to/sam-3d-objects/checkpoints /path/to/converted
```

The converter reads `pipeline.yaml` and the component yamls, refuses configurations it does not reproduce, maps
upstream keys onto the Transformers and package modules (torch-hub DINOv2 to `Dinov2WithRegistersModel`, converted
once after checking that all four copies match, timm window block to `BasicTransformerBlock`, Llama SwiGLU to Diffusers
`FeedForward`, the mixture-of-transformers streams), and loads every component with strict key checks. It reads `<name>.safetensors` when present and otherwise the released `.ckpt`
(`torch.load(weights_only=True)`).

## Parity

`tests/families/sam3d_objects/test_reference_parity.py` runs tiny random weights through the pinned upstream code and
the port on CPU float32. The condition embedders and the SLAT sampler are bitwise equal; preprocessing agrees within
4.8e-7, the sparse-structure flow within 5.4e-7, and the sparse-structure sampler with layout within 1e-4. By hand on
an A100 (2026-10-13), the published `suvadityamuk/sam-3d-objects-diffusers-3d`, given the TRELLIS `T.png` example
and its alpha as the mask, returned Gaussian splats and a 0.58M-face mesh in 37 s (float32, 11.1 GB peak).

## Not ported

- MoGe depth estimation (separate model; pass a pointmap instead) and its intrinsics inference.
- Layout post-optimization (render-and-compare with PyTorch3D/gsplat), mesh simplification, hole filling, texture
  baking, and plane estimation; all are off in upstream's default inference, which exports vertex-colored meshes.
- The 4-Gaussian decoder (`slat_decoder_gs_4`), the sparse-structure encoder, the copy of the occupancy decoder inside
  `ss_generator`, and shortcut (distilled) sampling, none of which the default inference uses.
- Upstream runs the sparse-structure stage under float16 autocast on CUDA; here each component runs in its own dtype.
