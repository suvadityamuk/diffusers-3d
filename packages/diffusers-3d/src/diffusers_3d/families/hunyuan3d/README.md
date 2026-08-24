# Hunyuan3D-2.1 image-to-shape

This reviewed family integrates the released Hunyuan3D-2.1 single-image shape
flow at upstream revision `82920d643c0dc2f7bfd7255f45f62d386edfe60c`.
The Hunyuan-derived implementation and converted checkpoints remain subject to
the Tencent Hunyuan 3D 2.1 Community License Agreement. Apache-2.0 applies only
to independently written `diffusers-3d` package glue as described in
`diffusers_3d_integration.json`.

## Production-compatible paths

- `Hunyuan3DShapeDiTModel` preserves released parameter names and the unusual
  concatenate-then-head-split attention order. Tiny CPU float32 output was
  measured against the pinned `HunYuanDiTPlain` with identical weights.
- `Hunyuan3DShapeVAE` maps `post_kl`, all latent transformer blocks, and the
  geometric cross-attention decoder without tensor surgery. Tiny CPU float32
  decode and field logits were measured against the pinned `ShapeVAE` with
  identical decoder weights.
- `Hunyuan3DDinov2Conditioner` preserves DINOv2 ImageNet normalization, resize /
  center-crop behavior, CLS-token handling, and all-zero unconditional tokens.
- `Hunyuan3DFlowMatchEulerDiscreteScheduler` preserves the released ascending
  sigma schedule, including its repeated terminal sigma, instead of using the
  opposite default Diffusers flow direction.
- `Hunyuan3DImageToShapePipeline` implements preprocessing, DINO conditioning,
  classifier-free guidance, Euler flow integration, chunked dense field
  evaluation, and `MeshAsset` extraction through `ScikitImageBackend`.
- The converter accepts local aggregate `.ckpt` or `.safetensors` files and
  writes ordinary per-component Diffusers folders plus object-3D metadata.

The released production configuration uses 4096x64 shape latents, a
21-layer/2048-width denoiser, a 16-layer/1024-width VAE decoder, DINOv2-large at
518 pixels, 50 flow steps, guidance 5, bounds 1.01, dense resolution 384, and
level 0. Production inference requires substantial accelerator memory; the
dense 385-cubed field is evaluated in chunks but surface extraction runs on
CPU.

## Training evidence

`Hunyuan3DShapeFlowMatchingRecipe` trains only the denoiser with the released
objective:

`x0 ~ N(0, I)`, `t ~ Uniform(0, 1)`,
`x_t = t*x1 + (1-t)*x0`, target velocity `x1-x0`, and mean squared error.

`Hunyuan3DShapeExample` is the recipe-owned exact dataset item contract and
contains an `ImageCondition` plus exactly one of precomputed shape latents or
surface samples. Recipe construction does not mutate the target. After exact
registration validation, `Object3DTrainer.prepare()` freezes the complete
pipeline and unfreezes only the policy-approved denoiser parameters.
Full-denoiser fine-tuning is registered. LoRA is not registered because
upstream Hunyuan3D-2.1 does not publish LoRA target evidence. Tests use
precomputed shape latents.

Tiny CPU evidence covers pinned-reference denoiser forward and selected
backward gradients, ShapeVAE decode/field values, a composed DINO-token /
denoising / decode-field path, one full trainer step with exact trainable and
optimizer parameter sets, and exact package checkpoint weight restoration.

## Explicit limitations

- The VAE is decode-only. Its point-cloud encoder is not ported because the
  released path depends on `torch_cluster.fps`; `encode()` raises
  `NotImplementedError`.
- Training accepts precomputed shape latents only. Surface-sample training is
  not implemented, so latents must be precomputed with the official licensed
  stack.
- FlashVDM, hierarchical volume decoding, DISO/DMC extraction, dual/multiview
  conditioning, and texture generation are unsupported.
- Core mesh output does not import or return `trimesh`.
- CI exercises tiny CPU configurations without downloads. The composed parity
  test stops at scalar-field logits. No official approximately 7 GB
  checkpoint/GPU quality run was performed. Full official checkpoint loading,
  production-resolution GPU output parity, quality metrics, and end-to-end GPU
  parity are not claimed.
- Hunyuan-derived code and converted checkpoints remain under the restricted
  Tencent Hunyuan 3D 2.1 Community License Agreement.

Install the mesh backend and converter dependency with:

```bash
pip install "diffusers-3d[hunyuan3d]"
```

Convert a local official checkpoint:

```bash
diffusers-3d-convert-hunyuan3d model.fp16.ckpt output/ \
  --config config.yaml \
  --conditioner-path /local/path/to/dinov2-large
```

`--conditioner-path` is required only when the aggregate checkpoint has no
`conditioner` component.
