# Hunyuan3D-2.1 image-to-shape

This family ports the shape stage of Tencent Hunyuan3D-2.1 at upstream revision
`82920d643c0dc2f7bfd7255f45f62d386edfe60c` and converts the released weights from the Hub repository
`tencent/Hunyuan3D-2.1` (revision `0b94677654c57bb9a6b6845cd7b704ccf551d327`, folder `hunyuan3d-dit-v2-1`).
Everything runs in plain PyTorch on any device; mesh extraction needs scikit-image (`pip install "diffusers-3d[portable]"`).
The PBR texture stage (Hunyuan3D-Paint 2.1) is not ported yet.

> **License.** The code in this directory and the converted weights are governed by the
> [Tencent Hunyuan 3D 2.1 Community License](LICENSE-TENCENT-HUNYUAN-3D-2.1), not by Apache-2.0. The license **does
> not apply in the European Union, the United Kingdom, or South Korea**, requires a separate license from Tencent for
> products above 1 million monthly active users, forbids using outputs to improve other AI models, and incorporates
> an Acceptable Use Policy. Constructing `Hunyuan3DImageTo3DPipeline` emits a `UserWarning` saying so. See
> [NOTICE](NOTICE).

## Components

| Component | Class | Notes |
| --- | --- | --- |
| `image_encoder` | `transformers.Dinov2Model` | DINOv2-L/14 at 518 px; CLS and patch tokens after the final layer norm condition the DiT; all-zero unconditional context |
| `transformer` | `Hunyuan3DDiTModel` | 21-block U-skip DiT, hidden size 2048, MoE (8 experts, top 2, one shared expert) in the last 6 blocks; Diffusers `Timesteps`, `TimestepEmbedding`, `FeedForward` |
| `vae` | `Hunyuan3DShapeVAE` | ShapeVAE decoder: 16 self-attention blocks over the 4096 x 64 latent set, then a cross-attention occupancy query |
| `scheduler` | `Hunyuan3DFlowMatchEulerDiscreteScheduler` | Diffusers `FlowMatchEulerDiscreteScheduler` with Hunyuan's schedule: sigmas `linspace(0, 1, steps)` then a terminal 1 |

`Hunyuan3DImageTo3DPipeline` returns one `MeshAsset` (right-handed, Y up, inside `[-1.01, 1.01]^3`) per image with
the released defaults: 50 steps, classifier-free guidance 5.0 (negative disables it), a 385^3 occupancy grid
(`octree_resolution=384`), iso-level 0, and 8000 grid points per decoder chunk. It also supports
`Object3DLoraLoaderMixin` LoRAs on the transformer.

## Conversion

```bash
diffusers-3d-convert-hunyuan3d tencent/Hunyuan3D-2.1 hunyuan3d-2.1-diffusers   # or a local release folder
```

The converter reads `config.yaml` and the aggregate `model.fp16.ckpt`, rejects any option the release does not use,
remaps keys onto the Diffusers modules (fp16 is kept), re-saves DINOv2 as a stock Transformers component (in
float32), drops the ShapeVAE point-cloud encoder, and writes the
Diffusers folder, the `object3d_model_index.json` sidecar, and `hunyuan3d_conversion.json`. The released attention
layers split heads after concatenating the q/k/v projections; the converter permutes those projection rows once so
that ordinary per-projection head splitting computes the same attention. Every mapped key and shape of the published
checkpoint (752 transformer, 439 DINOv2, and 266 decoder tensors) was checked against these modules.

## Differences from upstream

- **Inputs.** The pipeline takes `ImageCondition` values whose alpha channel or mask already isolates the object; it
  never removes backgrounds. Upstream recenters with OpenCV; `resize_like_opencv` reproduces OpenCV's `INTER_AREA`
  and `INTER_CUBIC` uint8 arithmetic in NumPy (OpenCV is not a dependency). Measured against OpenCV 5.0, the
  preprocessed image differs by one level on 14 of 786,432 values for a shrunk crop and on 1 for an enlarged crop.
- **Surface extraction.** This is the released default path: a dense occupancy grid and scikit-image Lewiner
  marching cubes, with upstream's vertex scaling (divided by `octree_resolution + 1`) and its exported face winding.
  Upstream returns a `trimesh.Trimesh` built with `process=True`, which merges vertices equal to 8 decimals; this
  pipeline returns the marching-cubes mesh unmerged. If the occupancy field never crosses `mc_level`, upstream
  returns `None` for that object while this pipeline raises a `ValueError`. FlashVDM hierarchical decoding and the DISO extractor are
  opt-in upstream accelerations and are not ported.
- **Not ported.** The ShapeVAE encoder (it needs `torch_cluster`), multi-view and text conditioning, training code,
  and upstream's mesh post-processing (floater and degenerate-face removal, face reduction).

## Parity

`tests/families/hunyuan3d/test_reference_parity.py` loads the pinned checkout from
`DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT` and compares tiny random-weight models on CPU in float32: the transformer
output is bit-identical, the VAE decoder within 1e-6, the scheduler bit-identical, and upstream's own
`Hunyuan3DDiTFlowMatchingPipeline.__call__` against this pipeline gives identical conditioning, final latents, and
mesh. The released checkpoint has not been compared end to end on a GPU in this repository.
