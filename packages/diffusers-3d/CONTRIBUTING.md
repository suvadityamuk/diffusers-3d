# Contributing

The main way to contribute is to add a model family: port its networks to plain PyTorch, convert its checkpoint to the
Diffusers layout, and register its pipeline so `AutoPipelineFor3D.from_pretrained()` can load it. Start from
[`templates/reviewed-model-family`](templates/reviewed-model-family).

## Adding a model family

A family lives in `src/diffusers_3d/families/<family>/` and has:

- model and pipeline classes, registered in `registrations.py`;
- a converter, exposed as a `diffusers-3d-convert-<family>` script in `pyproject.toml`;
- outputs that are `Object3D` assets;
- tiny-config tests, including a `save_pretrained`/`from_pretrained` round trip and parity against the pinned upstream;
- a README naming the upstream repository and commit, and `LICENSE`/`NOTICE` files for ported code.

Reuse before porting:

- Use Transformers image and text encoders (DINOv2, DINOv3, CLIP) as stock pipeline components, loaded and saved with
  Transformers' own methods. Do not wrap them in a Diffusers model; only Transformers' loaders track its parameter
  renames between releases.
- Build on Diffusers blocks (`UNet2DConditionModel`, `AutoencoderKL`, attention processors, schedulers) and the
  package's shared backends (`mesh_rasterizer`, `texture_baking`, `multiview_baking`) instead of porting CUDA
  extensions.
- Remap checkpoint keys in the converter rather than mirroring upstream module trees.
- Keep it inference-only.

Registries resolve exact classes, so a subclass or look-alike class is never loaded automatically.

## Conversion checklist

1. Load converted state dicts with `strict=True`. For Transformers encoders, use `output_loading_info=True` and reject
   any missing, unexpected, or mismatched key.
2. Save and reload through the public Diffusers APIs. No custom runtime weight fetching.
3. Compare each converted component with the pinned upstream on deterministic CPU float32 inputs.
4. Check the coordinate convention, representation channels, dtype, and device of the outputs.

## Publishing converted checkpoints

Add the release to `scripts/checkpoint_sources.py` and a model card to `scripts/publish_hub_checkpoints.py`, then run
the script with access to the source weights. Check that the upstream license allows redistribution. Never upload
gated weights: list them in `WITHHELD` so users pass them in at load time.

## Optional dependencies

Never add a mandatory geometry, rendering, or CUDA dependency. An optional backend must import lazily, declare its
license class and supported versions in a `BackendSpec`, give an actionable error when missing, and never silently
replace another backend whose output differs. Pin source builds to a commit in `requirements/backends/`. Model weights
keep their own licenses regardless of the package license.

## Experimental Hub blocks

To try a workflow before it becomes a family, publish a custom `ModularPipelineBlocks` from
[`templates/experimental-custom-block`](templates/experimental-custom-block) with a pinned revision.

## Before opening a PR

Ruff, the package tests, and the distribution checks must pass; see [docs/testing.md](docs/testing.md).
