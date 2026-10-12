# Contributing

The quickest way to contribute is to add a 3D model family: port its networks to plain PyTorch, convert its
checkpoint into the Diffusers layout, and register its pipeline so `AutoPipelineFor3D.from_pretrained()` can load it.
Start from [`templates/reviewed-model-family`](templates/reviewed-model-family), and see
[templates/README.md](templates/README.md) for the starters.

## Adding a model family

A family lives in `src/diffusers_3d/families/<family>/` and includes:

- exact model and pipeline classes, registered in `registrations.py`;
- a converter from the upstream checkpoint, exposed as a `diffusers-3d-convert-<family>` script;
- tensor-native `Object3D` outputs;
- tiny real-class tests and `save_pretrained`/`from_pretrained` round trips;
- a README naming the upstream repository and commit, plus `LICENSE`/`NOTICE` files for ported code.

Registries resolve exact classes. A subclass, structurally similar class, or class from a different revision is not
loaded automatically.

### Conversion and parity checklist

1. Map one upstream component at a time and load converted state dictionaries with `strict=True`.
2. Save and reload through public Diffusers APIs; do not add custom runtime weight fetching.
3. Compare freshly converted and reference components side by side on deterministic CPU float32 inputs.
4. Run a tiny real-class pipeline save/load round trip and verify object-native output semantics.
5. Verify task metadata, coordinate conventions, representation channels, dtype, and device preservation.
6. Keep conversion tests distinct from normal model and pipeline contract tests.

## Experimental Hub blocks

To iterate on a workflow before it becomes a family, publish a custom `ModularPipelineBlocks` with a pinned revision
and explicit requirements. See [`templates/experimental-custom-block`](templates/experimental-custom-block).

## Dependency requirements

Do not add a mandatory geometry, rendering, or CUDA dependency. Optional adapters must:

- lazy-import the dependency;
- declare support and license classifications;
- report versions and build/runtime compatibility;
- normalize coordinate and representation conventions;
- provide actionable installation diagnostics;
- never silently replace an exact backend when outputs differ.

Source dependencies must be commit-pinned in `requirements/backends`. Do not use an unrelated package with a
colliding PyPI name. Model weights and converted artifacts keep their own licenses; a permissive package license does
not override a model, checkpoint, or renderer license.

## Before opening a PR

- Ruff and the package test suite pass (see [docs/testing.md](docs/testing.md)).
- `tools/verify_wheel.py` and `tools/verify_sdist.py` pass on freshly built distributions.
