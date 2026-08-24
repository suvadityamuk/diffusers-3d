# Contributing

New work progresses through three levels.

The versioned manifest schema, promotion flow, conversion/parity checklist, trainability review, and backend/license
review are documented in [docs/contributions.md](docs/contributions.md). Start from the
[contribution templates](templates/README.md) and validate the completed local record with:

```bash
diffusers-3d-validate path/to/integration_manifest.json
```

## Experimental Hub contribution

Publish custom `ModularPipelineBlocks` with a pinned revision and explicit requirements. Remote code is
inference-only and is never registered by `Object3DTrainer`.

## Reviewed package integration

A reviewed family must include:

- a versioned integration manifest;
- exact model and pipeline class registrations;
- component-level checkpoint conversion and parity evidence;
- tensor-native `Object3D` output validation;
- tiny real-class tests and save/load round trips;
- backend capability, installation, and license declarations;
- a model card documenting upstream revisions and output semantics.

Training is a separate qualification. It requires a concrete `TrainingRecipe3D`, typed examples and batches, a
component policy, objective parity, gradient-ownership tests, and checkpoint continuation tests.

## Upstream Diffusers primitive

Only dependency-light, generally useful model, scheduler, guider, or loading primitives should be proposed upstream.
The companion package retains object-3D task metadata and training authorization.

## Dependency requirements

Do not add a mandatory geometry, rendering, or CUDA dependency. Optional adapters must:

- lazy-import the dependency;
- declare support and license classifications;
- report versions and build/runtime compatibility;
- normalize coordinate and representation conventions;
- provide actionable installation diagnostics;
- never silently replace an exact backend when outputs differ.

Source dependencies must be commit-pinned in `requirements/backends`. Do not use an unrelated package with a
colliding PyPI name.

Before release, use the offline removal checker described in
[the lifecycle guide](docs/contributions.md#release-removal-check) against every selected source and build path.
