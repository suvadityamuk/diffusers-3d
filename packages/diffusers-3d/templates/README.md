# Contribution templates

These starters are review records and code scaffolds, not approved integrations. Copy a directory, replace every
example repository, revision, class, test, license, and evidence value, then run:

```bash
diffusers-3d-validate path/to/integration_manifest.json
```

Validation is entirely local and does not resolve Hub or Git references.

## Experimental custom block

`experimental-custom-block/` contains an inference-only `ModularPipelineBlocks` starter, its `modular_config.json`,
and an experimental integration manifest. Pin the actual Hub commit digest before sharing it. Package the block with:

```bash
diffusers-cli custom_blocks \
  --block_module_name block.py \
  --block_class_name ExperimentalObject3DBlock
```

Loading remote code requires explicit trust from the consumer. Experimental blocks cannot register a stable
`Object3DTrainer` recipe.

## Reviewed model family

`reviewed-model-family/` shows the package contracts for an exact model, object-native pipeline, converter,
registrations, and separately qualified training recipe. The manifest deliberately records every required review
surface. Replace the starter objective and conversion mapping only with behavior proven against the pinned upstream
revision.

Before proposing package review:

1. run component conversion, save/load, and CPU float32 parity tests;
2. run tiny real-class model and pipeline tests;
3. declare every runtime backend and model/artifact license;
4. validate the manifest;
5. request training qualification separately, with backward, checkpoint-continuation, and objective parity.

See [the contribution lifecycle](../docs/contributions.md) and [the package contribution guide](../CONTRIBUTING.md).
