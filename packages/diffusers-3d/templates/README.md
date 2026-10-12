# Contribution templates

These starters are code scaffolds. Copy a directory and replace every example repository, revision, class, and test.

## Reviewed model family

`reviewed-model-family/` shows the package contracts for an exact model, object-native pipeline, converter, and
registrations. Replace the starter conversion mapping only with behavior proven against the pinned upstream
revision, then follow the checklist in [the contribution guide](../CONTRIBUTING.md).

## Experimental custom block

`experimental-custom-block/` contains an inference-only `ModularPipelineBlocks` starter and its
`modular_config.json`. Pin the actual Hub commit digest before sharing it. Package the block with:

```bash
diffusers-cli custom_blocks \
  --block_module_name block.py \
  --block_class_name ExperimentalObject3DBlock
```

Loading remote code requires explicit trust from the consumer.
