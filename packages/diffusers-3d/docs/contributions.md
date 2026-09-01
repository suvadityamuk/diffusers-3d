# Integration contribution lifecycle

An integration manifest records review evidence; it does not register code by itself. Every manifest uses schema
`diffusers-3d-integration` version `2`, rejects unknown JSON fields at every level, and is validated without network
access.

Validate a local record with:

```bash
diffusers-3d-validate path/to/integration_manifest.json
```

## 1. Hub remote-code staging

Use an experimental Hub custom block to iterate on an inference workflow before package review.

- Pin the Hub repository to a full immutable commit digest.
- Declare the task, workflow, input/output representations, dependencies, and licenses known at this stage.
- Require consumers to opt into remote code explicitly.
- Do not add the class to package model, pipeline, or training registries.
- Do not declare a training qualification. Experimental blocks are inference-only.

The starter in [`templates/experimental-custom-block`](../templates/experimental-custom-block) includes the block
source, modular config, and manifest.

## 2. Reviewed companion-package integration

Promotion replaces remote code with exact package-owned classes and immutable review evidence. A reviewed integration
must declare:

- the full upstream repository commit digest;
- every exact model, pipeline, scheduler, processor, or converter class and its unique role;
- task routing and tensor-native input/output representations;
- checkpoint conversion and public `save_pretrained`/`from_pretrained` round trips;
- component and end-to-end parity tests against the pinned upstream implementation;
- every runtime backend, support level, install path, version/build constraint, and license classification;
- upstream model and redistributed artifact licenses.

Registries resolve exact classes. A subclass, structurally similar class, or class from a different revision does not
inherit review.

### Conversion and parity checklist

1. Map one upstream component at a time and load converted state dictionaries with `strict=True`.
2. Save and reload through public Diffusers APIs; do not add custom runtime weight fetching.
3. Compare freshly converted and reference components side by side on deterministic CPU float32 inputs.
4. Record the executable test path, reference behavior, tolerances, and passing result in the manifest.
5. Run a tiny real-class pipeline save/load round trip and verify object-native output semantics.
6. Verify task metadata, coordinate conventions, representation channels, dtype, and device preservation.
7. Keep conversion tests distinct from normal model and pipeline contract tests.

Use [`templates/reviewed-model-family`](../templates/reviewed-model-family) as the package skeleton.

## 3. Optional upstream Diffusers primitive

After package review, dependency-light and generally useful primitives may be proposed to Diffusers. Object-3D task
metadata, specialized backends, and training authorization can remain in this companion package.

Moving a model, scheduler, guider, loader, or utility upstream does not make it trainable. Training still requires the
exact recipe qualification below, and the companion registry must continue to reject unreviewed targets and
subclasses.

## Trainability review

Training is a separate qualification for a reviewed package or upstream integration. Its manifest record must pin:

- recipe identifier, recipe version, recipe class, target class, typed batch class, and stable registration;
- supported full or LoRA strategies and the exact trainable component roles;
- backward parity for recipe-owned parameters;
- checkpoint save/load and continuation parity;
- objective parity against the pinned upstream implementation.

Review also checks typed examples and batches, exact trainable and frozen component policies, gradient ownership,
public checkpoint APIs, and deterministic resume identity. Training manifest schema version `3` records canonical
objective settings and optimizer/scheduler/batch/precision/seed settings. Checkpoints retain family inference
artifacts and an `accelerator_state` continuation directory containing trainable and frozen model state, optimizer,
scheduler, scaler, RNG, trainer counters, and data position. Exact continuation requires
`dataloader_num_workers=0`, a synchronized optimizer-step boundary, and a non-empty caller-supplied
`dataset_fingerprint`. The fingerprint must identify the exact dataset contents, ordering, preprocessing, and
sampling contract; changing it is a resume-manifest mismatch. The trainer does not infer this identity from an
arbitrary dataset object, and asynchronous dataset worker state is intentionally not claimed. Distributed training
is supported, but exact checkpoint save/load currently requires `accelerator.num_processes == 1`; multi-process
persistence is rejected before filesystem or collective operations. Resume strictly validates this process's
Accelerator RNG payload and explicitly restores Python, NumPy, CPU torch, and recorded available device RNG states.
Accelerator state is the authoritative continuation format. Recipe `save_weights()` artifacts remain inference
artifacts and are not a second resume path.
`Object3DTrainer.train()` returns one CPU-only `TrainingSummary3D` instead of retaining every device output. Missing
or failed evidence keeps the integration inference-only.

## Backend and license review

Every backend declaration includes its capabilities, distribution or pinned source, tested version/build constraint,
support classification, exact license identifier, coarse license classification, and actionable installation hint.

- `portable` backends should work without a source build or vendor runtime.
- `accelerated` backends may require device-specific wheels or builds.
- `research_only` backends are never selected implicitly and generate a validation warning.
- `restricted` and `unknown` licenses generate warnings and require explicit review.

Model weights and every converted or generated artifact need separate license records. A permissive package license
does not override a model, checkpoint, renderer, or dataset license.

## First-release check

The pre-release-only scaffold has been removed. The first code release must satisfy this checklist:

1. Production execution and training registries are frozen and contain only exact reviewed family registrations.
2. Tiny model, pipeline, auto-loading, typed training, and checkpoint round trips cover every registered family.
3. The full tests, Ruff checks, package installation/import smoke test, and every integration manifest validator pass.
4. A wheel and sdist are built, verified for metadata/completeness, unpacked, and checked separately from the source
   tree.
5. Neither source, wheel, nor sdist contains the reserved release marker or removed scaffold names and paths.
6. Version finalization and publication remain explicit maintainer actions after the code gate.

Run the offline checker over all selected source paths and both unpacked distributions:

```bash
diffusers-3d-check-release \
  src tests templates docs requirements \
  README.md CONTRIBUTING.md COMPATIBILITY.md LICENSE-APACHE-2.0 MANIFEST.in pyproject.toml
diffusers-3d-check-release /tmp/diffusers-3d-wheel
diffusers-3d-check-release /tmp/diffusers-3d-sdist
```

The command reports every reserved-marker path, line, and column and exits nonzero until removal is complete. Its
default marker is reconstructed at runtime so the checker does not place the blocked byte sequence in its own wheel.
Callers may use `--marker` for another release-blocking marker.
