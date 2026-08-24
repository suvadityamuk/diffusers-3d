# Testing diffusers-3d

Run commands from the repository root. The ordinary CPU suite never needs CUDA or a compiled backend. Tests that
need optional portable packages, real accelerated backends, restricted research dependencies, or pinned source
checkouts have explicit markers.

## Marker policy

| Marker | Meaning | Ordinary CPU lane |
|---|---|---|
| `portable` | Uses an installed CPU geometry package from the `portable` extra | Separate portable lane |
| `accelerated` | Requires an actual GPU or compiled accelerated backend | Excluded |
| `research_only` | Requires an explicitly accepted research-only dependency/license | Excluded |
| `reference_parity` | Imports an identity-, origin-, tree-, and cleanliness-verified checkout | Separate required-reference lane |
| `integration` | Crosses model, pipeline, conversion, or training subsystem boundaries | Included |
| `release` | Checks manifests, release metadata, or a built distribution | Separate release lane |

Unknown markers are errors because pytest runs with `--strict-markers`. CPU fakes that verify an accelerated API or
research-license gate remain ordinary CPU tests: the markers describe runtime requirements, not the backend category
being simulated.

## Setup and exact commands

```bash
python -m pip install -e .
python -m pip install -e "packages/diffusers-3d[test,training]" --no-deps
python -m pip install \
  accelerate build peft pytest pytest-cov PyYAML ruff safetensors \
  "transformers>=5.5.0"
```

Core CPU, including deterministic equations and tiny integration tests:

```bash
python -m pytest packages/diffusers-3d/tests \
  -m "not portable and not accelerated and not research_only and not reference_parity and not release"
```

Portable CPU backends:

```bash
python -m pip install -e "packages/diffusers-3d[portable]" --no-deps
python -m pytest packages/diffusers-3d/tests -m portable
```

Useful focused lanes:

```bash
python -m pytest packages/diffusers-3d/tests -m "integration and not portable"
python -m pytest packages/diffusers-3d/tests -m reference_parity
python -m pytest packages/diffusers-3d/tests -m release
```

The full locally available suite is:

```bash
python -m pytest packages/diffusers-3d/tests
```

Reference parity defaults to these pinned source trees:

- `/tmp/Hunyuan3D-2.1` at `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- `/tmp/TRELLIS` at `442aa1e1afb9014e80681d3bf604e8d728a86ee7`
- `/tmp/TRELLIS.2` at `75fbf0183001ed9876c8dbb35de6b68552ee08bd`

The roots can be overridden with
`DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT`,
`DIFFUSERS_3D_TRELLIS_REFERENCE_ROOT`, and
`DIFFUSERS_3D_TRELLIS2_REFERENCE_ROOT`. Each value must name the repository
root. Before any upstream module is imported, tests use Git to require the
exact commit, expected `origin` URL, a clean tracked/untracked worktree, and
the expected source paths in that commit.

Generic local runs skip a missing, mismatched, dirty, or dependency-incomplete
reference. The dedicated CI lane installs `einops`, `omegaconf`,
`opencv-python-headless`, `PyYAML`, `scikit-image`, and `timm`, fetches all
three exact commits, and runs:

```bash
DIFFUSERS_3D_REQUIRE_REFERENCE=1 \
  python -m pytest packages/diffusers-3d/tests -m reference_parity
```

Required mode turns every reference or dependency skip condition into a test
failure. CI also parses JUnit output to require all 15 cases and zero skips.
Scheduler, guidance, objective, interpolation, target, and timestep equations
still run deterministically in the ordinary CPU lane without those trees.

## Static, manifest, and release checks

```bash
python -m ruff check packages/diffusers-3d/src packages/diffusers-3d/tests packages/diffusers-3d/tools
for manifest in packages/diffusers-3d/src/diffusers_3d/families/*/diffusers_3d_integration.json; do
  diffusers-3d-validate "$manifest"
done
python utils/check_ai.py
```

Build and inspect both distribution formats:

```bash
REPOSITORY_ROOT="$(pwd)"
python -m build --outdir /tmp/diffusers3d-dist packages/diffusers-3d
python packages/diffusers-3d/tools/verify_wheel.py /tmp/diffusers3d-dist/diffusers_3d-*.whl
python packages/diffusers-3d/tools/verify_sdist.py /tmp/diffusers3d-dist/diffusers_3d-*.tar.gz
python -m venv /tmp/diffusers3d-wheel-venv
/tmp/diffusers3d-wheel-venv/bin/python -m pip install -e .
/tmp/diffusers3d-wheel-venv/bin/python -m pip install \
  accelerate safetensors "transformers>=5.5.0"
/tmp/diffusers3d-wheel-venv/bin/python -m pip install --no-deps /tmp/diffusers3d-dist/diffusers_3d-*.whl
/tmp/diffusers3d-wheel-venv/bin/python -m pip check
cd /tmp
/tmp/diffusers3d-wheel-venv/bin/python -I -c \
  "import diffusers_3d; print(diffusers_3d.__version__)"
/tmp/diffusers3d-wheel-venv/bin/diffusers-3d-report
cd "$REPOSITORY_ROOT"
python -m zipfile -e /tmp/diffusers3d-dist/diffusers_3d-*.whl /tmp/diffusers3d-wheel
python -m tarfile -e /tmp/diffusers3d-dist/diffusers_3d-*.tar.gz /tmp/diffusers3d-sdist
diffusers-3d-check-release \
  packages/diffusers-3d/src/diffusers_3d \
  /tmp/diffusers3d-wheel \
  /tmp/diffusers3d-sdist
```

The manifest validator intentionally reports policy warnings for restricted Hunyuan assets, DINOv3, nvdiffrast,
diffoctreerast, and mip-Gaussian research dependencies. Any validation error is a failure.

## GPU and research verification

No hosted GitHub CPU job is evidence of CUDA, ROCm, compiled-extension, real-checkpoint, rendering, or production
quality. Such runs must be started manually on an appropriate licensed machine, record the package report from
`diffusers-3d-report`, and use `-m accelerated` or `-m research_only` as applicable. There are currently no shipped
tests that claim a completed production GPU quality run.

Shared reviewed-model contracts exercise CPU batching, dtype/device movement,
tuple output equivalence, component save/load, attention processor/backend
hooks, gradient checkpointing where implemented, and `torch.compile` with the
eager backend. Shared reviewed-pipeline contracts additionally exercise
batching, tuple output equivalence, and Hunyuan's supported step callback.
Sequential CPU offload, model CPU offload, and group offload are meaningful
only with an accelerator execution device for these pipelines; they remain
part of the manual GPU lane rather than being simulated on CPU.
