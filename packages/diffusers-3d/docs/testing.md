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
| `reference_parity` | Imports a pinned checkout under `/tmp` and compares tiny tensors | Collected; skips if absent |
| `integration` | Crosses model, pipeline, conversion, or training subsystem boundaries | Included |
| `release` | Checks manifests, release metadata, or a built distribution | Separate release lane |

Unknown markers are errors because pytest runs with `--strict-markers`. CPU fakes that verify an accelerated API or
research-license gate remain ordinary CPU tests: the markers describe runtime requirements, not the backend category
being simulated.

## Setup and exact commands

```bash
python -m pip install -e .
python -m pip install -e "packages/diffusers-3d[test,training]" --no-deps
python -m pip install accelerate peft pytest pytest-cov safetensors transformers
```

Core CPU, including deterministic equations, tiny integration tests, and pinned-reference collection:

```bash
python -m pytest packages/diffusers-3d/tests \
  -m "not portable and not accelerated and not research_only and not release"
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

Reference parity expects these pinned source trees:

- `/tmp/Hunyuan3D-2.1` at `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- `/tmp/TRELLIS` at `442aa1e1afb9014e80681d3bf604e8d728a86ee7`
- `/tmp/TRELLIS.2` at `75fbf0183001ed9876c8dbb35de6b68552ee08bd`

Missing trees are expected skips in generic CI. Scheduler, guidance, objective, interpolation, target, and timestep
equations still run deterministically without those trees.

## Static, manifest, and release checks

```bash
python -m ruff check packages/diffusers-3d/src packages/diffusers-3d/tests packages/diffusers-3d/tools
for manifest in packages/diffusers-3d/src/diffusers_3d/families/*/diffusers_3d_integration.json; do
  diffusers-3d-validate "$manifest"
done
python utils/check_ai.py
```

Build and inspect a wheel:

```bash
python -m build --wheel --outdir /tmp/diffusers3d-dist packages/diffusers-3d
python packages/diffusers-3d/tools/verify_wheel.py /tmp/diffusers3d-dist/diffusers_3d-*.whl
python -m venv /tmp/diffusers3d-wheel-venv
/tmp/diffusers3d-wheel-venv/bin/python -m pip install -e .
/tmp/diffusers3d-wheel-venv/bin/python -m pip install --no-deps /tmp/diffusers3d-dist/diffusers_3d-*.whl
cd /tmp
/tmp/diffusers3d-wheel-venv/bin/python -I -c \
  "import diffusers_3d; print(diffusers_3d.__version__)"
/tmp/diffusers3d-wheel-venv/bin/diffusers-3d-report
cd /workspace
python -m zipfile -e /tmp/diffusers3d-dist/diffusers_3d-*.whl /tmp/diffusers3d-wheel
diffusers-3d-check-release \
  packages/diffusers-3d/src/diffusers_3d /tmp/diffusers3d-wheel
```

The manifest validator intentionally reports policy warnings for restricted Hunyuan assets, DINOv3, nvdiffrast,
diffoctreerast, and mip-Gaussian research dependencies. Any validation error is a failure.

## GPU and research verification

No hosted GitHub CPU job is evidence of CUDA, ROCm, compiled-extension, real-checkpoint, rendering, or production
quality. Such runs must be started manually on an appropriate licensed machine, record the package report from
`diffusers-3d-report`, and use `-m accelerated` or `-m research_only` as applicable. There are currently no shipped
tests that claim a completed production GPU quality run.
