# Testing

Run commands from the repository root. The default suite needs no GPU, compiled backend, or checkpoint download.

## Setup

```bash
pip install -e .
pip install -e "packages/diffusers-3d[test,portable]"
```

## Markers

Pytest runs with `--strict-markers`, so an unknown marker is an error.

| Marker | Needs | In CI |
|---|---|---|
| `portable` | the `portable` extra | own job |
| `reference_parity` | pinned upstream checkouts | own job |
| `release` | built-distribution checks | own job |
| `integration` | nothing extra; crosses subsystems | core job |
| `accelerated` | a GPU or compiled backend | manual only |
| `research_only` | a research-licensed backend | manual only |

Tests that use a fake of an accelerated API are ordinary CPU tests. Markers describe what a test needs to run.

## Commands

```bash
python -m pytest packages/diffusers-3d/tests                       # everything available locally
python -m pytest packages/diffusers-3d/tests \
  -m "not portable and not accelerated and not research_only and not reference_parity and not release"  # core CPU
python -m pytest packages/diffusers-3d/tests -m portable
python -m pytest packages/diffusers-3d/tests -m release
python -m ruff check packages/diffusers-3d/src packages/diffusers-3d/tests packages/diffusers-3d/tools
python utils/check_ai.py
```

## Reference parity

These tests import the upstream code at a pinned commit and compare it with the port using tiny random weights.
Default checkout locations:

| Family | Path | Commit |
|---|---|---|
| TRELLIS | `/tmp/TRELLIS` (with its FlexiCubes submodule) | `442aa1e1afb9014e80681d3bf604e8d728a86ee7` |
| TRELLIS.2 | `/tmp/TRELLIS.2` | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| TripoSG | `/tmp/TripoSG` | `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` |
| SAM 3D Objects | `/tmp/sam-3d-objects` | `f91db411c50efee93d8db7aeb323885650f6f722` |
| Step1X-3D | `/tmp/Step1X-3D` | `cb5ac944709c6c913109070c7b90c3447f57f3d4` |
| Pixal3D | `/tmp/Pixal3D` | `f7cf38429b0bd264f1995f0f8743a88b1c728b94` |
| NAF (for Pixal3D) | `/tmp/NAF` | `37f2dfc180f2de53d98bd601109c0da0dd6b0f43` |
| Hunyuan3D-2.1 | `/tmp/Hunyuan3D-2.1` | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` |

Override a path with `DIFFUSERS_3D_<NAME>_REFERENCE_ROOT`, where `<NAME>` is `TRELLIS`, `TRELLIS2`, `TRIPOSG`,
`SAM3D`, `STEP1X3D`, `PIXAL3D`, `NAF`, or `HUNYUAN3D`. Before importing anything, the tests check the commit, the
`origin` URL, and that the worktree is clean. Install the `reference-test` extra plus OpenCV, PyYAML, tqdm, and trimesh,
which the upstream modules import.

A missing or mismatched checkout is skipped locally. `DIFFUSERS_3D_REQUIRE_REFERENCE=1` turns those skips into
failures, which is how CI runs them. CI also requires exactly 48 parity cases.

## Distributions

```bash
python -m build --outdir /tmp/dist packages/diffusers-3d
python packages/diffusers-3d/tools/verify_wheel.py /tmp/dist/diffusers_3d-*.whl
python packages/diffusers-3d/tools/verify_sdist.py /tmp/dist/diffusers_3d-*.tar.gz
```

## GPU runs

GitHub CPU jobs cannot test CUDA backends or the released weights. `scripts/gpu_smoke.py` does: it converts the
TRELLIS and TRELLIS.2 releases, generates from an image and a prompt, and exercises every compiled backend built by
`scripts/install_gpu_backends.sh`. It checks coarse invariants only (finite, non-empty, not blank). Run it on any CUDA
machine with `HF_TOKEN` set, or trigger the `Diffusers 3D GPU smoke` workflow by hand, which runs it on a Hugging Face
Jobs A100 and needs an `HF_TOKEN` repository secret.

The other families were run by hand on the same hardware. [compatibility.md](compatibility.md) records every run.
Offloading (sequential, model, and group) only makes sense with an accelerator, so it is tested manually too.
