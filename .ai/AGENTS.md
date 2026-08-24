# Diffusers — Agent Guide

## Setup

We recommend developing in a virtual environment managed by [uv](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .                       # provides diffusers-cli
```

List the available skills, and what each one is for, with:

```bash
diffusers-cli skills list
```

Install them with:

```bash
diffusers-cli skills add <skill name>      # install one, or --all for every skill
                                          # --claude / --codex / --cursor to pick one target
```

`diffusers-cli skills update` refreshes what you installed.

Skills used to be installed through the `Makefile`, which symlinked the project-level `.claude/skills` and
`.agents/skills` at `.ai/skills` in this repo. Remove those links if they exist — otherwise the install writes through
them into `.ai/` itself:

```bash
rm -f .claude/skills .agents/skills
```

At the start of a session, confirm the setup and tell the user what you find:

```bash
python utils/check_ai.py                  # guides and skills are consistent
diffusers-cli skills list                 # which skills are installed
```

Claude Code and Codex can also install via plugins

```bash
claude plugin marketplace add huggingface/diffusers
claude plugin install diffusers@diffusers-skills --scope project

codex plugin marketplace add huggingface/diffusers    # then install from the Plugins Directory
```


## Code formatting

- `make style` and `make fix-copies` should be run before opening a PR

## Reference guides

- **Coding style** — see [code_style.md](references/code_style.md) for how code should read, and the `# Copied from` rules.
- **Models** — see [models.md](references/models.md) for model conventions, attention pattern, implementation rules, dependencies, and gotchas. For adding or converting a model, use the [model-integration](skills/model-integration/SKILL.md) skill.
- **Pipelines** — see [pipelines.md](references/pipelines.md) for pipeline conventions, patterns, and gotchas.
- **Modular pipelines** — see [modular.md](references/modular.md) for modular pipeline conventions, patterns, and gotchas.
- **Tests** — see [testing.md](references/testing.md) for test conventions: required test layers, tester mixins, and dummy-component rules.

## Skills

Task-specific guides live in `.ai/skills/` and are loaded on demand by AI agents. Available skills include:

- [model-integration](skills/model-integration/SKILL.md) (adding/converting pipelines)
- [custom-blocks](skills/custom-blocks/SKILL.md) (packaging a `ModularPipelineBlocks` subclass for the Hub)
- [diffusers-cli](skills/diffusers-cli/SKILL.md) (running pipelines, inspecting schemas, and using the Diffusers CLI)
- [self-review](skills/self-review/SKILL.md) (pre-PR self-review against the project rules)

## Self-review before a PR

Before opening a PR, run self-review against [review-rules.md](references/review-rules.md). The [self-review skill](skills/self-review/SKILL.md) runs this as the same pass the `@claude` CI reviewer uses. Share the final report on the PR (description or comment) — see the skill for details.

## Cursor Cloud specific instructions

- **CPU-only, no GPU.** This VM has no CUDA/GPU (`diffusers-cli env` reports `PyTorch ... (False)`). Torch is the CPU wheel (installed from `https://download.pytorch.org/whl/cpu`). Anything gated on a GPU won't run here: `@slow` / `RUN_SLOW=1` nightly integration tests, `@require_torch_gpu`, quantization backends (bitsandbytes, etc.), and most `tests/pipelines/**` integration tests that load full checkpoints. Stick to CPU unit tests (schedulers, small model/pipeline unit tests with tiny dummy components).
- **The venv already exists at `.venv`** (the update script builds it). Activate with `source .venv/bin/activate` before running anything; `diffusers` is installed editable, so `src/` edits are picked up immediately with no reinstall.
- **Lint/format/test/run** commands are the standard ones in the `Makefile` (`make quality` to check, `make style` to fix) and Setup section above (`uv pip install -e ".[dev]"`). Run tests with `python -m pytest tests/<path>` (e.g. `python -m pytest tests/schedulers/test_scheduler_ddpm.py`).
- **Network to the HF Hub works** and many CPU tests plus `diffusers-cli run` download tiny fixture repos (e.g. `hf-internal-testing/tiny-random-*`) on demand. Expect a "sending unauthenticated requests" warning — that's fine; set `HF_TOKEN` only if you hit rate limits.
- **CPU smoke test / hello-world:** `diffusers-cli run --model hf-internal-testing/tiny-stable-diffusion-torch --pipeline-kwargs '{"prompt": "a cat", "num_inference_steps": 5}' --output /tmp/out.png`. Use the `...-torch` repo, not `hf-internal-testing/tiny-stable-diffusion-pipe` (the latter's `model_index.json` references a Flax component and fails to load in this torch-only env).
