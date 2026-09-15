# Fine-tuning with diffusers-3d

Training in `diffusers-3d` is recipe-gated. Instead of a generic "train any `nn.Module` on any dataset" loop, each
model stage ships a reviewed `TrainingRecipe3D` that fixes the objective, the example type, which components may be
trained, and how checkpoints are written. `Object3DTrainer` runs whatever recipe you hand it on top of Accelerate.
This keeps a fine-tune reproducible against the upstream training code and makes the resulting checkpoint
self-describing.

Everything below runs on CPU with tiny components in a few seconds; swap in a converted checkpoint and a real
dataset for actual training.

## The pieces

| Object | Role |
|---|---|
| `TrainingRecipe3D` subclass (e.g. `Trellis2SparseStructureFlowRecipe`) | Owns the objective, collation, component policies, and weight saving for one model stage. Wraps the pipeline being trained. |
| Example type (e.g. `Trellis2SparseStructureExample`) | Typed dataclass your dataset must return. Validated on construction. |
| `Object3DDataset` | Any map-style dataset (`__len__`, `__getitem__`) yielding the recipe's example type. A protocol, not a base class. |
| `FullFineTune` / `LoRAFineTune` | Which components to train and how. Component keys must be approved by the recipe. |
| `TrainingConfig3D` | Optimizer, schedule, batch, precision, seed, output directory, and provenance (`base_model`, `revision`, `dataset_fingerprint`). |
| `Object3DTrainer` | `prepare()`, `train()`, `save_checkpoint()`, `load_checkpoint()`. |
| `TrainingManifest3D` | JSON written next to every checkpoint recording recipe, strategy, config, package versions, and a hash of trainable parameter names. |

## Registered recipes

| Recipe | Family | Trains | Frozen | Strategies | Example type |
|---|---|---|---|---|---|
| `Trellis2SparseStructureFlowRecipe` | TRELLIS.2 | `sparse_structure_flow_model` | conditioner, decoder | full | `Trellis2SparseStructureExample` |
| `Trellis2ShapeSLatFlowRecipe` | TRELLIS.2 | `shape_slat_flow_model` | conditioner | full | `Trellis2SLatExample` |
| `Trellis2TextureSLatFlowRecipe` | TRELLIS.2 | `texture_slat_flow_model` | conditioner | full | `Trellis2TextureSLatExample` |
| `TrellisSparseStructureFlowRecipe` | TRELLIS | `sparse_structure_flow_model` | conditioner, decoder | full | `TrellisSparseStructureExample` |
| `TrellisSLatFlowRecipe` | TRELLIS | `slat_flow_model` | conditioner | full | `TrellisSLatExample` |

The sparse-structure recipes train the first-stage flow model on precomputed dense latents; the SLAT recipes train
the second-stage flow models on precomputed sparse latents (`TrellisSparseTensor` coordinates and features). No LoRA
recipe is registered yet; passing `LoRAFineTune` to any recipe raises `TrainingPolicyError`. The authoritative list is
each family's `registrations.py` (`trellis_training_registrations`, `trellis2_training_registrations`).

## 1. Prepare data

The reviewed recipes consume precomputed sparse-structure latents (the SS-VAE encoder output from the upstream data
pipeline), paired with the conditioning image. Encoding raw meshes to latents is outside this package; run the
upstream TRELLIS/TRELLIS.2 dataset tooling and store the results.

```python
import torch
from diffusers_3d import ImageCondition, Trellis2SparseStructureExample

class LatentDataset:
    """Any map-style dataset returning the recipe's example type."""

    def __init__(self, records):
        self.records = records  # e.g. list of (image_path, latent_path, id)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, latent_path, example_id = self.records[index]
        image = load_rgba_tensor(image_path)                     # (4, H, W) float in [0, 1]
        latents = torch.load(latent_path)                         # (C, D, H, W) float
        return Trellis2SparseStructureExample(
            condition=ImageCondition(image=image),
            sparse_structure_latents=latents,
            example_id=example_id,
        )
```

`ImageCondition` accepts RGB, RGBA, or RGB plus a separate `mask`. The recipe's collator applies the pinned training
transform once: crop to the alpha bounding box, LANCZOS resize, premultiply RGB by alpha. Examples validate their
shapes on construction, so malformed records fail at dataset time rather than mid-step.

## 2. Load the pipeline and pick a recipe

```python
from diffusers_3d import AutoPipelineForImageTo3D, Trellis2SparseStructureFlowRecipe

pipeline = AutoPipelineForImageTo3D.from_pretrained("/path/to/trellis2")
recipe = Trellis2SparseStructureFlowRecipe(pipeline)
```

The recipe checks that the pipeline is the exact type it was reviewed for and that the frozen components have the
expected classes. Hyperparameters of the objective are constructor arguments with released defaults
(`sigma_min=1e-5`, `timestep_mean=1.0`, `timestep_std=1.0`, `p_uncond=0.1`); the objective is

```
t      = sigmoid(N(timestep_mean, timestep_std))
x_t    = (1 - t) * x0 + (sigma_min + (1 - sigma_min) * t) * noise
target = (1 - sigma_min) * noise - x0
loss   = MSE(model(x_t, t * 1000, image_tokens), target)
```

with conditioning dropped with probability `p_uncond`.

## 3. Choose what to train

```python
from diffusers_3d import FullFineTune

strategy = FullFineTune(("sparse_structure_flow_model",))
```

Component keys are the pipeline's constructor argument names. The recipe's `ComponentPolicy` list decides which keys
and which strategies are legal; anything else is a `TrainingPolicyError` before any parameter is touched:

```
FullFineTune(("conditioner",))                      -> Unknown recipe component keys: conditioner
LoRAFineTune(("sparse_structure_flow_model",), 4)   -> Component 'sparse_structure_flow_model' does not support lora fine-tuning
```

Frozen components are moved to the training device, put in eval mode, and have `requires_grad` disabled.

## 4. Configure and train

```python
from pathlib import Path
from diffusers_3d import Object3DTrainer, TrainingConfig3D

config = TrainingConfig3D(
    base_model="/path/to/trellis2",          # provenance, stored in the manifest
    revision=None,
    dataset_fingerprint="my-latents-v1",     # required for exact checkpoint resume
    output_dir=Path("runs/trellis2-ss"),
    train_batch_size=8,
    gradient_accumulation_steps=1,
    max_train_steps=2000,
    learning_rate=1e-4,
    weight_decay=0.0,
    lr_scheduler="constant",                 # any Diffusers SchedulerType name
    lr_warmup_steps=0,
    max_grad_norm=1.0,
    mixed_precision="bf16",                  # "no" | "fp16" | "bf16"
    seed=0,
)

trainer = Object3DTrainer(recipe, LatentDataset(records), strategy, config)
summary = trainer.prepare().train()
print(summary.final_loss, summary.final_metrics, summary.optimizer_steps)
```

`prepare()` builds the Accelerator, optimizer, LR scheduler, and dataloader, freezes non-selected components, and
records the manifest. `train()` runs to `max_train_steps` (or `train(max_optimizer_steps=n)` for a shorter burst)
and returns a `TrainingSummary3D` with the final loss, recipe metrics (`flow_matching_mse`, `mean_timestep`,
`condition_dropout_fraction` for this recipe), and step counts. `trainer.train_step(batch)` is available if you want
your own loop.

Multi-GPU runs go through Accelerate as usual (`accelerate launch train.py`). Set `cpu=True` for CPU-only smoke runs.

## 5. Checkpoints

```python
manifest_path = trainer.save_checkpoint()      # -> runs/trellis2-ss/diffusers_3d_training.json
```

The checkpoint directory contains:

```
runs/trellis2-ss/
├── accelerator_state/                       # optimizer, scheduler, RNG, trainable weights
├── sparse_structure_flow_model/             # trained component in Diffusers layout
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── diffusers_3d_training.json               # TrainingManifest3D (schema 5)
```

Resuming requires an identical setup: same recipe and version, strategy, component configs, dataset fingerprint,
and package versions. `validate_resume()` compares the on-disk manifest against the live trainer and raises with the
first mismatch; `load_checkpoint()` restores state, including frozen-component weights:

```python
trainer = Object3DTrainer(recipe, dataset, strategy, config).prepare()
trainer.validate_resume("runs/trellis2-ss")
trainer.load_checkpoint("runs/trellis2-ss")
trainer.train()
```

Exact checkpoint continuation is single-process only, needs `dataloader_num_workers=0`, and loads Accelerate/PyTorch
state in weights-only mode.

## 6. Use the fine-tuned model

The trainer mutates the pipeline it was given, so after training the pipeline is ready for inference or for saving
as a complete, reloadable pipeline:

```python
pipeline.save_pretrained("checkpoints/trellis2-finetuned")

from diffusers_3d import AutoPipelineForImageTo3D
tuned = AutoPipelineForImageTo3D.from_pretrained("checkpoints/trellis2-finetuned")
output = tuned(condition, sparse_structure_sampler_params={"steps": 12})
```

`save_pretrained` writes the Diffusers layout plus the `object3d_model_index.json` sidecar, so the result loads
through the auto-loader exactly like a converted upstream checkpoint. Alternatively, load only the trained component
from the checkpoint directory into an existing pipeline:

```python
from diffusers_3d import Trellis2SparseStructureFlowModel

pipeline.sparse_structure_flow_model = Trellis2SparseStructureFlowModel.from_pretrained(
    "runs/trellis2-ss", subfolder="sparse_structure_flow_model"
)
```

See [inference.md](inference.md) for what to do with the pipeline from here.

## Smoke-testing without a checkpoint

Every TRELLIS.2 model class has a `tiny_config()`. `build_tiny_pipeline()` in the examples package assembles a
CPU-runnable pipeline from them, which is the fastest way to validate a dataset class or training script before
committing GPU time:

```python
from diffusers_3d.families.trellis2.examples.tiny_components import build_tiny_pipeline

pipeline = build_tiny_pipeline(include_slat=False)  # include_slat=True adds the SLAT and O-Voxel stages
recipe = Trellis2SparseStructureFlowRecipe(pipeline)
# tiny latents are (2, 2, 2, 2); tiny conditioners take any image size and resize to 8x8
```

## Adding a recipe

A new trainable stage needs a `TrainingRecipe3D` subclass with `component_policies`,
`frozen_component_policies`, `collate`, `compute_loss`, and `save_weights`, plus a registration in the family's
`registrations.py`. Registration is the review boundary: the production registry is frozen at import and only
contains reviewed families. [contributions.md](contributions.md) describes the evidence required to promote a recipe
from experimental to registered.
