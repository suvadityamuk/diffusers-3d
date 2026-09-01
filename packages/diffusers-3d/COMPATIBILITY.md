# Compatibility policy

The core package supports Python 3.10 or newer, PyTorch 2.4 or newer,
Accelerate 1.1.0 or newer, Transformers 5.5.0 or newer, and the current
Diffusers minor release declared in `pyproject.toml`. DINOv3 first appeared in Transformers 4.56.0, but that
line is incompatible with the required Hugging Face Hub 1.x stack and
predates security fixes included in the 5.5.0 floor.

Compatibility is split into five independently tested lanes:

1. **Core CPU**: import, object schemas, registries, loading, and target rejection without optional backends.
2. **Portable**: trimesh, scikit-image, and xatlas adapters on CPU.
3. **Accelerated**: permissively licensed CUDA/JIT backends on explicitly listed Torch/CUDA/OS combinations.
4. **Reference**: source-built or restricted upstream backends used only for numerical parity.
5. **Checkpoint integration**: opt-in tests against real model repositories and their declared hardware.

An accelerated backend is supported only when its `BackendSpec` advertises the running device, dtype, package
version, and build ABI. Source backends additionally require exact PEP 610 repository/revision provenance before
import, followed by runtime API and Torch/CUDA/dtype/Triton checks. Upstream modules are not required to invent
non-upstream revision or build attributes; optional version/build strings are diagnostic only. Availability of an
importable module alone is not treated as compatibility.

Training manifest schema 4 adds the effective LoRA adapter seed to strategy identity. Schema-3 training checkpoints
must be recreated because exact continuation cannot infer whether their adapter initialization matches. Removing or
changing other serialized object, pipeline, recipe, or training-manifest fields requires an explicit migration.
