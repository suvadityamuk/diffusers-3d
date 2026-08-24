# Compatibility policy

The core package supports Python 3.10 or newer, PyTorch 2.4 or newer, and the current Diffusers minor release declared
in `pyproject.toml`.

Compatibility is split into five independently tested lanes:

1. **Core CPU**: import, object schemas, registries, loading, and target rejection without optional backends.
2. **Portable**: trimesh, scikit-image, and xatlas adapters on CPU.
3. **Accelerated**: permissively licensed CUDA/JIT backends on explicitly listed Torch/CUDA/OS combinations.
4. **Reference**: source-built or restricted upstream backends used only for numerical parity.
5. **Checkpoint integration**: opt-in tests against real model repositories and their declared hardware.

An accelerated backend is supported only when its `BackendSpec` advertises the running device, dtype, package
version, and build ABI. Availability of an importable module alone is not treated as compatibility.

Schema changes are additive within a minor release. Removing or changing a serialized object, pipeline, recipe, or
training-manifest field requires a migration and a deprecation cycle.
