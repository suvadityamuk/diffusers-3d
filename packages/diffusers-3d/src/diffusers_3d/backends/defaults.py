from __future__ import annotations

from .registry import BackendRegistry
from .types import BackendCapability, BackendLicenseClass, BackendSpec, BackendSupportLevel

_UTILS3D_REVISION = "9a4eb15e4021b67b12c460c7057d642626897ec8"
_FLEX_GEMM_REVISION = "6dd94a859c26ee8246888502eada3dd8ad85532e"
_CUMESH_REVISION = "12289e1062f0603f2f0d0771b02e1395d247f26f"
_NVDIFFRAST_REVISION = "253ac4fcea7de5f396371124af597e6cc957bfae"
_NVDIFFREC_REVISION = "a3e73909a01887c8a135235ff860dd23a045cc1b"
_DIFFOCTREERAST_REVISION = "b09c20b84ec3aace4729e6e18a613112320eca3a"
_MIP_SPLATTING_REVISION = "dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"


DEFAULT_BACKEND_SPECS = (
    BackendSpec(
        name="trimesh",
        import_names=("trimesh",),
        distribution_names=("trimesh",),
        capabilities=frozenset(
            {
                BackendCapability.GEOMETRY_PROCESSING,
                BackendCapability.SERIALIZATION,
                BackendCapability.CONVERSION,
            }
        ),
        support_level=BackendSupportLevel.PORTABLE,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32", "float64"}),
        differentiable=False,
        install_hint='Install the portable backend with `pip install "diffusers-3d[portable]"`',
    ),
    BackendSpec(
        name="scikit-image",
        import_names=("skimage",),
        distribution_names=("scikit-image",),
        capabilities=frozenset({BackendCapability.SURFACE_EXTRACTION}),
        support_level=BackendSupportLevel.PORTABLE,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32", "float64"}),
        differentiable=False,
        install_hint='Install the portable backend with `pip install "diffusers-3d[portable]"`',
    ),
    BackendSpec(
        name="xatlas",
        import_names=("xatlas",),
        distribution_names=("xatlas",),
        capabilities=frozenset({BackendCapability.GEOMETRY_PROCESSING}),
        support_level=BackendSupportLevel.PORTABLE,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32"}),
        differentiable=False,
        install_hint='Install the portable backend with `pip install "diffusers-3d[portable]"`',
    ),
    BackendSpec(
        name="utils3d",
        import_names=("utils3d",),
        distribution_names=("utils3d",),
        capabilities=frozenset(
            {
                BackendCapability.GEOMETRY_PROCESSING,
                BackendCapability.MESH_RASTERIZATION,
                BackendCapability.CONVERSION,
            }
        ),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu", "cuda"}),
        dtypes=frozenset({"float16", "float32", "float64"}),
        differentiable=True,
        install_hint=(
            "Install the EasternJournalist source, not the colliding PyPI project: "
            f"`pip install git+https://github.com/EasternJournalist/utils3d.git@{_UTILS3D_REVISION}`"
        ),
        tested_build=f"EasternJournalist/utils3d at {_UTILS3D_REVISION}",
        source_url="https://github.com/EasternJournalist/utils3d.git",
        source_revision=_UTILS3D_REVISION,
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="gsplat",
        import_names=("gsplat",),
        distribution_names=("gsplat",),
        capabilities=frozenset({BackendCapability.GAUSSIAN_RASTERIZATION}),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float16", "float32"}),
        differentiable=True,
        install_hint='Install the Gaussian backend with `pip install "diffusers-3d[gaussian]"`',
    ),
    BackendSpec(
        name="spconv",
        import_names=("spconv",),
        distribution_names=(
            "spconv",
            "spconv-cu118",
            "spconv-cu120",
            "spconv-cu121",
            "spconv-cu124",
        ),
        capabilities=frozenset({BackendCapability.SPARSE_COMPUTE}),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float16", "float32"}),
        differentiable=True,
        install_hint=(
            "Install the official spconv wheel whose distribution suffix matches the environment CUDA runtime"
        ),
        tested_build="A CUDA-matched official spconv wheel is required",
        source_url="https://github.com/traveller59/spconv",
    ),
    BackendSpec(
        name="flex_gemm",
        import_names=("flex_gemm",),
        distribution_names=("flex_gemm",),
        capabilities=frozenset({BackendCapability.SPARSE_COMPUTE}),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cuda", "rocm"}),
        dtypes=frozenset({"float16", "bfloat16", "float32"}),
        differentiable=True,
        install_hint=(
            f"Build FlexGEMM from JeffreyXiang/FlexGEMM revision {_FLEX_GEMM_REVISION} "
            "with a compatible Triton toolchain"
        ),
        tested_build="Source build; PyTorch and Triton versions must be recorded together",
        source_url="https://github.com/JeffreyXiang/FlexGEMM.git",
        source_revision=_FLEX_GEMM_REVISION,
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="cumesh",
        import_names=("cumesh",),
        distribution_names=("cumesh",),
        capabilities=frozenset(
            {
                BackendCapability.GEOMETRY_PROCESSING,
                BackendCapability.CONVERSION,
            }
        ),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
        differentiable=False,
        install_hint=(
            f"Build CuMesh from JeffreyXiang/CuMesh revision {_CUMESH_REVISION} against the active PyTorch and CUDA"
        ),
        tested_build="Source build; PyTorch, CUDA, compiler, and GPU architecture must be recorded",
        source_url="https://github.com/JeffreyXiang/CuMesh.git",
        source_revision=_CUMESH_REVISION,
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="kaolin",
        import_names=("kaolin",),
        distribution_names=("kaolin",),
        capabilities=frozenset(
            {
                BackendCapability.MESH_RASTERIZATION,
                BackendCapability.SURFACE_EXTRACTION,
                BackendCapability.GEOMETRY_PROCESSING,
            }
        ),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float16", "float32", "float64"}),
        differentiable=True,
        install_hint=(
            "Install an official Kaolin wheel matching PyTorch and CUDA; adapters are limited to Apache-2.0 modules "
            "and must not use kaolin.non_commercial"
        ),
        tested_build="Only permissively licensed Kaolin modules are eligible",
        source_url="https://github.com/NVIDIAGameWorks/kaolin",
    ),
    BackendSpec(
        name="o_voxel",
        import_names=("o_voxel",),
        distribution_names=("o_voxel",),
        capabilities=frozenset(
            {
                BackendCapability.NATIVE_REPRESENTATION,
                BackendCapability.GEOMETRY_PROCESSING,
                BackendCapability.SERIALIZATION,
                BackendCapability.CONVERSION,
            }
        ),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float16", "float32"}),
        differentiable=False,
        install_hint=(
            "Build the o-voxel source package from an audited pinned microsoft/TRELLIS.2 revision with "
            "`--no-build-isolation`"
        ),
        tested_build="Source build coupled to pinned CuMesh and FlexGEMM builds",
        source_url="https://github.com/microsoft/TRELLIS.2.git",
        source_revision="75fbf0183001ed9876c8dbb35de6b68552ee08bd",
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="nvdiffrast",
        import_names=("nvdiffrast",),
        distribution_names=("nvdiffrast",),
        capabilities=frozenset({BackendCapability.MESH_RASTERIZATION}),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        license_class=BackendLicenseClass.RESTRICTED,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
        differentiable=True,
        install_hint=(
            f"After license review, build NVlabs/nvdiffrast revision {_NVDIFFRAST_REVISION} "
            "from source against the active PyTorch and CUDA"
        ),
        tested_version="0.4.0",
        tested_build="CUDA source build; non-commercial research/evaluation use",
        source_url="https://github.com/NVlabs/nvdiffrast.git",
        source_revision=_NVDIFFRAST_REVISION,
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="nvdiffrec_render",
        import_names=("nvdiffrec_render",),
        distribution_names=("nvdiffrec_render",),
        capabilities=frozenset(
            {
                BackendCapability.MESH_RASTERIZATION,
                BackendCapability.PBR_BAKING,
                BackendCapability.FIELD_RENDERING,
            }
        ),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        license_class=BackendLicenseClass.RESTRICTED,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
        differentiable=True,
        install_hint=(
            f"After license review, package the JeffreyXiang/nvdiffrec renderutils fork at {_NVDIFFREC_REVISION} "
            "with direct-URL provenance"
        ),
        tested_build="CUDA source build; non-commercial research/evaluation use",
        source_url="https://github.com/JeffreyXiang/nvdiffrec.git",
        source_revision=_NVDIFFREC_REVISION,
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="diffoctreerast",
        import_names=("diffoctreerast",),
        distribution_names=("diffoctreerast",),
        capabilities=frozenset(
            {
                BackendCapability.GAUSSIAN_RASTERIZATION,
                BackendCapability.FIELD_RENDERING,
            }
        ),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        license_class=BackendLicenseClass.RESTRICTED,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
        differentiable=True,
        install_hint=(f"After license review, build JeffreyXiang/diffoctreerast revision {_DIFFOCTREERAST_REVISION}"),
        tested_build="CUDA source build; non-commercial research/evaluation use",
        source_url="https://github.com/JeffreyXiang/diffoctreerast.git",
        source_revision=_DIFFOCTREERAST_REVISION,
        requires_source_provenance=True,
    ),
    BackendSpec(
        name="mip_gaussian",
        import_names=("diff_gaussian_rasterization",),
        distribution_names=("diff_gaussian_rasterization",),
        capabilities=frozenset({BackendCapability.GAUSSIAN_RASTERIZATION}),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        license_class=BackendLicenseClass.RESTRICTED,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
        differentiable=True,
        install_hint=(
            "After license review, build the mip-splatting diff-gaussian-rasterization submodule at revision "
            f"{_MIP_SPLATTING_REVISION}"
        ),
        tested_build="CUDA source build; non-commercial research/evaluation use",
        source_url="https://github.com/autonomousvision/mip-splatting.git",
        source_revision=_MIP_SPLATTING_REVISION,
        requires_source_provenance=True,
    ),
)


def create_default_backend_registry() -> BackendRegistry:
    """Create an independent mutable registry populated with planned backends."""

    return BackendRegistry(DEFAULT_BACKEND_SPECS)


BACKEND_REGISTRY = create_default_backend_registry().freeze()


__all__ = ["BACKEND_REGISTRY", "DEFAULT_BACKEND_SPECS", "create_default_backend_registry"]
