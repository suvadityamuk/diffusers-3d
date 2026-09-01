from __future__ import annotations

from collections.abc import Callable

import pytest

from diffusers_3d import (
    ArtifactLicense3D,
    BackendLicenseClass,
    BackendRequirement3D,
    BackendSupportLevel,
    CheckpointConversion3D,
    ComponentIntegration3D,
    ContributionLevel,
    FineTuneStrategy,
    IntegrationManifest3D,
    LicenseDeclarations3D,
    LicenseRecord3D,
    ParityEvidence3D,
    ParityKind,
    TaskWorkflow3D,
    TrainingRecipeQualification3D,
    UpstreamSource3D,
)


def parity(kind: ParityKind, *, passed: bool = True) -> ParityEvidence3D:
    return ParityEvidence3D(
        kind=kind,
        reference=f"pinned upstream {kind.value}",
        test=f"tests/example/test_parity.py::test_{kind.value}",
        passed=passed,
        atol=1e-4,
        rtol=1e-4,
    )


def component(role: str, class_name: str) -> ComponentIntegration3D:
    return ComponentIntegration3D(
        role=role,
        class_name=class_name,
        checkpoint_conversion=CheckpointConversion3D(
            source_format="reference-state-dict",
            target_format="diffusers-safetensors",
            converter=f"example.conversion.convert_{role.replace('-', '_')}",
            test=f"tests/example/test_conversion.py::test_{role.replace('-', '_')}",
        ),
        parity=(parity(ParityKind.INFERENCE),),
    )


def training() -> TrainingRecipeQualification3D:
    return TrainingRecipeQualification3D(
        recipe_id="example-family",
        recipe_version="1.0",
        recipe_class="example.training.ExampleRecipe",
        target_class="example.pipeline.ExamplePipeline",
        example_class="example.training.ExampleExample",
        batch_class="example.training.ExampleBatch",
        trainer_registration="example.registrations.EXAMPLE_TRAINING_REGISTRATION",
        strategies=(FineTuneStrategy.FULL, FineTuneStrategy.LORA),
        components=("denoiser",),
        backward_parity=parity(ParityKind.BACKWARD),
        checkpoint_parity=parity(ParityKind.CHECKPOINT),
        objective_parity=parity(ParityKind.OBJECTIVE),
    )


@pytest.fixture
def manifest_factory() -> Callable[..., IntegrationManifest3D]:
    def make_manifest(
        *,
        level: ContributionLevel = ContributionLevel.REVIEWED_PACKAGE,
        include_training: bool = True,
    ) -> IntegrationManifest3D:
        license_record = LicenseRecord3D(
            identifier="Apache-2.0",
            classification=BackendLicenseClass.PERMISSIVE,
            url="https://www.apache.org/licenses/LICENSE-2.0",
        )
        return IntegrationManifest3D.create(
            integration_id="example-family",
            level=level,
            upstream=UpstreamSource3D(
                repository="https://github.com/example/example-family",
                revision="a" * 40,
            ),
            components=(
                component("denoiser", "example.model.ExampleDenoiser"),
                component("pipeline", "example.pipeline.ExamplePipeline"),
            ),
            workflow=TaskWorkflow3D(
                task_ids=("image-to-3d",),
                workflow="standard-pipeline",
                input_representations=("image",),
                output_representations=("triangle-mesh",),
            ),
            backends=(
                BackendRequirement3D(
                    name="torch",
                    distribution="torch",
                    version="2.4",
                    capabilities=("tensor-compute",),
                    support_level=BackendSupportLevel.PORTABLE,
                    license_identifier="BSD-3-Clause",
                    license_class=BackendLicenseClass.PERMISSIVE,
                    required=True,
                    install_hint="Install diffusers-3d.",
                    source=None,
                ),
            ),
            licenses=LicenseDeclarations3D(
                model=license_record,
                artifacts=(
                    ArtifactLicense3D(
                        artifact="converted-checkpoints",
                        license=license_record,
                    ),
                ),
            ),
            training=training() if include_training else None,
        )

    return make_manifest
