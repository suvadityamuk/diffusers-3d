from __future__ import annotations

from typing import Any


def production_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build exact reviewed execution registrations for released families."""

    from .hunyuan3d.registrations import hunyuan3d_execution_registrations
    from .trellis.registrations import trellis_execution_registrations
    from .trellis2.registrations import trellis2_execution_registrations

    hunyuan_models, hunyuan_pipelines = hunyuan3d_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )
    trellis_models, trellis_pipelines = trellis_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )
    trellis2_models, trellis2_pipelines = trellis2_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )
    return (
        hunyuan_models + trellis_models + trellis2_models,
        hunyuan_pipelines + trellis_pipelines + trellis2_pipelines,
    )


def production_training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Build exact reviewed training registrations for released families."""

    from .hunyuan3d.registrations import hunyuan3d_training_registrations
    from .trellis.registrations import trellis_training_registrations
    from .trellis2.registrations import trellis2_training_registrations

    return (
        hunyuan3d_training_registrations(training_registration_type)
        + trellis_training_registrations(training_registration_type)
        + trellis2_training_registrations(training_registration_type)
    )


__all__ = ["production_execution_registrations", "production_training_registrations"]
