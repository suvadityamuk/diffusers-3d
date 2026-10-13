from __future__ import annotations

from typing import Any


def production_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build exact reviewed execution registrations for released families."""

    from .hunyuan3d.registrations import hunyuan3d_execution_registrations
    from .pixal3d.registrations import pixal3d_execution_registrations
    from .sam3d_objects.registrations import sam3d_objects_execution_registrations
    from .step1x3d.registrations import step1x3d_execution_registrations
    from .trellis.registrations import trellis_execution_registrations
    from .trellis2.registrations import trellis2_execution_registrations
    from .triposg.registrations import triposg_execution_registrations

    models, pipelines = (), ()
    for register in (
        trellis_execution_registrations,
        trellis2_execution_registrations,
        triposg_execution_registrations,
        sam3d_objects_execution_registrations,
        step1x3d_execution_registrations,
        pixal3d_execution_registrations,
        hunyuan3d_execution_registrations,
    ):
        family_models, family_pipelines = register(model_registration_type, pipeline_registration_type)
        models += family_models
        pipelines += family_pipelines
    return models, pipelines


__all__ = ["production_execution_registrations"]
