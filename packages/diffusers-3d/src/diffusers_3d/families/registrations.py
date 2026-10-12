from __future__ import annotations

from typing import Any


def production_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build exact reviewed execution registrations for released families."""

    from .trellis.registrations import trellis_execution_registrations
    from .trellis2.registrations import trellis2_execution_registrations

    trellis_models, trellis_pipelines = trellis_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )
    trellis2_models, trellis2_pipelines = trellis2_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )
    return (
        trellis_models + trellis2_models,
        trellis_pipelines + trellis2_pipelines,
    )


__all__ = ["production_execution_registrations"]
