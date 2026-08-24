from __future__ import annotations

from typing import Any


def production_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build exact reviewed execution registrations for released families."""

    from .hunyuan3d.registrations import hunyuan3d_execution_registrations

    return hunyuan3d_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )


def production_training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Build exact reviewed training registrations for released families."""

    from .hunyuan3d.registrations import hunyuan3d_training_registrations

    return hunyuan3d_training_registrations(training_registration_type)


__all__ = ["production_execution_registrations", "production_training_registrations"]
