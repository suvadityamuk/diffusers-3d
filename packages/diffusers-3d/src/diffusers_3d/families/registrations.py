from __future__ import annotations

from typing import Any


def production_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build exact reviewed execution registrations for released families."""

    from .trellis.registrations import trellis_execution_registrations

    return trellis_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )


__all__ = ["production_execution_registrations"]
