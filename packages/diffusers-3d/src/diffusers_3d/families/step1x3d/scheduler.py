from __future__ import annotations

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SchedulerMixin,  # noqa: F401 - required by external-component loading
)


class Step1X3DFlowMatchEulerScheduler(FlowMatchEulerDiscreteScheduler):
    """The stock flow-matching Euler scheduler, under the exact class name that Step1X-3D pipelines register.

    The released ``scheduler_config.json`` (``shift=1.0``, no dynamic shifting) is used unchanged.
    """


__all__ = ["Step1X3DFlowMatchEulerScheduler"]
