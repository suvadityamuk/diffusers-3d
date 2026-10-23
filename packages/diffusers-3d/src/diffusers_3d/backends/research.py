from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability, BackendSpec, BackendStatus


@dataclass(frozen=True, slots=True)
class ResearchOnlyBackendFacade:
    """Discovery and explicit license opt-in for a research-only backend.

    It never imports the backend; :meth:`require` only selects it in the registry once the caller passes
    ``accept_research_license=True``.
    """

    name: str
    registry: BackendRegistry = BACKEND_REGISTRY

    def status(self) -> BackendStatus:
        return self.registry.status(self.name)

    def require(
        self,
        capability: BackendCapability | str,
        *,
        device: Any = "cuda",
        dtype: Any = "float32",
        differentiable: bool = True,
        accept_research_license: bool = False,
    ) -> BackendSpec:
        if not accept_research_license:
            raise ValueError("accept_research_license=True is required for a research-only backend")
        return self.registry.select(
            capability,
            name=self.name,
            device=device,
            dtype=dtype,
            differentiable=differentiable,
            allow_research_only=True,
        )


class NvdiffrastBackendFacade(ResearchOnlyBackendFacade):
    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        super().__init__("nvdiffrast", registry)


class DiffoctreerastBackendFacade(ResearchOnlyBackendFacade):
    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        super().__init__("diffoctreerast", registry)


class MipGaussianBackendFacade(ResearchOnlyBackendFacade):
    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        super().__init__("mip_gaussian", registry)


__all__ = [
    "DiffoctreerastBackendFacade",
    "MipGaussianBackendFacade",
    "NvdiffrastBackendFacade",
    "ResearchOnlyBackendFacade",
]
