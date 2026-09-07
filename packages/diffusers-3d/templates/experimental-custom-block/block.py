from __future__ import annotations

import torch
from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, OutputParam, PipelineState


class ExperimentalObject3DBlock(ModularPipelineBlocks):
    """Inference-only remote-code block starter."""

    model_name = "experimental-object3d-block"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Object latents consumed by this experimental inference block.",
            )
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "object_latents",
                type_hint=torch.Tensor,
                description="Object latents emitted for the next inference block.",
            )
        ]

    def __call__(self, components: object, state: PipelineState) -> tuple[object, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.object_latents = block_state.latents
        self.set_block_state(state, block_state)
        return components, state
