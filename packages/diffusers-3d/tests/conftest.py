from __future__ import annotations

import json

import pytest
from transformers import CLIPTokenizer


@pytest.fixture(scope="session")
def tiny_clip_tokenizer(tmp_path_factory) -> CLIPTokenizer:
    """A real byte-level CLIP BPE tokenizer over a-z with a few merges, small enough for the tiny text encoder."""

    root = tmp_path_factory.mktemp("tiny-clip-tokenizer")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for code in range(ord("a"), ord("z") + 1):
        vocab[chr(code)] = len(vocab)
        vocab[chr(code) + "</w>"] = len(vocab)
    for token in ("ch", "re", "red</w>", "ai", "air</w>", "chair</w>"):
        vocab[token] = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\nc h\nr e\nre d</w>\na i\nai r</w>\nch air</w>\n")
    return CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"))


@pytest.fixture
def make_diffusers_lora():
    """Build a Diffusers-format LoRA state dict for named pipeline components, keyed ``<component>.<module>...``."""

    peft = pytest.importorskip("peft")
    import copy

    import torch
    from peft.utils import get_peft_model_state_dict

    def make(pipeline, component_names, *, rank: int = 2, alpha: int | None = None, seed: int = 0):
        state_dict, donors = {}, {}
        for name in component_names:
            donor = copy.deepcopy(getattr(pipeline, name))
            torch.manual_seed(seed)
            donor.add_adapter(
                peft.LoraConfig(
                    r=rank,
                    lora_alpha=rank if alpha is None else alpha,
                    target_modules=["to_qkv", "to_q", "to_kv", "to_out"],
                    init_lora_weights=False,
                )
            )
            donors[name] = donor
            for key, value in get_peft_model_state_dict(donor).items():
                state_dict[f"{name}.{key}"] = value
        return state_dict, donors

    return make


@pytest.fixture
def slab_ovoxel():
    """One layer of 8x8 voxels whose z edges all cross the surface: a flat grid of quads with a linear colour ramp."""

    import torch

    from diffusers_3d import CoordinateSystem, OVoxelAsset

    xs, ys = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    coordinates = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.full((64,), 4)], dim=1)
    ramp = coordinates[:, :2].float() / 7
    flags = torch.zeros(64, 3, dtype=torch.uint8)
    flags[:, 2] = 1
    return OVoxelAsset(
        active_coordinates=coordinates,
        base_color=torch.cat([ramp, torch.full((64, 1), 0.25)], dim=1),
        metallic=ramp[:, 0],
        roughness=ramp[:, 1],
        opacity=torch.ones(64),
        dual_grid_vertex_offsets=torch.full((64, 3), 0.5),
        intersection_data=flags,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        metadata={"resolution": 8, "aabb": [[-0.5] * 3, [0.5] * 3]},
    )
