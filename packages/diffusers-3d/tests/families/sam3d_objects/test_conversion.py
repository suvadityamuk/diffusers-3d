from __future__ import annotations

import re
import warnings

import pytest
import torch
import yaml
from safetensors.torch import save_file

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    Sam3DConditionEmbedder,
    Sam3DSparseStructureFlowModel,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSparseStructureDecoder,
    convert_sam3d_checkpoint,
)
from diffusers_3d.families.sam3d_objects import conversion

from .conftest import tiny_condition, tiny_dinov2_config, tiny_sam3d_pipeline

pytestmark = pytest.mark.integration

_UPSTREAM = {"rotation": "6drotation_normalized"}


def _upstream_flow_state(model: Sam3DSparseStructureFlowModel) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in model.state_dict().items():
        key = re.sub(r"self_attn\.streams\.(\w+)\.(\w+)\.", r"self_attn.\2.\1.", key)
        key = key.replace(".rotation.", ".6drotation_normalized.").replace(".pose", ".6drotation_normalized")
        state[f"_base_models.generator.reverse_fn.backbone.{key}"] = value
    return state


def _upstream_dinov2(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Transformers 5.18 renamed the attention projections; map them back to the Hub layout first.
    for new, old in (("q_proj", "attention.query"), ("k_proj", "attention.key"), ("v_proj", "attention.value")):
        state = {key.replace(f".attention.{new}.", f".attention.{old}."): value for key, value in state.items()}
    state = {key.replace(".attention.o_proj.", ".attention.output.dense."): value for key, value in state.items()}
    renames = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "register_tokens",
        "embeddings.position_embeddings": "pos_embed",
        "embeddings.patch_embeddings.projection.": "patch_embed.proj.",
        "layernorm.": "norm.",
        "attention.output.dense.": "attn.proj.",
        "layer_scale1.lambda1": "ls1.gamma",
        "layer_scale2.lambda1": "ls2.gamma",
        "encoder.layer.": "blocks.",
    }
    converted = {}
    for key, value in state.items():
        if ".attention.attention." in key:
            if ".query." in key:
                parts = [state[key.replace(".query.", f".{name}.")] for name in ("query", "key", "value")]
                converted[
                    re.sub(r"attention\.attention\.query\.", "attn.qkv.", key).replace("encoder.layer.", "blocks.")
                ] = torch.cat(parts)
            continue
        for old, new in renames.items():
            key = key.replace(old, new)
        converted[key] = value
    return converted


def _upstream_embedder_state(pipeline, embedder: Sam3DConditionEmbedder, prefix: str) -> dict[str, torch.Tensor]:
    state = embedder.state_dict()
    upstream = {"idx_emb": state["idx_emb"]}
    # Upstream keeps a copy of the frozen DINOv2 for the image and for the mask.
    dinov2 = _upstream_dinov2(pipeline.image_encoder.state_dict())
    for index in (0, 1):
        upstream |= {f"module_list.{index}.backbone.{key}": value.clone() for key, value in dinov2.items()}
    for key, value in state.items():
        if key.startswith("pointmap_encoder.block.attn1.to_q."):
            suffix = key.rsplit(".", 1)[1]
            parts = [state[f"pointmap_encoder.block.attn1.to_{name}.{suffix}"] for name in "qkv"]
            upstream[f"module_list.2.blocks.0.attn.qkv.{suffix}"] = torch.cat(parts)
        elif key.startswith("pointmap_encoder.") and not re.match(r"pointmap_encoder\.block\.attn1\.to_[kv]\.", key):
            key = key.removeprefix("pointmap_encoder.")
            for old, new in (
                ("block.norm1.", "blocks.0.norm1."),
                ("block.norm3.", "blocks.0.norm2."),
                ("block.ff.net.0.proj.", "blocks.0.mlp.fc1."),
                ("block.ff.net.2.", "blocks.0.mlp.fc2."),
                ("block.attn1.to_out.0.", "blocks.0.attn.proj."),
            ):
                key = key.replace(old, new)
            upstream[f"module_list.2.{key}"] = value
        elif key.startswith("projections."):
            _, index, rest = key.split(".", 2)
            if rest == "1.net.0.proj.weight":
                w3, w1 = value.chunk(2)
                upstream |= {f"projection_nets.{index}.1.w1.weight": w1, f"projection_nets.{index}.1.w3.weight": w3}
            elif rest == "1.net.2.weight":
                upstream[f"projection_nets.{index}.1.w2.weight"] = value
            else:
                upstream[f"projection_nets.{index}.{rest}"] = value
    return {f"{prefix}{key}": value for key, value in upstream.items()}


@pytest.fixture
def tiny_production(monkeypatch):
    """Point every ``production_config`` the converter checks and builds at the tiny test configurations."""

    for model_type in (
        Sam3DSparseStructureFlowModel,
        TrellisSparseStructureDecoder,
        TrellisSLatFlowModel,
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
    ):
        monkeypatch.setattr(model_type, "production_config", classmethod(lambda cls: cls.tiny_config()))
    tiny_embedder = Sam3DConditionEmbedder.tiny_config()
    monkeypatch.setattr(
        Sam3DConditionEmbedder,
        "production_config",
        classmethod(
            lambda cls, *, prenorm_features=False, pointmap=True: {
                **tiny_embedder,
                "prenorm_features": prenorm_features,
                "pointmap_config": tiny_embedder["pointmap_config"] if pointmap else None,
            }
        ),
    )
    monkeypatch.setattr(
        Sam3DConditionEmbedder, "production_pointmap_config", staticmethod(lambda: tiny_embedder["pointmap_config"])
    )
    monkeypatch.setattr(conversion, "_dinov2_config", tiny_dinov2_config)
    representation = TrellisSLatGaussianDecoder.tiny_config()["representation_config"]
    monkeypatch.setattr(
        TrellisSLatGaussianDecoder, "production_representation_config", staticmethod(lambda: representation)
    )


def _write_release(root, pipeline) -> None:
    flow = pipeline.sparse_structure_flow_model
    embedder = pipeline.sparse_structure_conditioner
    pointmap = embedder.config.pointmap_config
    dinov2 = {"dino_model": "dinov2_vitl14_reg", "input_size": pipeline.image_encoder.config.image_size}
    ss_generator = {
        "module": {
            "generator": {
                "backbone": {
                    "reverse_fn": {
                        "backbone": {
                            **{
                                key: flow.config[key]
                                for key in ("resolution", "in_channels", "model_channels", "num_blocks")
                            },
                            "latent_mapping": {
                                _UPSTREAM.get(name, name): {"in_channels": shape[-1]}
                                for name, shape in flow.latent_shapes(1).items()
                            },
                            "latent_share_transformer": {
                                "6drotation_normalized": [_UPSTREAM.get(name, name) for name in flow.pose_names]
                            },
                        }
                    }
                }
            },
            "condition_embedder": {
                "backbone": {
                    "embedder_list": [
                        [dinov2, []],
                        [dinov2, []],
                        [
                            {
                                **{key: pointmap[key] for key in ("embed_dim", "input_size", "patch_size")},
                                "remap_output": "linear",
                            },
                            [],
                        ],
                    ]
                }
            },
        }
    }
    slat_generator = {
        "module": {
            "generator": {"backbone": {"reverse_fn": {"backbone": dict(pipeline.slat_flow_model.config)}}},
            "condition_embedder": {"backbone": {"embedder_list": [[{**dinov2, "prenorm_features": True}, []]] * 2}},
        }
    }
    configs = {
        "pipeline.yaml": {
            "pose_decoder_name": "ScaleShiftInvariant",
            "slat_cfg_strength": 1,
            "slat_rescale_t": 1,
            "downsample_ss_dist": 1,
            "slat_mean": pipeline.config.slat_mean,
            "slat_std": pipeline.config.slat_std,
        },
        "ss_generator.yaml": ss_generator,
        "slat_generator.yaml": slat_generator,
        "ss_decoder.yaml": dict(pipeline.sparse_structure_decoder.config),
        "slat_decoder_gs.yaml": dict(pipeline.gaussian_decoder.config),
        "slat_decoder_mesh.yaml": dict(pipeline.mesh_decoder.config),
    }
    for name, config in configs.items():
        (root / name).write_text(yaml.safe_dump(yaml.safe_load(yaml.safe_dump(dict(config), default_flow_style=None))))

    ss_state = _upstream_flow_state(flow) | _upstream_embedder_state(
        pipeline, embedder, "_base_models.condition_embedder."
    )
    save_file({key: value.contiguous() for key, value in ss_state.items()}, root / "ss_generator.safetensors")
    slat_state = {
        f"_base_models.generator.reverse_fn.backbone.{key}": value
        for key, value in pipeline.slat_flow_model.state_dict().items()
    } | _upstream_embedder_state(pipeline, pipeline.slat_conditioner, "_base_models.condition_embedder.")
    torch.save({"state_dict": slat_state, "epoch": 3}, root / "slat_generator.ckpt")
    for name, component in (
        ("ss_decoder", pipeline.sparse_structure_decoder),
        ("slat_decoder_gs", pipeline.gaussian_decoder),
        ("slat_decoder_mesh", pipeline.mesh_decoder),
    ):
        torch.save(component.state_dict(), root / f"{name}.ckpt")


def test_released_layout_converts_and_reproduces_the_pipeline(tmp_path, tiny_production):
    pipeline = tiny_sam3d_pipeline()
    source = tmp_path / "checkpoints"
    source.mkdir()
    _write_release(source, pipeline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        convert_sam3d_checkpoint(source, tmp_path / "converted")
        converted = AutoPipelineForImageTo3D.from_pretrained(tmp_path / "converted", local_files_only=True)
    assert (tmp_path / "converted" / "object3d_model_index.json").is_file()
    assert converted.sparse_structure_scheduler.config.sigma_min == 0.0

    condition, pointmap = tiny_condition()
    arguments = {
        "formats": ("gaussian", "mesh"),
        "sparse_structure_num_inference_steps": 2,
        "slat_num_inference_steps": 2,
    }
    expected = pipeline(condition, pointmap=pointmap, generator=torch.Generator().manual_seed(0), **arguments)
    actual = converted(condition, pointmap=pointmap, generator=torch.Generator().manual_seed(0), **arguments)
    torch.testing.assert_close(actual.objects[0].means, expected.objects[0].means)
    torch.testing.assert_close(actual.objects[1].vertices, expected.objects[1].vertices)
    actual_layout, expected_layout = actual.objects[0].metadata["layout"], expected.objects[0].metadata["layout"]
    assert actual_layout.keys() == expected_layout.keys()
    for key, value in expected_layout.items():
        if isinstance(value, (str, bool)):
            assert actual_layout[key] == value, key
        else:  # float poses differ in the last bits across platforms
            torch.testing.assert_close(torch.tensor(actual_layout[key]), torch.tensor(value), msg=key)


def test_unsupported_release_configuration_is_rejected(tmp_path, tiny_production):
    source = tmp_path / "checkpoints"
    source.mkdir()
    _write_release(source, tiny_sam3d_pipeline())
    config = yaml.safe_load((source / "pipeline.yaml").read_text())
    (source / "pipeline.yaml").write_text(yaml.safe_dump({**config, "ss_cfg_strength": 3}))
    with pytest.raises(ValueError, match="sparse-structure sampler"):
        convert_sam3d_checkpoint(source, tmp_path / "converted")


def test_differing_dinov2_copies_are_rejected(tmp_path, tiny_production):
    source = tmp_path / "checkpoints"
    source.mkdir()
    _write_release(source, tiny_sam3d_pipeline())
    release = torch.load(source / "slat_generator.ckpt", weights_only=True)
    release["state_dict"]["_base_models.condition_embedder.module_list.1.backbone.norm.weight"] += 1.0
    torch.save(release, source / "slat_generator.ckpt")
    with pytest.raises(ValueError, match="DINOv2 image and mask backbones"):
        convert_sam3d_checkpoint(source, tmp_path / "converted")
