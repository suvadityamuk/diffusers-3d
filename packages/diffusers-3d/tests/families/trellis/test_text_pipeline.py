from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    AutoPipelineForTextTo3D,
    GaussianSplatAsset,
    Object3DPipelineOutput,
    SparseVoxelAsset,
    TextCondition,
    TrellisClipTextConditioner,
    TrellisImageTo3DPipeline,
    TrellisTextTo3DPipeline,
)
from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint

pytestmark = pytest.mark.integration


def test_text_pipeline_is_deterministic_and_conditions_both_stages(tiny_trellis_text_pipeline):
    pipeline = tiny_trellis_text_pipeline
    latents = torch.linspace(-1.0, 1.0, 2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4)
    # Upstream initialization zeroes adaLN gates and the output layer, so a fresh flow predicts zero velocity
    # whatever the prompt; perturb the SLAT flow so conditioning can reach its output.
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in pipeline.slat_flow_model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)

    def run(prompt):
        return pipeline(
            prompt,
            formats=("sparse_structure", "slat", "gaussian"),
            sparse_structure_latents=latents.clone(),
            sparse_structure_num_inference_steps=2,
            slat_num_inference_steps=2,
            generator=torch.Generator().manual_seed(0),
        )

    first = run("a red chair")
    second = run(TextCondition(text="a red chair"))
    other = run("a chair")
    assert type(first) is Object3DPipelineOutput
    assert [type(asset) for asset in first.objects] == [SparseVoxelAsset, SparseVoxelAsset, GaussianSplatAsset]
    assert first.objects[1].features.shape[1] == pipeline.slat_flow_model.config.out_channels
    torch.testing.assert_close(first.objects[1].features, second.objects[1].features)
    torch.testing.assert_close(first.objects[2].means, second.objects[2].means)
    # A different prompt changes the structured latent: text tokens reach the flows' cross-attention.
    assert not torch.allclose(first.objects[1].features, other.objects[1].features)


def test_text_pipeline_negative_prompt_replaces_the_released_empty_unconditional(tiny_trellis_text_pipeline):
    pipeline = tiny_trellis_text_pipeline
    prompts, negatives = pipeline.preprocess(["a red chair", TextCondition(text="a chair", negative_text="red")])
    assert prompts == ["a red chair", "a chair"]
    assert negatives == ["", "red"]
    conditional, unconditional = pipeline.encode_conditioning(prompts, negatives)
    assert conditional.shape == unconditional.shape == (2, pipeline.conditioner.max_length, 12)
    torch.testing.assert_close(unconditional[1], pipeline.conditioner(["red"]).embeddings[0])

    _, released = pipeline.encode_conditioning(["a red chair"], [""])
    torch.testing.assert_close(released, pipeline.conditioner.unconditional_embedding(1))
    with pytest.raises(TypeError, match="strings or exact TextCondition"):
        pipeline.preprocess([1])
    with pytest.raises(ValueError, match="must not be empty"):
        pipeline.preprocess([])


def test_text_pipeline_save_load_auto_loader_and_task_gating(tmp_path, tiny_trellis_text_pipeline):
    pipeline = tiny_trellis_text_pipeline
    pipeline.save_pretrained(tmp_path)
    model_index = json.loads((tmp_path / "model_index.json").read_text())
    assert model_index["_class_name"] == "TrellisTextTo3DPipeline"
    assert model_index["conditioner"][1] == "TrellisClipTextConditioner"

    loaded = TrellisTextTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    automatic = AutoPipelineForTextTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is type(automatic) is TrellisTextTo3DPipeline
    assert type(automatic.conditioner) is TrellisClipTextConditioner
    assert automatic.conditioner.tokenizer is not None
    with torch.no_grad():
        torch.testing.assert_close(
            automatic.conditioner(["a red chair"]).embeddings, pipeline.conditioner(["a red chair"]).embeddings
        )
    with pytest.raises(Exception, match="text-to-3d|image-to-3d"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


def test_text_pipeline_requires_the_clip_conditioner(tiny_trellis_components, tiny_text_conditioner):
    conditioner, flow, decoder, scheduler = tiny_trellis_components()
    with pytest.raises(TypeError, match="TrellisClipTextConditioner"):
        TrellisTextTo3DPipeline(conditioner, flow, decoder, scheduler)
    with pytest.raises(TypeError, match="TrellisDinov2Conditioner"):
        TrellisImageTo3DPipeline(tiny_text_conditioner, flow, decoder, scheduler)


def _write_component(root, stem, upstream_name, model):
    (root / f"{stem}.json").write_text(json.dumps({"name": upstream_name, "args": dict(model.config)}))
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()}, root / f"{stem}.safetensors"
    )


def test_synthetic_text_conversion_from_a_clip_folder_auto_loads(
    tmp_path, tiny_trellis_components, tiny_text_conditioner
):
    source = tmp_path / "source"
    (source / "ckpts").mkdir(parents=True)
    (source / "shared" / "ckpts").mkdir(parents=True)
    _, flow, decoder, _, slat_flow, _, gaussian_decoder, mesh_decoder = tiny_trellis_components(include_slat=True)
    # Text releases reference the shared decoders from another repository, as ``<repo>/ckpts/<name>``.
    _write_component(source / "ckpts", "ss_flow_txt", "SparseStructureFlowModel", flow)
    _write_component(source / "ckpts", "slat_flow_txt", "SLatFlowModel", slat_flow)
    _write_component(source / "shared" / "ckpts", "ss_dec", "SparseStructureDecoder", decoder)
    _write_component(source / "shared" / "ckpts", "slat_gs", "SLatGaussianDecoder", gaussian_decoder)
    _write_component(source / "shared" / "ckpts", "slat_mesh", "SLatMeshDecoder", mesh_decoder)
    sampler = {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {"steps": 25, "cfg_strength": 7.5, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
    }
    (source / "pipeline.json").write_text(
        json.dumps(
            {
                "name": "TrellisTextTo3DPipeline",
                "args": {
                    "text_cond_model": "openai/clip-vit-large-patch14",
                    "models": {
                        "sparse_structure_flow_model": "ckpts/ss_flow_txt",
                        "slat_flow_model": "ckpts/slat_flow_txt",
                        "sparse_structure_decoder": "shared/ckpts/ss_dec",
                        "slat_decoder_gs": "shared/ckpts/slat_gs",
                        "slat_decoder_mesh": "shared/ckpts/slat_mesh",
                        "slat_decoder_rf": "shared/ckpts/slat_rf",
                    },
                    "slat_normalization": {"mean": [0.0] * 4, "std": [1.0] * 4},
                    "slat_sampler": sampler,
                    "sparse_structure_sampler": sampler,
                },
            }
        )
    )
    # The released conditioner is the Hub CLIP checkpoint: weights plus tokenizer files in one folder.
    tiny_text_conditioner.model.save_pretrained(tmp_path / "clip")
    tiny_text_conditioner.tokenizer.save_pretrained(tmp_path / "clip")

    output = convert_trellis_checkpoint(source, tmp_path / "converted", conditioner_path=tmp_path / "clip")
    model_index = json.loads((output / "model_index.json").read_text())
    report = json.loads((output / "trellis_conversion.json").read_text())
    # The released text pipeline.json omits text_cond_model entirely (upstream hard-codes CLIP): still accepted.
    released = json.loads((source / "pipeline.json").read_text())
    del released["args"]["text_cond_model"]
    (source / "pipeline.json").write_text(json.dumps(released))
    convert_trellis_checkpoint(source, tmp_path / "converted_released", conditioner_path=tmp_path / "clip")
    assert model_index["_class_name"] == "TrellisTextTo3DPipeline"
    assert model_index["conditioner"][1] == "TrellisClipTextConditioner"
    assert set(report["skipped_components"]) == {"slat_decoder_rf"}
    assert report["samplers"]["sparse_structure"]["cfg_strength"] == 7.5

    loaded = AutoPipelineForTextTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is TrellisTextTo3DPipeline
    # max_length follows the CLIP checkpoint's max_position_embeddings; the tiny one is 8 rather than 77.
    assert loaded.conditioner.max_length == 8
    output_objects = loaded(
        "a red chair",
        formats=("gaussian",),
        sparse_structure_num_inference_steps=1,
        slat_num_inference_steps=1,
        generator=torch.Generator().manual_seed(0),
    ).objects
    assert type(output_objects[0]) is GaussianSplatAsset

    with pytest.raises(ValueError, match="only the released 'openai/clip-vit-large-patch14'"):
        bad = json.loads((source / "pipeline.json").read_text())
        bad["args"]["text_cond_model"] = "other"
        (source / "pipeline.json").write_text(json.dumps(bad))
        convert_trellis_checkpoint(source, tmp_path / "bad", conditioner_path=tmp_path / "clip")
