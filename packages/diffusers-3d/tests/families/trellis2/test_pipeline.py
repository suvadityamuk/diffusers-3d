from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    CoordinateSystem,
    ImageCondition,
    MeshAsset,
    Object3DPipelineOutput,
    OVoxelAsset,
    SparseVoxelAsset,
    Trellis2ImageTo3DPipeline,
    preprocess_image_condition,
)
from diffusers_3d.families.trellis.decoders import TrellisSparseStructureDecoderOutput
from diffusers_3d.families.trellis.sparse import trellis_grid_transform

pytestmark = pytest.mark.integration


def test_batch_conditioning_uses_pinned_trellis2_preprocessing(tiny_trellis2_pipeline):
    rgba = torch.zeros(4, 10, 12)
    rgba[:3] = 1
    rgba[3, 2:8, :5] = 1
    conditions = (
        ImageCondition(rgba),
        ImageCondition(rgba[:3], mask=rgba[3:4]),
    )

    images = tiny_trellis2_pipeline.preprocess(conditions)
    expected = torch.stack(
        [
            preprocess_image_condition(
                condition,
                image_size=8,
                foreground_scale=1.0,
                premultiply_before_resize=True,
            ).image
            for condition in conditions
        ]
    )
    conditional, negative = tiny_trellis2_pipeline.encode_conditioning(images)

    torch.testing.assert_close(images, expected, atol=0.0, rtol=0.0)
    assert conditional.shape[0] == negative.shape[0] == 2


def test_reviewed_sparse_structure_pipeline_is_deterministic_and_uses_released_defaults(
    tiny_trellis2_pipeline,
    monkeypatch,
):
    pipeline = tiny_trellis2_pipeline
    defaults = pipeline.config.sparse_structure_sampler_defaults
    assert defaults == {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "guidance_interval": (0.6, 1.0),
        "rescale_t": 5.0,
    }
    image = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(3, 8, 8)
    latents = torch.linspace(-1.0, 1.0, 2 * 2 * 2 * 2).reshape(1, 2, 2, 2, 2)
    first = pipeline(
        image,
        sparse_structure_latents=latents.clone(),
        sparse_structure_sampler_params={"steps": 2},
    )
    second = pipeline(
        image,
        sparse_structure_latents=latents.clone(),
        sparse_structure_sampler_params={"steps": 2},
    )
    assert type(first) is Object3DPipelineOutput
    assert len(first.objects) == 1
    assert type(first.objects[0]) is SparseVoxelAsset
    assert first.objects[0].metadata["representation"] == "sparse_structure"
    assert first.objects[0].metadata["decoder_checkpoint_semantics"] == "trellis-image-large-exact-reuse"
    # The tiny decoder emits a 4^3 grid; the 512 presets max-pool it by two before the SLAT stage.
    assert first.objects[0].metadata["resolution"] == 2
    torch.testing.assert_close(first.objects[0].features, second.objects[0].features)
    torch.testing.assert_close(first.latents.latents, second.latents.latents)

    logits = torch.full((1, 1, 64, 64, 64), -1.0)
    logits[0, 0, 2, 4, 6] = 1.0
    logits[0, 0, 63, 63, 63] = 1.0
    monkeypatch.setattr(
        pipeline.sparse_structure_decoder,
        "forward",
        lambda hidden_states: TrellisSparseStructureDecoderOutput(sample=logits),
    )
    pooled = pipeline(
        image,
        sparse_structure_latents=latents,
        sparse_structure_sampler_params={"steps": 1},
        pipeline_type="1024_cascade",
    ).objects[0]
    expected = F.max_pool3d((logits > 0).float(), 2, 2, 0) > 0.5
    assert torch.equal(pooled.coordinates, torch.argwhere(expected[0, 0]))
    assert pooled.metadata["resolution"] == 32
    torch.testing.assert_close(pooled.grid_transform, trellis_grid_transform(32))
    full = pipeline(
        image,
        sparse_structure_latents=latents,
        sparse_structure_sampler_params={"steps": 1},
        pipeline_type="1024",
    ).objects[0]
    assert torch.equal(full.coordinates, torch.argwhere(logits[0, 0] > 0))
    assert full.metadata["resolution"] == 64


def test_pipeline_save_load_auto_and_serialized_capability_limitations(
    tmp_path, tiny_trellis2_pipeline, tiny_trellis2_full_pipeline
):
    tiny_trellis2_pipeline.save_pretrained(tmp_path / "sparse")
    loaded = Trellis2ImageTo3DPipeline.from_pretrained(tmp_path / "sparse", local_files_only=True)
    automatic = AutoPipelineForImageTo3D.from_pretrained(tmp_path / "sparse", local_files_only=True)
    assert type(loaded) is Trellis2ImageTo3DPipeline
    assert type(automatic) is Trellis2ImageTo3DPipeline
    assert loaded.shape_slat_flow_model is None
    assert loaded.texture_slat_flow_model is None
    assert loaded.pbr_decoder is None
    assert automatic.config.capability_limitations == {
        "official_full_checkpoint_parity": False,
        "production_gpu_quality_verified": False,
    }

    tiny_trellis2_full_pipeline.save_pretrained(tmp_path / "full")
    full = AutoPipelineForImageTo3D.from_pretrained(tmp_path / "full", local_files_only=True)
    assert type(full) is Trellis2ImageTo3DPipeline
    assert full.shape_slat_flow_model_1024 is not None
    assert full.texture_slat_flow_model_1024 is not None
    assert full.pbr_decoder is not None
    assert full.config.shape_slat_mean == tiny_trellis2_full_pipeline.config.shape_slat_mean


def test_shape_texture_and_ovoxel_stages_return_native_assets(tiny_trellis2_full_pipeline):
    pipeline = tiny_trellis2_full_pipeline
    image = torch.zeros(3, 8, 8)
    output = pipeline(
        image,
        formats=("sparse_structure", "shape_slat", "texture_slat", "o_voxel"),
        sparse_structure_sampler_params={"steps": 1, "guidance_rescale": 0.0},
        shape_slat_sampler_params={"steps": 1},
        texture_slat_sampler_params={"steps": 1},
        sparse_structure_latents=torch.zeros(1, 2, 2, 2, 2),
        generator=torch.Generator().manual_seed(5),
    )
    assert [type(value) for value in output.objects] == [
        SparseVoxelAsset,
        SparseVoxelAsset,
        SparseVoxelAsset,
        OVoxelAsset,
    ]
    structure, shape_slat, texture_slat, ovoxel = output.objects
    assert torch.equal(shape_slat.coordinates, texture_slat.coordinates)
    assert torch.equal(shape_slat.coordinates, structure.coordinates)
    assert shape_slat.metadata["resolution"] == structure.metadata["resolution"]
    # The tiny shape decoder upsamples once, so O-Voxels sit on a grid twice the SLAT grid.
    upsamples = pipeline.shape_slat_decoder.num_upsamples
    assert ovoxel.metadata["resolution"] == [structure.metadata["resolution"] * 2**upsamples] * 3
    assert bool((ovoxel.active_coordinates // 2**upsamples).unique(dim=0).shape[0] <= structure.coordinates.shape[0])
    assert ovoxel.metadata["stage"] == "pbr_decoder"
    assert ovoxel.metadata["pbr_channel_layout"] == ["base_color", "metallic", "roughness", "alpha"]
    assert ovoxel.base_color.shape[1] == 3
    assert ovoxel.metallic.shape[1] == ovoxel.roughness.shape[1] == ovoxel.opacity.shape[1] == 1
    assert ovoxel.normals.shape[1] == ovoxel.emissive.shape[1] == 3
    assert ovoxel.split_weights.shape[1] == 1
    assert ovoxel.dual_grid_vertex_offsets.shape == (ovoxel.active_coordinates.shape[0], 3)
    assert ovoxel.intersection_data.dtype is torch.bool

    default = pipeline(
        image,
        sparse_structure_sampler_params={"steps": 1, "guidance_rescale": 0.0},
        shape_slat_sampler_params={"steps": 1},
        texture_slat_sampler_params={"steps": 1},
        sparse_structure_latents=torch.zeros(1, 2, 2, 2, 2),
    )
    assert [type(value) for value in default.objects] == [OVoxelAsset]


@pytest.mark.parametrize("pipeline_type", ("1024_cascade", "1536_cascade"))
def test_cascade_presets_grow_the_slat_grid_with_the_shape_decoder(tiny_trellis2_full_pipeline, pipeline_type):
    pipeline = tiny_trellis2_full_pipeline
    scale = {"1024_cascade": 2, "1536_cascade": 3}[pipeline_type]
    calls = []
    original = pipeline.upsample_structures

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    pipeline.upsample_structures = recording
    output = pipeline(
        torch.zeros(3, 8, 8),
        formats=("sparse_structure", "shape_slat", "o_voxel"),
        pipeline_type=pipeline_type,
        sparse_structure_sampler_params={"steps": 1, "guidance_rescale": 0.0},
        shape_slat_sampler_params={"steps": 1},
        texture_slat_sampler_params={"steps": 1},
        sparse_structure_latents=torch.zeros(1, 2, 2, 2, 2),
        generator=torch.Generator().manual_seed(5),
    )
    structure, shape_slat, ovoxel = output.objects
    assert len(calls) == 1
    grown, resolution = calls[0]
    assert resolution == structure.metadata["resolution"] * scale
    assert shape_slat.metadata["resolution"] == resolution
    assert torch.equal(shape_slat.coordinates, grown[0].coordinates)
    assert bool((shape_slat.coordinates < resolution).all())
    assert ovoxel.metadata["resolution"] == [resolution * 2**pipeline.shape_slat_decoder.num_upsamples] * 3

    # Cascades need both stages; a pipeline with only the 512 models says so.
    pipeline.shape_slat_flow_model_1024 = None
    with pytest.raises(RuntimeError, match="1024 shape SLAT flow model"):
        pipeline(torch.zeros(3, 8, 8), formats=("shape_slat",), pipeline_type=pipeline_type)


def test_cascade_token_budget_shrinks_the_target_grid(tiny_trellis2_full_pipeline):
    pipeline = tiny_trellis2_full_pipeline
    structure = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64),
        features=torch.ones(2, 1),
        grid_transform=trellis_grid_transform(2),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        metadata={"resolution": 2},
    )
    latents = pipeline.prepare_slat_latents(
        (structure,),
        pipeline.shape_slat_flow_model,
        channels=pipeline.shape_slat_decoder.config.latent_channels,
        generator=torch.Generator().manual_seed(0),
    )
    with torch.no_grad():
        pipeline.shape_slat_decoder.blocks[0][-1].to_subdiv.bias.fill_(10.0)
    grown, resolution = pipeline.upsample_structures((latents), (structure,), cascade_scale=3, max_num_tokens=10**6)
    assert resolution == 6
    grown, resolution = pipeline.upsample_structures((latents), (structure,), cascade_scale=3, max_num_tokens=1)
    assert resolution == 4
    assert grown[0].metadata["stage"] == "cascade_upsample"


def test_slat_and_ovoxel_assets_preserve_centered_grid_and_world_transform(tiny_trellis2_full_pipeline):
    pipeline = tiny_trellis2_full_pipeline
    world_transform = torch.eye(4)
    world_transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    structure = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64),
        features=torch.ones(2, 1),
        grid_transform=trellis_grid_transform(4),
        transform=world_transform,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        metadata={"resolution": 4},
    )
    latents = pipeline.prepare_slat_latents(
        (structure,),
        pipeline.shape_slat_flow_model,
        channels=pipeline.shape_slat_flow_model.config.in_channels,
        generator=torch.Generator().manual_seed(0),
    )
    slat = pipeline._slat_assets(latents, resolution=8, stage="shape")[0]

    torch.testing.assert_close(slat.grid_transform, trellis_grid_transform(8))
    torch.testing.assert_close(slat.transform, world_transform)
    decoder_input = latents.replace(
        torch.randn(
            latents.features.shape[0],
            pipeline.shape_slat_decoder.config.latent_channels,
            generator=torch.Generator().manual_seed(1),
        )
    )
    ovoxel = pipeline.shape_slat_decoder(decoder_input, resolution=16).assets[0]
    torch.testing.assert_close(ovoxel.transform, world_transform)
    assert ovoxel.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP


def test_missing_slat_components_fail_explicitly(tiny_trellis2_pipeline):
    with pytest.raises(RuntimeError, match="shape SLAT"):
        tiny_trellis2_pipeline(
            torch.zeros(3, 8, 8),
            formats=("shape_slat",),
            pipeline_type="512",
        )
    with pytest.raises(ValueError, match="max_num_tokens"):
        tiny_trellis2_pipeline(torch.zeros(3, 8, 8), max_num_tokens=0)


@pytest.mark.parametrize("pipeline_type", ("unknown", "", 1024))
def test_pipeline_type_is_validated_for_sparse_only_calls(tiny_trellis2_pipeline, pipeline_type):
    with pytest.raises(ValueError, match="pipeline_type must be one of"):
        tiny_trellis2_pipeline(
            torch.zeros(3, 8, 8),
            formats=("sparse_structure",),
            pipeline_type=pipeline_type,
        )


def test_glb_is_explicit_postprocess_and_return_dict_false_preserves_mesh(
    tiny_trellis2_full_pipeline,
):
    pipeline = tiny_trellis2_full_pipeline
    with pytest.raises(ValueError, match="formats must contain unique values"):
        pipeline(torch.zeros(3, 8, 8), formats=("glb",))

    class MeshBackend:
        asset = None

        def to_mesh(self, asset, **kwargs):
            self.asset = asset
            return MeshAsset(
                vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            )

    mesh_backend = MeshBackend()
    objects, latent_output = pipeline(
        torch.zeros(3, 8, 8),
        formats=("mesh",),
        sparse_structure_sampler_params={"steps": 1, "guidance_rescale": 0.0},
        shape_slat_sampler_params={"steps": 1},
        texture_slat_sampler_params={"steps": 1},
        sparse_structure_latents=torch.zeros(1, 2, 2, 2, 2),
        ovoxel_backend=mesh_backend,
        return_dict=False,
    )
    assert len(objects) == 1
    assert type(objects[0]) is MeshAsset
    assert latent_output is not None

    sentinel = object()

    class PBRPostprocess:
        def to_glb(self, asset, **kwargs):
            assert asset is mesh_backend.asset
            return sentinel

    assert (
        pipeline.postprocess_ovoxel(
            mesh_backend.asset,
            output_format="glb",
            pbr_postprocess=PBRPostprocess(),
        )
        is sentinel
    )
