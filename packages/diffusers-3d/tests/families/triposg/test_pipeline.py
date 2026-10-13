from __future__ import annotations

import numpy as np
import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    CoordinateSystem,
    ImageCondition,
    MeshAsset,
    TripoSGImageTo3DPipeline,
    prepare_triposg_image,
)

_SMALL = {"num_inference_steps": 2, "num_tokens": 6, "dense_octree_depth": 3, "hierarchical_octree_depth": 4}


def test_prepare_image_crops_composites_on_white_and_pads_like_upstream(rgba_condition):
    image = np.asarray(prepare_triposg_image(rgba_condition))
    # Foreground is 20 rows x 15 columns: pad rows by int(0.1 * 20) = 2 and columns by int(2 + 5 / 2) = 4.
    assert image.shape == (24, 23, 3)
    assert (image[:2] == 255).all() and (image[:, :4] == 255).all()
    expected = rgba_condition.image[:3, 10:30, 5:20].mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(image[2:22, 4:19], expected)


def test_prepare_image_uses_the_mask_and_rejects_inputs_that_need_background_removal(rgba_condition):
    rgb = rgba_condition.image[:3]
    mask = rgba_condition.image[3:4]
    np.testing.assert_array_equal(
        np.asarray(prepare_triposg_image(ImageCondition(image=rgb, mask=mask))),
        np.asarray(prepare_triposg_image(rgba_condition)),
    )
    with pytest.raises(ValueError, match="RMBG"):
        prepare_triposg_image(ImageCondition(image=rgb))
    with pytest.raises(ValueError, match="clear foreground and background"):
        prepare_triposg_image(ImageCondition(image=rgb, mask=torch.full_like(mask, 0.5)))


def test_scheduler_timesteps_follow_the_rectified_flow_schedule(tiny_triposg_pipeline):
    scheduler = tiny_triposg_pipeline.scheduler
    scheduler.set_timesteps(sigmas=[1.0 - index / 4 for index in range(4)])
    assert scheduler.timesteps.tolist() == [1000.0, 750.0, 500.0, 250.0]
    assert scheduler.sigmas[-1] == 0


def test_extract_mesh_rejects_batches_and_inverted_depths(tiny_triposg_pipeline):
    with pytest.raises(ValueError, match="one shape at a time"):
        tiny_triposg_pipeline.extract_mesh(torch.zeros(2, 6, 4))
    with pytest.raises(ValueError, match="octree depths"):
        tiny_triposg_pipeline.extract_mesh(torch.zeros(1, 6, 4), dense_octree_depth=5, hierarchical_octree_depth=4)


@pytest.mark.portable
def test_pipeline_generates_cpu_y_up_meshes_and_round_trips(tiny_triposg_pipeline, rgba_condition, tmp_path):
    pipeline = tiny_triposg_pipeline
    pipeline.set_progress_bar_config(disable=True)
    generators = [torch.Generator().manual_seed(seed) for seed in (0, 1)]
    output = pipeline([rgba_condition, rgba_condition], generator=generators, **_SMALL)
    assert len(output.objects) == 2
    assert output.latents.latents.shape == (2, 6, 4)
    for mesh in output.objects:
        assert type(mesh) is MeshAsset
        assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Y_UP
        assert mesh.metadata["resolution"] == 16
        assert mesh.vertices.abs().max() <= 1.005
    assert not torch.equal(output.objects[0].vertices, output.objects[1].vertices)

    unguided = pipeline(rgba_condition, guidance_scale=1.0, generator=torch.Generator().manual_seed(0), **_SMALL)
    assert unguided.latents.latents.shape == (1, 6, 4)

    single = pipeline(rgba_condition, generator=torch.Generator().manual_seed(0), **_SMALL).objects[0]
    pipeline.save_pretrained(tmp_path)
    loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is TripoSGImageTo3DPipeline
    loaded.set_progress_bar_config(disable=True)
    reloaded = loaded(rgba_condition.image, generator=torch.Generator().manual_seed(0), **_SMALL).objects[0]
    assert torch.equal(reloaded.vertices, single.vertices)
    assert torch.equal(reloaded.faces, single.faces)


def test_construction_warns_about_the_hunyuan_derived_transformer(tiny_triposg_components):
    with pytest.warns(UserWarning, match="Tencent Hunyuan Community License"):
        TripoSGImageTo3DPipeline(**tiny_triposg_components)
