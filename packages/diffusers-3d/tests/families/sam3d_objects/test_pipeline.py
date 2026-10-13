from __future__ import annotations

import warnings

import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    CoordinateSystem,
    GaussianSplatAsset,
    ImageCondition,
    MeshAsset,
    Sam3DObjectsPipeline,
    SparseVoxelAsset,
    preprocess_sam3d_condition,
)

from .conftest import tiny_condition, tiny_sam3d_components, tiny_sam3d_pipeline

pytestmark = pytest.mark.integration


def _run(pipeline, condition, pointmap, **kwargs):
    return pipeline(
        condition,
        pointmap=pointmap,
        sparse_structure_num_inference_steps=2,
        slat_num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
        **kwargs,
    )


def test_construction_warns_about_the_sam_license():
    with pytest.warns(UserWarning, match="SAM License.*complies with that license"):
        Sam3DObjectsPipeline(**tiny_sam3d_components())


def test_all_formats_share_one_camera_layout(tiny_pipeline):
    condition, pointmap = tiny_condition()
    output = _run(tiny_pipeline, condition, pointmap, formats=("sparse_structure", "gaussian", "mesh"))
    assert [type(asset) for asset in output.objects] == [SparseVoxelAsset, GaussianSplatAsset, MeshAsset]
    assert output.latents.latents.shape == (1, 8, 2)
    transforms = [asset.transform for asset in output.objects]
    for transform in transforms[1:]:
        torch.testing.assert_close(transform, transforms[0])

    mesh = output.objects[2]
    layout = mesh.metadata["layout"]
    assert layout["frame"] == "pytorch3d_camera" and layout["metric"] is True
    assert mesh.metadata["family"] == "sam3d_objects"
    rotation = torch.tensor(layout["rotation"])
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(3), atol=1e-5, rtol=0)
    # In the Y-up convention the placed mesh lives in PyTorch3D camera coordinates (upstream's row convention).
    y_up = mesh.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP)
    world = y_up.vertices @ y_up.transform[:3, :3].T + y_up.transform[:3, 3]
    expected = layout["scale"] * mesh.vertices @ rotation + torch.tensor(layout["translation"])
    torch.testing.assert_close(world, expected, atol=1e-5, rtol=1e-5)


def test_layout_follows_the_pointmap_units_and_dropping_it_is_scale_free(tiny_pipeline):
    condition, pointmap = tiny_condition()
    metric = _run(tiny_pipeline, condition, pointmap, formats=("sparse_structure",)).objects[0]
    doubled = _run(tiny_pipeline, condition, 2 * pointmap, formats=("sparse_structure",)).objects[0]
    # Pointmaps are scale-shift normalized, so doubling them doubles scale and translation only.
    assert doubled.metadata["layout"]["scale"] == pytest.approx(2 * metric.metadata["layout"]["scale"], rel=1e-5)
    torch.testing.assert_close(
        torch.tensor(doubled.metadata["layout"]["translation"]),
        2 * torch.tensor(metric.metadata["layout"]["translation"]),
    )
    dropped = _run(tiny_pipeline, condition, None, formats=("sparse_structure",)).objects[0]
    assert dropped.metadata["layout"]["metric"] is False


def test_pointmaps_are_opencv_points_resized_to_the_image(tiny_pipeline):
    condition, pointmap = tiny_condition()
    low_resolution = pointmap[:, ::2, ::2]
    views = tiny_pipeline.preprocess(condition, low_resolution)
    # Nearest upsampling by two, then OpenCV (x right, y down) to PyTorch3D (x left, y up) camera axes.
    upsampled = low_resolution.repeat_interleave(2, 1).repeat_interleave(2, 2)
    expected = preprocess_sam3d_condition(
        condition.image, condition.mask, upsampled * torch.tensor([-1.0, -1.0, 1.0])[:, None, None], image_size=8
    )
    for name, value in expected.items():
        torch.testing.assert_close(views[name][0], value, equal_nan=True)
    assert views["images"].shape == (1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="one pointmap per image"):
        tiny_pipeline.preprocess([condition, condition], [pointmap, None])
    with pytest.raises(ValueError, match="object mask"):
        tiny_pipeline.preprocess(ImageCondition(image=condition.image))


def test_generator_determinism_batching_and_tuple_output(tiny_pipeline):
    condition, pointmap = tiny_condition()
    other, other_pointmap = tiny_condition(1)
    single = _run(tiny_pipeline, condition, pointmap, formats=("gaussian",))
    repeated, latents = _run(tiny_pipeline, condition, pointmap, formats=("gaussian",), return_dict=False)
    torch.testing.assert_close(repeated[0].means, single.objects[0].means)
    torch.testing.assert_close(latents.latents, single.latents.latents)
    batched = tiny_pipeline(
        [condition, other],
        pointmap=[pointmap, other_pointmap],
        formats=("gaussian",),
        sparse_structure_num_inference_steps=2,
        slat_num_inference_steps=2,
        generator=[torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)],
    )
    assert len(batched.objects) == 2
    torch.testing.assert_close(batched.objects[0].transform, single.objects[0].transform, atol=1e-5, rtol=1e-5)


def test_save_load_round_trip_and_auto_pipeline(tmp_path, tiny_pipeline):
    condition, pointmap = tiny_condition()
    expected = _run(tiny_pipeline, condition, pointmap, formats=("gaussian",)).objects[0]
    tiny_pipeline.save_pretrained(tmp_path / "full")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path / "full", local_files_only=True)
        assert type(loaded) is Sam3DObjectsPipeline
        torch.testing.assert_close(
            _run(loaded, condition, pointmap, formats=("gaussian",)).objects[0].means, expected.means
        )

        tiny_sam3d_pipeline(include_slat=False).save_pretrained(tmp_path / "structure")
        structure_only = Sam3DObjectsPipeline.from_pretrained(tmp_path / "structure", local_files_only=True)
    assert structure_only.slat_flow_model is None
    assert [type(asset) for asset in _run(structure_only, condition, pointmap).objects] == [SparseVoxelAsset]
    with pytest.raises(RuntimeError, match="structured-latent stage"):
        _run(structure_only, condition, pointmap, formats=("mesh",))
