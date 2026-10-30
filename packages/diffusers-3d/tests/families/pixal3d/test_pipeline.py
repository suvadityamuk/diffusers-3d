from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
from _configs import tiny_config
from moge.configs import tiny_moge2_config

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    CameraRig,
    CoordinateSystem,
    ImageCondition,
    MeshAsset,
    MultiViewCondition,
    OVoxelAsset,
    Pixal3DImageTo3DPipeline,
    Pixal3DSparseStructureFlowModel,
    SparseVoxelAsset,
)
from diffusers_3d.data import preprocess_image_condition
from diffusers_3d.families.moge import MoGe2Model
from diffusers_3d.families.pixal3d.conditioner import (
    camera_distance_from_fov,
    front_view_camera,
    project_grid_features,
)

from .conftest import TINY_IMAGE_SIZE, tiny_pipeline

pytestmark = pytest.mark.integration

STEPS = {
    "sparse_structure_sampler_params": {"steps": 1},
    "shape_slat_sampler_params": {"steps": 1},
    "texture_slat_sampler_params": {"steps": 1},
}


def _rgba(height: int = 20, width: int = 24) -> torch.Tensor:
    image = torch.rand(4, height, width, generator=torch.Generator().manual_seed(0))
    image[3] = 0
    image[3, 4:16, 5:19] = 1
    return image


def _square_rig(views: int, fov: float) -> CameraRig:
    focal = 12 / math.tan(fov / 2)
    world_to_camera = torch.eye(4).repeat(views, 1, 1)
    world_to_camera[:, 2, 3] = 2.0
    return CameraRig(
        world_to_camera=world_to_camera,
        intrinsics=torch.tensor([[focal, 0.0, 12.0], [0.0, focal, 12.0], [0.0, 0.0, 1.0]]).repeat(views, 1, 1),
        image_sizes=torch.full((views, 2), 24),
    )


@pytest.mark.parametrize("pipeline_type", ("1024_cascade", "1536_cascade"))
def test_cascades_return_view_aligned_native_assets(tiny_pixal3d_pipeline, pipeline_type):
    output = tiny_pixal3d_pipeline(
        ImageCondition(_rgba()),
        camera_angle_x=0.4,
        formats=("sparse_structure", "shape_slat", "texture_slat", "o_voxel"),
        pipeline_type=pipeline_type,
        generator=torch.Generator().manual_seed(1),
        **STEPS,
    )
    structure, shape_slat, texture_slat, ovoxel = output.objects
    assert [type(value) for value in output.objects] == [SparseVoxelAsset] * 3 + [OVoxelAsset]
    resolution = structure.metadata["resolution"] * {"1024_cascade": 2, "1536_cascade": 3}[pipeline_type]
    assert shape_slat.metadata["resolution"] == texture_slat.metadata["resolution"] == resolution
    assert torch.equal(shape_slat.coordinates, texture_slat.coordinates)
    assert ovoxel.metadata["resolution"] == [resolution * 2] * 3
    assert {value.coordinate_system for value in output.objects} == {CoordinateSystem.RIGHT_HANDED_Y_UP}
    assert structure.metadata["family"] == "pixal3d"


def test_cascade_rounds_onto_the_linspace_grid():
    # Upstream ``run``: ((fine + 0.5) / 512 * (grid - 1)).round(), unlike TRELLIS.2's truncation onto ``grid``.
    fine = torch.tensor([[0, 0, 0], [255, 256, 511], [100, 31, 480]])
    expected = ((fine + 0.5) / 512 * 95).round().to(fine.dtype)
    assert torch.equal(Pixal3DImageTo3DPipeline._quantize_cascade_coordinates(fine, 512, 96), expected)
    assert int(Pixal3DImageTo3DPipeline._quantize_cascade_coordinates(fine, 512, 96).max()) == 95


def test_single_view_preprocessing_and_camera(tiny_pixal3d_pipeline):
    pipeline = tiny_pixal3d_pipeline
    condition = ImageCondition(_rgba())
    views = pipeline.prepare_views(condition, image_sizes=(TINY_IMAGE_SIZE,), camera_angle_x=0.5, mesh_scale=0.8)
    expected = preprocess_image_condition(
        condition, image_size=TINY_IMAGE_SIZE, foreground_scale=1.1, premultiply_before_resize=True
    ).image
    torch.testing.assert_close(views.images[TINY_IMAGE_SIZE][0, 0], expected, atol=0.0, rtol=0.0)
    assert float(views.camera_to_world[0, 0, 1, 3]) == pytest.approx(-0.5 / (0.8 * math.tan(0.25)))
    assert camera_distance_from_fov(0.5, 0.8) == pytest.approx(0.5 / (0.8 * math.tan(0.25)))

    # The field of view can come from the condition camera instead (after the crop updates its intrinsics).
    focal = 10 / math.tan(0.25)
    camera = CameraRig(
        world_to_camera=torch.eye(4)[None],
        intrinsics=torch.tensor([[[focal, 0.0, 12.0], [0.0, focal, 10.0], [0.0, 0.0, 1.0]]]),
        image_sizes=torch.tensor([[20, 24]]),
    )
    with_camera = pipeline.prepare_views(
        ImageCondition(_rgba(), camera=camera), image_sizes=(TINY_IMAGE_SIZE,)
    ).camera_angle_x
    cropped = preprocess_image_condition(
        ImageCondition(_rgba(), camera=camera), image_size=TINY_IMAGE_SIZE, foreground_scale=1.1
    ).camera
    expected_fov = 2 * math.atan(TINY_IMAGE_SIZE / (2 * float(cropped.intrinsics[0, 0, 0])))
    assert float(with_camera[0, 0]) == pytest.approx(expected_fov)

    with pytest.raises(ValueError, match="field of view"):
        pipeline(condition)
    with pytest.raises(TypeError, match="MultiViewCondition"):
        tiny_pipeline(multiview=True)(condition, camera_angle_x=0.5)


def test_projection_follows_the_view_aligned_y_up_grid():
    # One channel holds the pixel row and one the column, so samples reveal where each cell projects.
    rows, columns = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    feature_map = torch.stack([rows, columns])
    distance = camera_distance_from_fov(0.6)
    cells = torch.tensor([[1, 0, 1], [1, 2, 1], [0, 1, 1], [2, 1, 1], [1, 1, 0], [1, 1, 2]])
    samples = project_grid_features(
        [feature_map],
        cells,
        grid_resolution=3,
        camera_to_world=front_view_camera(torch.tensor(distance)),
        camera_angle_x=torch.tensor(0.6),
        mesh_scale=1.0,
        image_size=16,
    )
    bottom, top, left, right, back, front = samples
    assert top[0] < bottom[0]  # grid +y is up: higher cells land on smaller image rows
    assert right[1] > left[1]  # grid +x is image right
    # Grid +z points at the camera: the front cell's offset from the centre is magnified by perspective.
    centre = samples.new_tensor([7.5, 7.5])
    shifted = torch.tensor([[1, 0, 2], [1, 0, 0]])
    near, far = project_grid_features(
        [feature_map],
        shifted,
        grid_resolution=3,
        camera_to_world=front_view_camera(torch.tensor(distance)),
        camera_angle_x=torch.tensor(0.6),
        mesh_scale=1.0,
        image_size=16,
    )
    assert (near - centre).abs()[0] > (far - centre).abs()[0]
    torch.testing.assert_close(front, back)  # on the optical axis


def test_multi_view_pipeline_runs_and_validates_inputs():
    pipeline = tiny_pipeline(multiview=True)
    images = torch.rand(2, 4, 24, 24, generator=torch.Generator().manual_seed(2))
    condition = MultiViewCondition(images=images, cameras=_square_rig(2, 0.5))
    output = pipeline(
        condition,
        formats=("sparse_structure", "texture_slat"),
        generator=torch.Generator().manual_seed(3),
        **STEPS,
    )
    assert [type(value) for value in output.objects] == [SparseVoxelAsset, SparseVoxelAsset]
    with pytest.raises(ValueError, match="come from the condition cameras"):
        pipeline(condition, camera_angle_x=0.5)
    with pytest.raises(ValueError, match="square"):
        pipeline(MultiViewCondition(images=images[:, :, :20], cameras=_rig_for(20, 24)))


def _rig_for(height: int, width: int) -> CameraRig:
    rig = _square_rig(2, 0.5)
    return CameraRig(rig.world_to_camera, rig.intrinsics, torch.tensor([[height, width]] * 2))


_DECODERS = ("sparse_structure_decoder", "shape_slat_decoder", "pbr_decoder")


def _external_modules(pipeline) -> dict:
    names = ("image_encoder", "feature_upsampler", *_DECODERS)
    return {name: getattr(pipeline, name) for name in names}


def test_save_load_auto_and_preset_validation(tmp_path, tiny_pixal3d_pipeline, local_external_components):
    pipeline = tiny_pixal3d_pipeline
    calls = local_external_components(Pixal3DImageTo3DPipeline, **_external_modules(pipeline))
    pipeline.save_pretrained(tmp_path)
    # DINOv3, NAF, and the TRELLIS decoders are third-party: never saved, loaded from their sources; no depth
    # estimator is named here.
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_dir()) == [
        "shape_slat_flow_model",
        "shape_slat_flow_model_1024",
        "shape_slat_scheduler",
        "sparse_structure_flow_model",
        "sparse_structure_scheduler",
        "texture_slat_flow_model_1024",
        "texture_slat_scheduler",
    ]
    loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is Pixal3DImageTo3DPipeline
    assert loaded.config.default_pipeline_type == "1536_cascade"
    assert loaded.config.multiview is False
    torch.testing.assert_close(
        loaded.feature_upsampler.state_dict(), tiny_pixal3d_pipeline.feature_upsampler.state_dict()
    )
    with pytest.raises(ValueError, match="pipeline_type must be one of"):
        loaded(ImageCondition(_rgba()), camera_angle_x=0.4, pipeline_type="512")
    # Upstream runs DINOv3 and NAF in float32 under a bfloat16 load.
    low_precision = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True, dtype=torch.bfloat16)
    low_precision.disable_background_removal()
    # The gated DINOv3 loads from its own repository, in float32 under a bfloat16 load as upstream runs it.
    encoders = [(call[0], call[3]) for call in calls if call[0] in ("image_encoder", "feature_upsampler")]
    assert encoders == [
        ("image_encoder", None),
        ("feature_upsampler", None),
        ("image_encoder", torch.float32),
        ("feature_upsampler", torch.float32),
    ]
    assert {call[0] for call in calls} == {"image_encoder", "feature_upsampler", *_DECODERS}
    assert (
        low_precision.image_encoder.dtype is torch.float32 and low_precision.feature_upsampler.dtype is torch.float32
    )
    assert low_precision.sparse_structure_flow_model.dtype is torch.bfloat16
    low_precision.set_progress_bar_config(disable=True)
    output = low_precision(ImageCondition(_rgba()), camera_angle_x=0.4, formats=("sparse_structure",), **STEPS)
    assert isinstance(output.objects[0], SparseVoxelAsset)


def test_meshes_stay_in_the_view_aligned_frame(monkeypatch, tiny_pixal3d_pipeline, slab_ovoxel):
    # Tiny random decoders cross no voxel edge, so decode a known surface tagged with the Pixal3D frame.
    slab = dataclasses.replace(slab_ovoxel, coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP)
    monkeypatch.setattr(tiny_pixal3d_pipeline, "decode_ovoxel", lambda *args, **kwargs: (slab,))
    ovoxel, mesh = tiny_pixal3d_pipeline(
        ImageCondition(_rgba()), camera_angle_x=0.4, formats=("o_voxel", "mesh"), **STEPS
    ).objects
    assert type(mesh) is MeshAsset
    assert ovoxel.metadata["family"] == mesh.metadata["family"] == "pixal3d"
    assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Y_UP
    # Grid +y (up in the conditioning view) becomes world +z; grid +z (towards the camera) becomes world -y.
    world = mesh.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Z_UP)
    torch.testing.assert_close(world.vertices[:, 2], mesh.vertices[:, 1])
    torch.testing.assert_close(world.vertices[:, 1], -mesh.vertices[:, 2])


def test_projection_samples_bfloat16_features_on_a_float32_grid():
    feature_map = torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(4)).bfloat16()
    cells = torch.stack(torch.meshgrid(*[torch.arange(8)] * 3, indexing="ij"), dim=-1).reshape(-1, 3)
    arguments = {
        "grid_resolution": 8,
        "camera_to_world": front_view_camera(torch.tensor(camera_distance_from_fov(0.3))),
        "camera_angle_x": torch.tensor(0.3),
        "mesh_scale": 1.0,
        "image_size": 1024,
    }
    low_precision = project_grid_features([feature_map], cells, **arguments)
    reference = project_grid_features([feature_map.float()], cells, **arguments)
    assert low_precision.dtype is torch.float32
    torch.testing.assert_close(low_precision, reference, atol=0.0, rtol=0.0)


def test_missing_field_of_view_comes_from_the_depth_estimator(monkeypatch):
    pipeline = tiny_pipeline()
    with pytest.raises(ValueError, match="load the MoGe-2 depth_estimator"):
        pipeline.prepare_views(ImageCondition(_rgba()), image_sizes=(TINY_IMAGE_SIZE,))
    pipeline.register_modules(depth_estimator=MoGe2Model(**tiny_moge2_config()).eval())
    seen = []
    monkeypatch.setattr(
        pipeline, "estimate_camera_angle_x", lambda condition: seen.append(condition) or torch.tensor([0.5])
    )
    views = pipeline.prepare_views(ImageCondition(_rgba()), image_sizes=(TINY_IMAGE_SIZE,))
    assert len(seen) == 1 and views.camera_angle_x.tolist() == [[0.5]]
    # An explicit field of view skips the estimate.
    pipeline.prepare_views(ImageCondition(_rgba()), image_sizes=(TINY_IMAGE_SIZE,), camera_angle_x=0.4)
    assert len(seen) == 1


def test_plain_load_keeps_the_bfloat16_torso_and_folders_without_an_estimator_load(
    tmp_path, local_external_components
):
    pipeline = tiny_pipeline()
    local_external_components(Pixal3DImageTo3DPipeline, **_external_modules(pipeline))
    config = {**tiny_config(Pixal3DSparseStructureFlowModel), "dtype": "bfloat16"}
    flow = Pixal3DSparseStructureFlowModel(**config)
    pipeline.register_modules(sparse_structure_flow_model=flow)
    pipeline.save_pretrained(tmp_path)
    # Published folders predate the depth estimator.
    index = json.loads((tmp_path / "model_index.json").read_text())
    index.pop("depth_estimator", None)
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    loaded = Pixal3DImageTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.depth_estimator is None
    dtypes = {name: value.dtype for name, value in loaded.sparse_structure_flow_model.state_dict().items()}
    assert dtypes == {name: value.dtype for name, value in flow.state_dict().items()}
    assert torch.bfloat16 in dtypes.values() and torch.float32 in dtypes.values()
