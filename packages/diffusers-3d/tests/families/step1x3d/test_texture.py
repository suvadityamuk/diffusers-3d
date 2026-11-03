from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from safetensors.torch import save_file

from diffusers_3d import (
    AutoPipelineFor3D,
    ImageCondition,
    MeshAsset,
    Object3DPipelineOutput,
    Step1X3DMeshTexturePipeline,
    Step1X3DMultiViewAttnProcessor,
    convert_step1x3d_texture_checkpoint,
    preprocess_step1x3d_reference_image,
)
from diffusers_3d.families.step1x3d.texture_conversion import main, shifted_snr_betas
from diffusers_3d.families.step1x3d.texture_pipeline import (
    render_step1x3d_control_images,
    step1x3d_orthographic_projection,
    step1x3d_view_matrices,
    view_frame,
)
from diffusers_3d.families.step1x3d.texture_sync import Step1X3DLatentTextureSync, nearest_valid_index

from .conftest import make_tiny_step1x3d_texture_components

pytestmark = [pytest.mark.integration, pytest.mark.portable]


def _reference_image() -> ImageCondition:
    image = torch.linspace(0.0, 1.0, 4 * 20 * 16).reshape(4, 20, 16)
    image[3] = 0
    image[3, 4:15, 3:12] = 1
    return ImageCondition(image)


def test_nearest_valid_index_finds_a_nearest_texel():
    valid = torch.rand(37, 29, generator=torch.Generator().manual_seed(0)) > 0.97
    nearest = nearest_valid_index(valid)
    rows, columns = torch.meshgrid(torch.arange(37), torch.arange(29), indexing="ij")
    seeds = valid.nonzero()
    distance = (rows.reshape(-1, 1) - seeds[:, 0]) ** 2 + (columns.reshape(-1, 1) - seeds[:, 1]) ** 2
    found = (rows.flatten() - nearest // 29) ** 2 + (columns.flatten() - nearest % 29) ** 2
    assert valid.flatten()[nearest].all()
    assert torch.equal(found, distance.min(dim=1).values)
    assert (nearest_valid_index(torch.zeros(4, 4, dtype=torch.bool)) == -1).all()


def _voronoi_kernel(valid: np.ndarray) -> np.ndarray:
    """A line-by-line restatement of the released CuPy voronoiPass kernel and its ping-pong driver."""

    height, width = valid.shape
    ping = np.where(valid.flatten(), np.arange(height * width), -1)
    pong = ping.copy()
    step = max(height, width) // 2
    while step:
        for k in range(height * width):
            for row_step in (-1, 0, 1):
                for column_step in (-1, 0, 1):
                    source = k + step * row_step * width + step * column_step
                    if source < 0 or source >= height * width or ping[source] == -1:
                        continue
                    if pong[k] == -1:
                        pong[k] = ping[source]
                        continue
                    current = (k // width - pong[k] // width) ** 2 + (k % width - pong[k] % width) ** 2
                    jump = (k // width - ping[source] // width) ** 2 + (k % width - ping[source] % width) ** 2
                    if jump < current:
                        pong[k] = ping[source]
        ping, pong = pong, ping
        step //= 2
    return ping


@pytest.mark.parametrize("shape", [(9, 9), (7, 13), (16, 5)])
def test_nearest_valid_index_is_the_released_voronoi_kernel(shape):
    rng = np.random.default_rng(sum(shape))
    for density in (0.03, 0.2):
        valid = rng.random(shape) < density
        assert np.array_equal(nearest_valid_index(torch.from_numpy(valid)).numpy(), _voronoi_kernel(valid))


def test_view_layout_matches_the_released_cameras():
    camera_to_world = torch.linalg.inv(step1x3d_view_matrices())
    positions = camera_to_world[:, :3, 3]
    expected = torch.tensor([[0, -1.8, 0], [1.8, 0, 0], [0, 1.8, 0], [-1.8, 0, 0], [0, 0, 1.8], [0, 0, -1.8]])
    torch.testing.assert_close(positions, expected, atol=1e-3, rtol=0)
    # The front view looks along +Y with Z up and X to the right.
    torch.testing.assert_close(camera_to_world[0, :3, :3], torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]))


def test_control_images_encode_position_and_normal(icosphere_mesh):
    world_to_clip = step1x3d_orthographic_projection() @ step1x3d_view_matrices()
    positions = icosphere_mesh.vertices * 0.5 / icosphere_mesh.vertices.abs().max()
    images = render_step1x3d_control_images(positions, icosphere_mesh.faces, world_to_clip, 32)

    assert images.shape == (6, 6, 32, 32)
    assert torch.equal(images[:, :, 0, 0], torch.full((6, 6), 0.5))
    # The pixel in front of the sphere centre faces the front camera, which sits at -Y.
    front = images[0, :, 15:17, 15:17].mean(dim=(1, 2))
    torch.testing.assert_close(front[3:], torch.tensor([0.5, 0.0, 0.5]), atol=0.03, rtol=0)
    assert front[1] < 0.1


def test_reference_image_is_fitted_on_grey():
    framed = preprocess_step1x3d_reference_image(_reference_image(), image_size=40)
    assert framed.shape == (3, 40, 40)
    # The 12-row crop (one transparent row above) fills 36 rows, leaving two grey rows above and below.
    grey = (framed - 127 / 255).abs().sum(dim=0) == 0
    assert grey[:2].all() and grey[38:].all() and not grey[20, 20]


def test_texture_sync_bakes_back_a_rendered_texture(tiny_step1x3d_texture_pipeline, icosphere_mesh):
    pytest.importorskip("xatlas")
    meshes, positions = tiny_step1x3d_texture_pipeline.prepare_mesh(icosphere_mesh, max_faces=None)
    world_to_camera = step1x3d_view_matrices()
    sync = Step1X3DLatentTextureSync(
        positions,
        torch.from_numpy(meshes.faces),
        torch.from_numpy(meshes.sync_uvs),
        step1x3d_orthographic_projection() @ world_to_camera,
        torch.linalg.inv(world_to_camera)[:, :3, 3],
        image_size=24,
        texture_size=48,
        visibility_size=64,
        uv_faces=torch.from_numpy(meshes.sync_uv_faces),
    )
    texture = torch.rand(4, 48, 48, generator=torch.Generator().manual_seed(0))
    views = sync.render(texture)
    assert views.shape == (6, 4, 24, 24)
    assert sync.masks.mean() > 0.2
    assert torch.equal(views * (1 - sync.masks), (1 - sync.masks).expand_as(views))
    # Each view averages back the texels it sampled; blending also mixes in other views' nearest-texel fills.
    sampled = sync.counts > 0
    # Culling keeps the faces turned towards each camera; back faces would bake zero cosines.
    assert all(sync.cos[view, 0][sampled[view]].mean() > 0.4 for view in range(6))
    means = sync._mean(views)
    for view in range(6):
        torch.testing.assert_close(means[view][:, sampled[view]], texture.flatten(1)[:, sampled[view]])
    constant = torch.arange(4.0)[:, None, None].expand(4, 48, 48)
    torch.testing.assert_close(sync.bake(sync.render(constant), exp=3.0), constant)


def test_bake_views_covers_the_atlas(tiny_step1x3d_texture_pipeline, icosphere_mesh):
    pytest.importorskip("xatlas")
    meshes, _ = tiny_step1x3d_texture_pipeline.prepare_mesh(icosphere_mesh, max_faces=None)
    positions = view_frame(meshes.vertices[meshes.final_mapping])
    color = torch.tensor([0.2, 0.4, 0.6])
    views = color[None, :, None, None].expand(6, 3, 32, 32)
    faces, uvs = torch.from_numpy(meshes.final_faces), torch.from_numpy(meshes.final_uvs)
    texture = tiny_step1x3d_texture_pipeline.bake_views(views, positions, faces, uvs, texture_size=128)
    # The released inpainting hands on 8-bit texels.
    torch.testing.assert_close(texture, (color * 255).floor().div(255).expand(128, 128, 3), atol=1 / 255, rtol=0)


def test_multiview_attention_mixes_rows_and_columns_only():
    torch.manual_seed(0)
    attention = Attention(8, heads=2, dim_head=4, processor=Step1X3DMultiViewAttnProcessor(8))
    hidden_states = torch.randn(6, 16, 8)
    with torch.no_grad():
        output = attention(hidden_states)
        changed = hidden_states.clone()
        changed[4] += 1  # the top view
        changed_output = attention(changed)
        plain = Attention(8, heads=2, dim_head=4, processor=AttnProcessor2_0())
        plain.load_state_dict(attention.state_dict(), strict=False)
        for layer in (attention.processor.to_out_mv[0], attention.processor.to_out_ref[0]):
            layer.weight.zero_()
            layer.bias.zero_()
        cache = {}
        attention(hidden_states, reference_cache=cache)
        unbranched = attention(hidden_states, reference_hidden_states=cache)

    # Rows join views 0-3 and columns join views 0, 2, 4, 5: the right and left views never see the top one.
    assert torch.equal(changed_output[[1, 3]], output[[1, 3]])
    assert not torch.allclose(changed_output[[0, 2, 5]], output[[0, 2, 5]])
    assert torch.equal(cache[attention.processor], hidden_states)
    torch.testing.assert_close(unbranched, plain(hidden_states))


def test_pipeline_textures_meshes_and_round_trips(tiny_step1x3d_texture_pipeline, icosphere_mesh, tmp_path):
    pytest.importorskip("xatlas")
    pipeline = tiny_step1x3d_texture_pipeline
    processors = pipeline.unet.attn_processors
    arguments = {
        "num_inference_steps": 4,
        "resolution": 32,
        "latent_texture_size": 32,
        "texture_size": 64,
        "max_faces": None,
    }
    output = pipeline(icosphere_mesh, _reference_image(), generator=torch.Generator().manual_seed(0), **arguments)

    assert isinstance(output, Object3DPipelineOutput)
    (mesh,) = output.objects
    assert isinstance(mesh, MeshAsset) and mesh.uvs is not None
    assert mesh.materials[0].base_color.shape == (64, 64, 3)
    assert output.previews[0].shape == (6, 3, 32, 32)
    assert output.latents.latents.shape == (1, 6, 4, 16, 16)
    assert pipeline.unet.attn_processors == processors

    pipeline.save_pretrained(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "adapter",
        "model_index.json",
        "mv_adapter",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
    ]
    external = {name: getattr(pipeline, name) for name in Step1X3DMeshTexturePipeline.external_components}
    loaded = AutoPipelineFor3D.from_pretrained(tmp_path, **external)
    assert type(loaded) is Step1X3DMeshTexturePipeline
    reloaded = loaded(icosphere_mesh, _reference_image(), generator=torch.Generator().manual_seed(0), **arguments)
    torch.testing.assert_close(reloaded.latents.latents, output.latents.latents)
    torch.testing.assert_close(reloaded.objects[0].materials[0].base_color, mesh.materials[0].base_color)


def test_shifted_snr_betas_lower_the_terminal_snr():
    scheduler = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
    shifted = DDPMScheduler.from_config(
        scheduler.config, trained_betas=shifted_snr_betas(scheduler.alphas_cumprod).tolist()
    )
    snr = scheduler.alphas_cumprod / (1 - scheduler.alphas_cumprod)
    shifted_snr = shifted.alphas_cumprod / (1 - shifted.alphas_cumprod)
    torch.testing.assert_close(shifted_snr[0], snr[0])
    torch.testing.assert_close(shifted_snr[-1], snr[-1] / 8)


def _write_tiny_release(root, tokenizer):
    components = make_tiny_step1x3d_texture_components(tokenizer)
    base = root / "base"
    for name in ("unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
        components[name].save_pretrained(base / name)
    (base / "LICENSE.md").write_text("CreativeML Open RAIL++-M\n")
    components["vae"].save_pretrained(root / "vae")
    torch.manual_seed(3)
    release = {
        **{
            f"adapter.{key}": torch.randn_like(value)
            for key, value in components["adapter"].adapter.state_dict().items()
        },
        **{key: torch.randn_like(value) for key, value in components["mv_adapter"].state_dict().items()},
    }
    (root / "Step1X-3D-Texture").mkdir()
    save_file(release, root / "Step1X-3D-Texture" / "step1x-3d-ig2v.safetensors")
    return components, release


def test_conversion_writes_only_the_release_and_loads_it_onto_stock_sdxl_components(tiny_clip_tokenizer, tmp_path):
    components, release = _write_tiny_release(tmp_path, tiny_clip_tokenizer)
    output = tmp_path / "converted"
    # Revisions are ignored for local directories.
    main(
        [
            str(tmp_path / "Step1X-3D-Texture"),
            str(output),
            "--base-model",
            str(tmp_path / "base"),
            "--vae-model",
            str(tmp_path / "vae"),
        ]
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "LICENSE-SDXL.md",
        "adapter",
        "model_index.json",
        "mv_adapter",
        "scheduler",
        "step1x3d_texture_conversion.json",
        "tokenizer",
        "tokenizer_2",
    ]
    external = {name: components[name] for name in Step1X3DMeshTexturePipeline.external_components}
    pipeline = AutoPipelineFor3D.from_pretrained(output, task="mesh-texturing", **external)
    state = {**{f"adapter.{k}": v for k, v in pipeline.adapter.adapter.state_dict().items()}}
    state.update(pipeline.mv_adapter.state_dict())
    assert state.keys() == release.keys()
    assert all(torch.equal(state[key], value) for key, value in release.items())
    expected = shifted_snr_betas(components["scheduler"].alphas_cumprod)
    assert torch.equal(pipeline.scheduler.betas, expected)
    assert (output / "LICENSE-SDXL.md").read_text() == "CreativeML Open RAIL++-M\n"
    assert json.loads((output / "step1x3d_texture_conversion.json").read_text())["base_model"][0] == str(
        tmp_path / "base"
    )

    extra = dict(release, **{"unexpected.weight": torch.zeros(1)})
    save_file(extra, tmp_path / "Step1X-3D-Texture" / "step1x-3d-ig2v.safetensors")
    with pytest.raises(ValueError, match="unexpected Step1X-3D texture keys"):
        convert_step1x3d_texture_checkpoint(
            tmp_path / "Step1X-3D-Texture",
            tmp_path / "rejected",
            base_model=tmp_path / "base",
            vae_model=tmp_path / "vae",
        )
