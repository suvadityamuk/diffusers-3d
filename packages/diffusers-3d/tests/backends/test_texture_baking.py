from __future__ import annotations

import pytest
import torch

from diffusers_3d import CoordinateSystem, MeshAsset
from diffusers_3d.backends.texture_baking import bake_texture, rasterize_uv_atlas, sphere_hammersley_cameras

HALF = 0.3
# Face colour: axis channel = 1.0 on the +side, 0.5 on the -side.
_CORNERS = torch.tensor(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=torch.float32,
)
# (axis, positive side, four corner ids counter-clockwise seen from outside)
_FACES = (
    (2, False, (0, 3, 2, 1)),
    (2, True, (4, 5, 6, 7)),
    (1, False, (0, 1, 5, 4)),
    (1, True, (2, 3, 7, 6)),
    (0, True, (1, 2, 6, 5)),
    (0, False, (0, 4, 7, 3)),
)


def _cube_with_manual_atlas() -> MeshAsset:
    """Twelve triangles, each cube face its own UV square in a 3x2 grid (no unwrapper needed)."""

    vertices, faces, uvs = [], [], []
    for slot, (_, _, corner_ids) in enumerate(_FACES):
        column, row = slot % 3, slot // 3
        base = len(vertices)
        for local, corner in enumerate(corner_ids):
            vertices.append(_CORNERS[corner] * HALF)
            u = column / 3 + (0.05 if local in (0, 3) else 0.28)
            v = row / 2 + (0.05 if local in (0, 1) else 0.45)
            uvs.append(torch.tensor([u, v]))
        faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return MeshAsset(
        torch.stack(vertices),
        torch.tensor(faces),
        uvs=torch.stack(uvs),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )


def _face_color(points: torch.Tensor) -> torch.Tensor:
    axis = (points.abs() - HALF).abs().argmin(dim=-1)
    positive = torch.gather(points, -1, axis.unsqueeze(-1)).squeeze(-1) > 0
    color = torch.zeros_like(points)
    color[torch.arange(points.shape[0]), axis] = torch.where(positive, 1.0, 0.5)
    return color


def _render_cube(cameras, image_size: int):
    """Analytic ray cast of the cube: colour by face, camera-space depth, binary alpha."""

    count = cameras.world_to_camera.shape[0]
    colors = torch.zeros(count, 3, image_size, image_size)
    depths = torch.zeros(count, 1, image_size, image_size)
    alphas = torch.zeros(count, 1, image_size, image_size)
    rows, columns = torch.meshgrid(torch.arange(image_size) + 0.5, torch.arange(image_size) + 0.5, indexing="ij")
    for index in range(count):
        intrinsics = cameras.intrinsics[index]
        camera_to_world = torch.linalg.inv(cameras.world_to_camera[index])
        camera_dirs = torch.stack(
            [
                (columns - intrinsics[0, 2]) / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) / intrinsics[1, 1],
                torch.ones_like(rows),
            ],
            dim=-1,
        )
        world_dirs = camera_dirs @ camera_to_world[:3, :3].T
        origin = camera_to_world[:3, 3]
        inverse = 1.0 / world_dirs
        near = torch.minimum((-HALF - origin) * inverse, (HALF - origin) * inverse).amax(-1)
        far = torch.maximum((-HALF - origin) * inverse, (HALF - origin) * inverse).amin(-1)
        hit = (far > near) & (far > 0)
        points = origin + near.unsqueeze(-1) * world_dirs
        colors[index] = torch.where(
            hit.unsqueeze(-1), _face_color(points.reshape(-1, 3)).reshape_as(points), 0.0
        ).permute(2, 0, 1)
        depths[index, 0] = torch.where(hit, near * camera_dirs[..., 2], 0.0)
        alphas[index, 0] = hit.float()
    return colors, depths, alphas


def test_hammersley_cameras_look_at_the_origin_from_the_sphere():
    cameras = sphere_hammersley_cameras(16, image_size=64, radius=2.0, coordinate_system="right_handed_z_up")
    camera_to_world = torch.linalg.inv(cameras.world_to_camera)
    eyes = camera_to_world[:, :3, 3]
    torch.testing.assert_close(eyes.norm(dim=1), torch.full((16,), 2.0))
    # The optical axis (camera +z) points from the eye to the origin.
    torch.testing.assert_close(camera_to_world[:, :3, 2], torch.nn.functional.normalize(-eyes, dim=1))
    # Views cover both hemispheres rather than one ring.
    assert eyes[:, 2].min() < -0.5 and eyes[:, 2].max() > 0.5
    assert cameras.intrinsics[0, 0, 2] == 32.0


def test_uv_atlas_rasterization_places_texels_on_the_surface():
    mesh = _cube_with_manual_atlas()
    surface = rasterize_uv_atlas(mesh, 48)
    assert surface.texel_index.unique().shape[0] == surface.texel_index.shape[0]
    # Every texel's point lies on a cube face and its normal is that face's outward axis.
    on_face = ((surface.positions.abs() - HALF).abs().amin(dim=1) < 1e-5).all()
    assert bool(on_face)
    axis = (surface.positions.abs() - HALF).abs().argmin(dim=1)
    outward = torch.gather(surface.normals, 1, axis.unsqueeze(1)).squeeze(1)
    sign = torch.gather(surface.positions, 1, axis.unsqueeze(1)).squeeze(1).sign()
    torch.testing.assert_close(outward, sign)
    # Each of the six squares spans about (0.23 * 48) x (0.4 * 48) texels, plus a conservative edge ring.
    assert 6 * 10 * 18 < surface.texel_index.shape[0] < 6 * 13 * 21


def test_bake_texture_recovers_face_colours_and_fills_unseen_texels():
    mesh = _cube_with_manual_atlas()
    texture_size = 48
    surface = rasterize_uv_atlas(mesh, texture_size)
    cameras = sphere_hammersley_cameras(24, image_size=96, coordinate_system="right_handed_z_up")
    colors, depths, alphas = _render_cube(cameras, 96)

    texture, observed = bake_texture(
        surface, cameras, colors, depths, alphas, texture_size=texture_size, depth_tolerance=0.02
    )

    assert texture.shape == (texture_size, texture_size, 3)
    covered_observed = observed.reshape(-1)[surface.texel_index]
    assert covered_observed.float().mean() > 0.95
    expected = _face_color(surface.positions)
    baked = texture.reshape(-1, 3)[surface.texel_index]
    error = (baked - expected).abs().amax(dim=1)
    # Bilinear sampling mixes colours within a pixel of the cube edges; away from them the bake is exact.
    axis = (surface.positions.abs() - HALF).abs().argmin(dim=1)
    in_plane = surface.positions.abs().scatter(1, axis.unsqueeze(1), 0.0)
    interior = (HALF - in_plane).topk(2, dim=1, largest=False).values.amin(dim=1) > 0.05
    assert bool((error[interior & covered_observed] < 0.05).all())
    assert (error[covered_observed] < 0.05).float().mean() > 0.8
    # Unobserved texels inside the atlas are filled from neighbours, never left black.
    assert bool((texture.reshape(-1, 3)[surface.texel_index].sum(dim=1) > 0).all())

    # Depth agreement is what rejects occluded views: a far-off tolerance lets back faces bleed through.
    loose, _ = bake_texture(surface, cameras, colors, depths, alphas, texture_size=texture_size, depth_tolerance=10.0)
    loose_error = (loose.reshape(-1, 3)[surface.texel_index] - expected).abs().amax(dim=1)
    assert loose_error.mean() > error.mean()


@pytest.mark.portable
def test_trellis_glb_facade_bakes_a_textured_mesh_with_fake_gsplat(monkeypatch):
    pytest.importorskip("xatlas")
    import sys
    import types
    from importlib.machinery import ModuleSpec

    from diffusers_3d import (
        BackendCapability,
        BackendLicenseClass,
        BackendRegistry,
        BackendSpec,
        BackendSupportLevel,
        GaussianSplatAsset,
        TrellisGlbPostprocessFacade,
    )

    module = types.ModuleType("gsplat")

    def rasterization(**kwargs):
        count = kwargs["viewmats"].shape[0]
        rendered = torch.zeros(count, kwargs["height"], kwargs["width"], 4)
        rendered[..., :3] = torch.tensor([0.25, 0.5, 0.75])
        rendered[..., 3] = 2.0  # every pixel at the orbit radius
        return rendered, torch.ones(count, kwargs["height"], kwargs["width"], 1), {}

    module.rasterization = rasterization
    monkeypatch.setitem(sys.modules, "gsplat", module)

    def spec(name, capability, level):
        return BackendSpec(
            name=name,
            import_names=(name,),
            distribution_names=(name,),
            capabilities=frozenset({capability}),
            support_level=level,
            license_class=BackendLicenseClass.PERMISSIVE,
            devices=frozenset({"cpu"}),
            dtypes=frozenset({"float32"}),
            differentiable=name == "gsplat",
            install_hint=f"Install {name}",
        )

    registry = BackendRegistry(
        (
            spec("gsplat", BackendCapability.GAUSSIAN_RASTERIZATION, BackendSupportLevel.ACCELERATED),
            spec("xatlas", BackendCapability.GEOMETRY_PROCESSING, BackendSupportLevel.PORTABLE),
        ),
        module_finder=lambda candidate: ModuleSpec(candidate, loader=None),
        version_getter=lambda _: "1.0",
    )
    mesh = _cube_with_manual_atlas()
    gaussians = GaussianSplatAsset(
        means=torch.zeros(1, 3),
        log_scales=torch.zeros(1, 3),
        quaternions_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logits=torch.zeros(1, 1),
        sh_coefficients=torch.ones(1, 1, 3),
        active_sh_degree=0,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    calls = []

    class Processor:
        def process_geometry(self, mesh, *, operation, parameters=None):
            calls.append((operation, parameters))
            return mesh

    facade = TrellisGlbPostprocessFacade(registry=registry)
    textured = facade.to_textured_mesh(
        mesh,
        gaussians,
        device="cpu",
        simplify_ratio=0.5,
        texture_size=32,
        num_views=12,
        render_size=32,
        depth_tolerance=1.0,
        mesh_processor=Processor(),
    )

    assert [operation for operation, _ in calls] == ["repair", "simplify"]
    assert calls[1][1] == {"target_faces": 6}
    assert textured.uvs is not None and textured.uvs.shape[0] == textured.vertices.shape[0]
    assert textured.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    material = textured.materials[0]
    assert material.base_color.shape == (32, 32, 3)
    torch.testing.assert_close(material.base_color.reshape(-1, 3).mean(dim=0), torch.tensor([0.25, 0.5, 0.75]))
    assert float(material.roughness) == 1.0
    assert textured.metadata["texture"] == "baked-from-gaussians"
    assert textured.metadata["observed_texel_fraction"] == 1.0
    with pytest.raises(ValueError, match="same coordinate system"):
        facade.to_textured_mesh(mesh.to_coordinate_system("right_handed_y_up"), gaussians, device="cpu")
