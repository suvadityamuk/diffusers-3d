import sys
import types

import torch

from diffusers_3d import CameraRig, CoordinateSystem, GaussianSplatAsset, GsplatBackend


def test_gsplat_fake_module_converts_canonical_assets_and_scales_intrinsics(monkeypatch):
    captured = {}
    module = types.ModuleType("gsplat")

    def rasterization(**kwargs):
        captured.update(kwargs)
        rendered = torch.zeros(1, kwargs["height"], kwargs["width"], 4)
        rendered[..., :3] = 0.25
        rendered[..., 3] = 2.0
        alpha = torch.full((1, kwargs["height"], kwargs["width"], 1), 0.75)
        return rendered, alpha, {}

    module.rasterization = rasterization
    monkeypatch.setitem(sys.modules, "gsplat", module)
    backend = GsplatBackend(device="cpu")
    gaussians = GaussianSplatAsset(
        means=torch.tensor([[0.0, 0.0, 0.0]]),
        log_scales=torch.zeros(1, 3),
        quaternions_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logits=torch.zeros(1, 1),
        sh_coefficients=torch.ones(1, 1, 3),
        active_sh_degree=0,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    cameras = CameraRig(
        world_to_camera=torch.eye(4).unsqueeze(0),
        intrinsics=torch.tensor([[[4.0, 0.0, 4.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]]),
        image_sizes=torch.tensor([[4, 8]], dtype=torch.int64),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    output = backend.rasterize_gaussians(gaussians, cameras, image_size=(8, 4))
    assert output["color"].shape == (1, 3, 8, 4)
    assert output["depth"].shape == (1, 1, 8, 4)
    assert output["alpha"].shape == (1, 1, 8, 4)
    torch.testing.assert_close(captured["Ks"][0, 0], torch.tensor([2.0, 0.0, 2.0]))
    torch.testing.assert_close(captured["Ks"][0, 1], torch.tensor([0.0, 4.0, 4.0]))
    torch.testing.assert_close(captured["scales"], gaussians.log_scales.exp())
    torch.testing.assert_close(captured["opacities"], gaussians.opacity_logits.sigmoid().reshape(-1))
