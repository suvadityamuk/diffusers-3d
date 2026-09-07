from __future__ import annotations

from collections.abc import Mapping

import torch

from ..objects import CameraRig, GaussianSplatAsset
from ._optional import load_explicit_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability


class GsplatBackend:
    """Explicit adapter from package-owned Gaussian and camera conventions to ``gsplat``.

    The adapter uses gsplat's documented rasterization API directly. It does not
    contain or import Graphdeco rasterizer code.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self._gsplat = load_explicit_backend(
            "gsplat",
            "gsplat",
            (BackendCapability.GAUSSIAN_RASTERIZATION,),
            device=device,
            dtype=dtype,
            differentiable=True,
            registry=registry,
        )

    def rasterize_gaussians(
        self,
        gaussians: GaussianSplatAsset,
        cameras: CameraRig,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """Rasterize RGB, alpha, and expected depth as channel-first buffers."""

        if type(gaussians) is not GaussianSplatAsset:
            raise TypeError("gaussians must be an exact GaussianSplatAsset")
        if type(cameras) is not CameraRig:
            raise TypeError("cameras must be an exact CameraRig")
        gaussians.validate(expensive=True)
        cameras.validate()
        if gaussians.coordinate_system is not cameras.coordinate_system:
            raise ValueError("gaussians and cameras must use the same coordinate system")
        identity = torch.eye(4, device=gaussians.device, dtype=gaussians.transform.dtype)
        if not torch.allclose(gaussians.transform, identity):
            raise ValueError(
                "GsplatBackend requires an identity Gaussian object transform; bake the transform into Gaussian "
                "parameters before rasterization"
            )
        if gaussians.device != cameras.device:
            raise ValueError("gaussians and cameras must be on the same device")
        if gaussians.device.type != self.device.type or (
            self.device.index is not None and gaussians.device.index != self.device.index
        ):
            raise ValueError(f"GsplatBackend was configured for {self.device}, got {gaussians.device}")
        if gaussians.means.dtype is not self.dtype:
            raise ValueError(f"GsplatBackend was configured for {self.dtype}, got {gaussians.means.dtype}")
        if cameras.world_to_camera.dtype is not self.dtype or cameras.intrinsics.dtype is not self.dtype:
            raise ValueError(f"GsplatBackend requires camera tensors with configured dtype {self.dtype}")

        if image_size is None:
            if not bool((cameras.image_sizes == cameras.image_sizes[:1]).all()):
                raise ValueError("image_size is required when camera image sizes differ")
            height, width = (int(value) for value in cameras.image_sizes[0].tolist())
        else:
            if (
                not isinstance(image_size, tuple)
                or len(image_size) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in image_size)
            ):
                raise ValueError("image_size must be a positive (height, width) integer pair")
            height, width = image_size
        target_sizes = cameras.image_sizes.new_tensor([height, width]).expand_as(cameras.image_sizes)
        intrinsics = cameras.intrinsics
        if not torch.equal(cameras.image_sizes, target_sizes):
            scale_y = height / cameras.image_sizes[:, 0].to(dtype=intrinsics.dtype)
            scale_x = width / cameras.image_sizes[:, 1].to(dtype=intrinsics.dtype)
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, :] *= scale_x[:, None]
            intrinsics[:, 1, :] *= scale_y[:, None]

        rasterization = getattr(self._gsplat, "rasterization", None)
        if not callable(rasterization):
            raise RuntimeError("the selected gsplat build does not expose the documented rasterization function")
        result = rasterization(
            means=gaussians.means,
            quats=gaussians.quaternions_wxyz,
            scales=gaussians.log_scales.exp(),
            opacities=gaussians.opacity_logits.reshape(-1).sigmoid(),
            colors=gaussians.sh_coefficients,
            viewmats=cameras.world_to_camera,
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=gaussians.active_sh_degree,
            render_mode="RGB+ED",
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("gsplat rasterization returned an unsupported result")
        rendered, alpha = result[:2]
        if rendered.ndim != 4 or rendered.shape[-1] != 4:
            raise RuntimeError("gsplat RGB+ED output must have shape (cameras, height, width, 4)")
        if alpha.ndim != 4 or alpha.shape[-1] != 1:
            raise RuntimeError("gsplat alpha output must have shape (cameras, height, width, 1)")
        return {
            "color": rendered[..., :3].permute(0, 3, 1, 2).contiguous(),
            "depth": rendered[..., 3:].permute(0, 3, 1, 2).contiguous(),
            "alpha": alpha.permute(0, 3, 1, 2).contiguous(),
        }


__all__ = ["GsplatBackend"]
