# Portions of this file reproduce BRIA RMBG-1.4 (an IS-Net style U^2-Net) as TripoSG vendors it, and TripoSG's
# post-processing of its prediction:
# https://huggingface.co/briaai/RMBG-1.4 (revision 2ceba5a5efaec153162aedea169f76caf9b46cf8)
# https://github.com/VAST-AI-Research/TripoSG (revision fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c),
# scripts/briarmbg.py and scripts/image_process.py
#
# The network code is BRIA's; TripoSG's code is MIT-licensed, Copyright (c) 2025 VAST-AI-Research. The RMBG-1.4
# weights are under BRIA's own non-commercial license and are never converted or redistributed: they load from
# briaai/RMBG-1.4.
#
# Inference-only rewrite in plain PyTorch: one module per residual U-block height, and only the first side output,
# the one TripoSG reads, is computed. Parameter names match the released checkpoint.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ._hub import require_hub_access

RMBG14_REPO = "briaai/RMBG-1.4"
RMBG14_REVISION = "2ceba5a5efaec153162aedea169f76caf9b46cf8"


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn_s1 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn_s1(self.conv_s1(x)))


def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(source, size=target.shape[2:], mode="bilinear")


class _RSU(nn.Module):
    """Residual U-block of ``height`` levels (upstream ``RSU7`` ... ``RSU4``); ``dilated`` is ``RSU4F``."""

    def __init__(self, height: int, in_channels: int, mid_channels: int, out_channels: int, dilated: bool = False):
        super().__init__()
        self.height, self.dilated = height, dilated
        self.rebnconvin = _ConvBNReLU(in_channels, out_channels)
        for level in range(1, height + 1):
            dilation = 2 ** (level - 1) if dilated else (2 if level == height else 1)
            channels = out_channels if level == 1 else mid_channels
            setattr(self, f"rebnconv{level}", _ConvBNReLU(channels, mid_channels, dilation))
        for level in range(1, height):
            dilation = 2 ** (level - 1) if dilated else 1
            channels = out_channels if level == 1 else mid_channels
            setattr(self, f"rebnconv{level}d", _ConvBNReLU(2 * mid_channels, channels, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        features, hx = [], hxin
        for level in range(1, self.height + 1):
            hx = getattr(self, f"rebnconv{level}")(hx)
            features.append(hx)
            if not self.dilated and level < self.height - 1:
                hx = F.max_pool2d(hx, 2, stride=2, ceil_mode=True)
        hx = getattr(self, f"rebnconv{self.height - 1}d")(torch.cat((features[-1], features[-2]), 1))
        for level in range(self.height - 2, 0, -1):
            skip = features[level - 1]
            hx = hx if self.dilated else _upsample_like(hx, skip)
            hx = getattr(self, f"rebnconv{level}d")(torch.cat((hx, skip), 1))
        return hx + hxin


class BriaRMBG(nn.Module):
    """BRIA RMBG-1.4: maps ``image - 0.5`` (RGB in ``[0, 1]``) to the foreground probability of its first side output."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.stage1 = _RSU(7, 64, 32, 64)
        self.stage2 = _RSU(6, 64, 32, 128)
        self.stage3 = _RSU(5, 128, 64, 256)
        self.stage4 = _RSU(4, 256, 128, 512)
        self.stage5 = _RSU(4, 512, 256, 512, dilated=True)
        self.stage6 = _RSU(4, 512, 256, 512, dilated=True)
        self.stage5d = _RSU(4, 1024, 256, 512, dilated=True)
        self.stage4d = _RSU(4, 1024, 128, 256)
        self.stage3d = _RSU(5, 512, 64, 128)
        self.stage2d = _RSU(6, 256, 32, 64)
        self.stage1d = _RSU(7, 128, 16, 64)
        # Only ``side1`` feeds the prediction; the others are training-time deep supervision kept for strict loading.
        for index, channels in enumerate((64, 64, 128, 256, 512, 512), 1):
            setattr(self, f"side{index}", nn.Conv2d(channels, out_channels, 3, padding=1))

    @classmethod
    def from_pretrained(cls, repo_id: str = RMBG14_REPO, revision: str | None = RMBG14_REVISION) -> BriaRMBG:
        """Load the released weights from ``model.safetensors`` in ``repo_id`` (accept its license on the Hub first)."""

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        require_hub_access(repo_id, used_for="RMBG-1.4 background removal", alternative="pass your own remover")
        model = cls()
        model.load_state_dict(load_file(hf_hub_download(repo_id, "model.safetensors", revision=revision)))
        return model.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = lambda value: F.max_pool2d(value, 2, stride=2, ceil_mode=True)  # noqa: E731
        hx1 = self.stage1(self.conv_in(x))
        hx2 = self.stage2(pool(hx1))
        hx3 = self.stage3(pool(hx2))
        hx4 = self.stage4(pool(hx3))
        hx5 = self.stage5(pool(hx4))
        hx6 = self.stage6(pool(hx5))
        hx = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), 1))
        hx = self.stage4d(torch.cat((_upsample_like(hx, hx4), hx4), 1))
        hx = self.stage3d(torch.cat((_upsample_like(hx, hx3), hx3), 1))
        hx = self.stage2d(torch.cat((_upsample_like(hx, hx2), hx2), 1))
        hx = self.stage1d(torch.cat((_upsample_like(hx, hx1), hx1), 1))
        return torch.sigmoid(_upsample_like(self.side1(hx), x))


class RMBG14BackgroundRemover:
    """TripoSG's background removal: RMBG-1.4, then a cleaned binary mask; returns the RGBA image.

    As upstream ``load_image``: the RGB image is resized to 1024x1024 (bilinear, antialiased), shifted by ``-0.5``,
    and segmented; the prediction is resized back, min-max normalised, truncated to 8 bits, binarised with Otsu's
    threshold, and connected components under 200 pixels are dropped. Upstream then crops to the bounding box of the
    largest external contour, so the alpha outside that box is cleared here: cropping to the alpha's bounds crops to
    the same box. The weights load from ``briaai/RMBG-1.4`` (non-commercial license) unless ``model`` is given.
    """

    def __init__(self, model: BriaRMBG | None = None) -> None:
        self.model = (BriaRMBG.from_pretrained() if model is None else model).eval().requires_grad_(False)

    def to(self, device: torch.device | str) -> RMBG14BackgroundRemover:
        self.model.to(device=device, dtype=torch.float32)
        return self

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        import cv2
        import torchvision.transforms.functional as TF
        from skimage.measure import label

        image = image.convert("RGB")
        device = next(self.model.parameters()).device
        rgb = torch.from_numpy(np.asarray(image)).to(device).float().permute(2, 0, 1) / 255.0
        resized = TF.resize(rgb, [1024, 1024], antialias=True)
        if resized.flatten().max() < 1e-3:
            raise ValueError("background removal needs an image that is not pure black")
        probability = self.model(TF.normalize(resized, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).unsqueeze(0))[0]
        probability = TF.resize(probability, [rgb.shape[1], rgb.shape[2]], antialias=True)
        high, low = probability.max(), probability.min()
        alpha = ((probability - low) / (high - low) * 255).to(torch.uint8).squeeze().cpu().numpy()
        _, alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ``remove_small_objects(label(alpha), min_size=200)``: drop labelled components of fewer than 200 pixels.
        labels = label(alpha)
        alpha = ((np.bincount(labels.ravel())[labels] >= 200) & (labels > 0)).astype(np.uint8) * 255
        _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("background removal found no object in the image")
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = np.zeros_like(alpha)
        cropped[y : y + height, x : x + width] = alpha[y : y + height, x : x + width]
        image.putalpha(Image.fromarray(cropped, "L"))
        return image


__all__ = ["RMBG14_REPO", "RMBG14_REVISION", "BriaRMBG", "RMBG14BackgroundRemover"]
