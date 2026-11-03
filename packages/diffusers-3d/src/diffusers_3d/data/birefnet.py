# Portions of this file reproduce BiRefNet (Swin-L backbone, multi-scale input, ASPP-deformable decoder) and the
# Swin Transformer backbone it embeds:
# https://huggingface.co/ZhengPeng7/BiRefNet (revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4)
# https://github.com/ZhengPeng7/BiRefNet
#
# MIT License. Copyright (c) 2024 ZhengPeng. Swin Transformer: MIT License, Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice
# and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Inference-only rewrite in plain PyTorch: training heads, other backbones, and the timm/kornia/einops imports are
# dropped. Parameter names match the released checkpoint.

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ._hub import require_hub_access

BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
# BRIA's RMBG-2.0 weights for the same network (gated, CC BY-NC 4.0), which TRELLIS.2 and Pixal3D run.
RMBG2_REPO = "briaai/RMBG-2.0"
RMBG2_REVISION = "2fd7b4b0f2393ddb39a267fb62df6c010928a6c6"
# Multi-scale supervision heads that only the training loss reads.
_TRAINING_KEYS = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _window_partition(x: torch.Tensor, size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // size, size, width // size, size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, size, size, channels)


def _window_reverse(windows: torch.Tensor, size: int, height: int, width: int) -> torch.Tensor:
    batch = int(windows.shape[0] / (height * width / size / size))
    x = windows.view(batch, height // size, width // size, size, size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = coords.flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x).reshape(windows, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(tokens, tokens, -1)
        attention = attention + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            attention = attention.view(windows // mask.shape[0], mask.shape[0], self.num_heads, tokens, tokens)
            attention = (attention + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.num_heads, tokens, tokens)
        x = (attention.softmax(dim=-1) @ value).transpose(1, 2).reshape(windows, tokens, channels)
        return self.proj(x)


class _SwinMlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _SwinMlp(dim, 4 * dim)

    def forward(self, x: torch.Tensor, height: int, width: int, mask: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        size = self.window_size
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        pad_right, pad_bottom = (size - width % size) % size, (size - height % size) % size
        x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = x.shape[1:3]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = _window_partition(x, size).view(-1, size * size, channels)
        windows = self.attn(windows, mask if self.shift_size > 0 else None).view(-1, size, size, channels)
        x = _window_reverse(windows, size, padded_height, padded_width)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_right > 0 or pad_bottom > 0:
            x = x[:, :height, :width, :].contiguous()
        x = shortcut + x.view(batch, height * width, channels)
        return x + self.mlp(self.norm2(x))


class _PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)
        if height % 2 == 1 or width % 2 == 1:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(batch, -1, 4 * channels)))


class _SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, downsample: bool) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            _SwinBlock(dim, num_heads, window_size, 0 if index % 2 == 0 else window_size // 2)
            for index in range(depth)
        )
        self.downsample = _PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        size, shift = self.window_size, self.shift_size
        image_mask = torch.zeros((1, -(-height // size) * size, -(-width // size) * size, 1), device=x.device)
        slices = (slice(0, -size), slice(-size, -shift), slice(-shift, None))
        count = 0
        for rows in slices:
            for columns in slices:
                image_mask[:, rows, columns, :] = count
                count += 1
        mask_windows = _window_partition(image_mask, size).view(-1, size * size)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0).to(x.dtype)
        for block in self.blocks:
            x = block(x, height, width, mask)
        if self.downsample is None:
            return x, x, height, width
        return x, self.downsample(x, height, width), (height + 1) // 2, (width + 1) // 2


class _PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        if width % 4 != 0:
            x = F.pad(x, (0, 4 - width % 4))
        if height % 4 != 0:
            x = F.pad(x, (0, 0, 0, 4 - height % 4))
        x = self.proj(x)
        height, width = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).view(-1, x.shape[-1], height, width)


class _SwinBackbone(nn.Module):
    def __init__(self, embed_dim: int, depths: tuple[int, ...], num_heads: tuple[int, ...], window_size: int) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(embed_dim)
        self.num_features = [embed_dim * 2**index for index in range(len(depths))]
        self.layers = nn.ModuleList(
            _SwinStage(self.num_features[index], depth, heads, window_size, index < len(depths) - 1)
            for index, (depth, heads) in enumerate(zip(depths, num_heads))
        )
        for index, features in enumerate(self.num_features):
            self.add_module(f"norm{index}", nn.LayerNorm(features))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        height, width = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        outputs = []
        for index, layer in enumerate(self.layers):
            out, x, next_height, next_width = layer(x, height, width)
            out = getattr(self, f"norm{index}")(out)
            outputs.append(out.view(-1, height, width, self.num_features[index]).permute(0, 3, 1, 2).contiguous())
            height, width = next_height, next_width
        return outputs


class _DeformableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size**2, kernel_size, padding=padding)
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size**2, kernel_size, padding=padding)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        from torchvision.ops import deform_conv2d

        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=(1, 1),
        )


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels: int, planes: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.atrous_conv = _DeformableConv2d(in_channels, planes, kernel_size, padding)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.atrous_conv(x)))


class _ASPPDeformable(nn.Module):
    def __init__(self, channels: int, inter_channels: int = 256) -> None:
        super().__init__()
        self.aspp1 = _ASPPModuleDeformable(channels, inter_channels, 1, padding=0)
        self.aspp_deforms = nn.ModuleList(
            _ASPPModuleDeformable(channels, inter_channels, size, padding=size // 2) for size in (1, 3, 7)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(inter_channels * 5, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.aspp1(x)
        pooled = F.interpolate(self.global_avg_pool(x), size=first.shape[2:], mode="bilinear", align_corners=True)
        y = torch.cat((first, *[module(x) for module in self.aspp_deforms], pooled), dim=1)
        return self.relu(self.bn1(self.conv1(y)))


class _BasicDecBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = _ASPPDeformable(inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dec_att(self.relu_in(self.bn_in(self.conv_in(x))))
        return self.bn_out(self.conv_out(x))


class _BasicLatBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv1(x))


def _image_to_patches_in_order(image: torch.Tensor, reference: torch.Tensor, order: str = "channels") -> torch.Tensor:
    """Stack the ``reference``-sized patches of ``image`` along channels.

    ``"channels"``: ``einops.rearrange(image, "b c (hg h) (wg w) -> b (c hg wg) h w")``, as ``ZhengPeng7/BiRefNet``.
    ``"columns"``: ``b (wg hg c) h w``, the column-major patch split of the remote code ``briaai/RMBG-2.0`` was
    trained with (``get_patches_batch``).
    """

    batch, channels, height, width = image.shape
    rows, columns = height // reference.shape[-2], width // reference.shape[-1]
    patches = image.view(batch, channels, rows, height // rows, columns, width // columns)
    patches = patches.permute(0, 1, 2, 4, 3, 5) if order == "channels" else patches.permute(0, 4, 2, 1, 3, 5)
    return patches.reshape(batch, channels * rows * columns, height // rows, width // columns)


def _gdt_convs(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))


class _Decoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], patch_order: str = "channels") -> None:
        super().__init__()
        self.patch_order = patch_order
        c = channels
        # The image is split into 32x32, 16x16, 8x8, 4x4, and 1x1 patch grids matching the feature resolutions.
        self.ipt_blk5 = _SimpleConvs(2**10 * 3, c[0] // 8)
        self.ipt_blk4 = _SimpleConvs(2**8 * 3, c[0] // 8)
        self.ipt_blk3 = _SimpleConvs(2**6 * 3, c[1] // 8)
        self.ipt_blk2 = _SimpleConvs(2**4 * 3, c[2] // 8)
        self.ipt_blk1 = _SimpleConvs(2**0 * 3, c[3] // 8)
        self.decoder_block4 = _BasicDecBlk(c[0] + c[0] // 8, c[1])
        self.decoder_block3 = _BasicDecBlk(c[1] + c[0] // 8, c[2])
        self.decoder_block2 = _BasicDecBlk(c[2] + c[1] // 8, c[3])
        self.decoder_block1 = _BasicDecBlk(c[3] + c[2] // 8, c[3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(c[3] // 2 + c[3] // 8, 1, 1, 1, 0))
        self.lateral_block4 = _BasicLatBlk(c[1], c[1])
        self.lateral_block3 = _BasicLatBlk(c[2], c[2])
        self.lateral_block2 = _BasicLatBlk(c[3], c[3])
        self.gdt_convs_4 = _gdt_convs(c[1])
        self.gdt_convs_3 = _gdt_convs(c[2])
        self.gdt_convs_2 = _gdt_convs(c[3])
        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(16, 1, 1, 1, 0))

    def forward(self, x, x1, x2, x3, x4) -> torch.Tensor:
        def upsample(tensor, reference):
            return F.interpolate(tensor, size=reference.shape[2:], mode="bilinear", align_corners=True)

        def _image_to_patches(image, reference):
            return _image_to_patches_in_order(image, reference, self.patch_order)

        p4 = self.decoder_block4(torch.cat((x4, self.ipt_blk5(_image_to_patches(x, x4))), 1))
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        p3 = upsample(p4, x3) + self.lateral_block4(x3)
        p3 = self.decoder_block3(torch.cat((p3, self.ipt_blk4(_image_to_patches(x, p3))), 1))
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        p2 = upsample(p3, x2) + self.lateral_block3(x2)
        p2 = self.decoder_block2(torch.cat((p2, self.ipt_blk3(_image_to_patches(x, p2))), 1))
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        p1 = upsample(p2, x1) + self.lateral_block2(x1)
        p1 = self.decoder_block1(torch.cat((p1, self.ipt_blk2(_image_to_patches(x, p1))), 1))
        p1 = upsample(p1, x)
        return self.conv_out1(torch.cat((p1, self.ipt_blk1(_image_to_patches(x, p1))), 1))


class BiRefNet(nn.Module):
    """BiRefNet dichotomous image segmentation; ``forward`` maps normalized images to foreground probabilities.

    The defaults are the released Swin-L configuration. Inputs need sides divisible by 32. ``patch_order`` is how
    the decoder stacks image patches: ``"channels"`` for ``ZhengPeng7/BiRefNet``, ``"columns"`` for
    ``briaai/RMBG-2.0``, whose remote code splits them column by column.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depths: tuple[int, ...] = (2, 2, 18, 2),
        num_heads: tuple[int, ...] = (6, 12, 24, 48),
        window_size: int = 12,
        patch_order: Literal["channels", "columns"] = "channels",
    ) -> None:
        super().__init__()
        self.bb = _SwinBackbone(embed_dim, depths, num_heads, window_size)
        # Features of the full and half-size image are concatenated, doubling the backbone channels.
        channels = tuple(2 * features for features in reversed(self.bb.num_features))
        self.squeeze_module = nn.Sequential(_BasicDecBlk(channels[0] + sum(channels[1:]), channels[0]))
        self.decoder = _Decoder(channels, patch_order)

    @classmethod
    def from_pretrained(cls, repo_id: str = BIREFNET_REPO, revision: str | None = BIREFNET_REVISION) -> BiRefNet:
        """Load released Swin-L weights from ``model.safetensors`` in ``repo_id``, as float32.

        ``ZhengPeng7/BiRefNet`` (MIT) and ``briaai/RMBG-2.0`` (gated; accept its license on the Hub first) share the
        layout; RMBG-2.0 gets its column-major ``patch_order``. Weights load from their own repository and are never converted or redistributed.
        """

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        require_hub_access(
            repo_id, used_for="BiRefNet background removal", alternative="pass your own remover or model"
        )
        state_dict = load_file(hf_hub_download(repo_id, "model.safetensors", revision=revision))
        model = cls(patch_order="columns" if repo_id == RMBG2_REPO else "channels")
        model.load_state_dict({key: value for key, value in state_dict.items() if not key.startswith(_TRAINING_KEYS)})
        return model.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        features = self.bb(x)
        half = self.bb(F.interpolate(x, size=(height // 2, width // 2), mode="bilinear", align_corners=True))
        x1, x2, x3, x4 = (
            torch.cat([full, F.interpolate(low, size=full.shape[2:], mode="bilinear", align_corners=True)], 1)
            for full, low in zip(features, half)
        )
        context = [
            F.interpolate(feature, size=x4.shape[2:], mode="bilinear", align_corners=True) for feature in (x1, x2, x3)
        ]
        x4 = self.squeeze_module(torch.cat((*context, x4), 1))
        return torch.sigmoid(self.decoder(x, x1, x2, x3, x4))


class BiRefNetBackgroundRemover:
    """Cut out the object of an RGB ``PIL`` image with BiRefNet; returns the RGBA image with the predicted alpha.

    ``resize`` picks the upstream wrapper:

    - ``"pil"``: TRELLIS.2, Pixal3D, and the Step1X-3D texture stage. Torchvision ``Resize((1024, 1024))`` on the PIL
      image, ``ToTensor``, ImageNet ``Normalize``; the probability map goes through ``ToPILImage`` (truncated to 8
      bits) and PIL's default resize back to the input size.
    - ``"tensor"``: IDOL and TripoSplat. Bilinear ``align_corners=True`` tensor resizes both ways, then
      ``(alpha.clamp(0, 1) * 255).to(uint8)``.

    The default weights are ``ZhengPeng7/BiRefNet`` (MIT), byte-identical to TripoSplat's bundled
    ``birefnet.safetensors``; IDOL's bundled file holds the same tensors without the training heads. ``repository``
    and ``revision`` pick other released BiRefNet weights (TRELLIS.2 and Pixal3D use :data:`RMBG2_REPO`), or pass a
    loaded ``model``. The model moves to the device given to :meth:`to`, in ``cuda_dtype`` there (IDOL runs
    it in float16 on CUDA) and float32 elsewhere.
    """

    def __init__(
        self,
        model: BiRefNet | None = None,
        *,
        resize: Literal["pil", "tensor"] = "pil",
        cuda_dtype: torch.dtype = torch.float32,
        repository: str = BIREFNET_REPO,
        revision: str | None = BIREFNET_REVISION,
    ) -> None:
        if model is None:
            model = BiRefNet.from_pretrained(repository, revision)
        self.model = model.eval().requires_grad_(False)
        self.resize = resize
        self.cuda_dtype = cuda_dtype

    def to(self, device: torch.device | str) -> BiRefNetBackgroundRemover:
        device = torch.device(device)
        self.model.to(device=device, dtype=self.cuda_dtype if device.type == "cuda" else torch.float32)
        return self

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        parameter = next(self.model.parameters())
        image = image.convert("RGB")
        if self.resize == "pil":
            import torchvision.transforms.functional as TF

            pixels = TF.normalize(TF.to_tensor(TF.resize(image, [1024, 1024])), _MEAN, _STD)
            probability = self.model(pixels[None].to(parameter.device)).cpu()[0].squeeze()
            mask = TF.to_pil_image(probability).resize(image.size)
        else:
            pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
            pixels = F.interpolate(pixels, size=(1024, 1024), mode="bilinear", align_corners=True)
            pixels = (pixels - torch.tensor(_MEAN).view(1, 3, 1, 1)) / torch.tensor(_STD).view(1, 3, 1, 1)
            alpha = self.model(pixels.to(device=parameter.device, dtype=parameter.dtype))
            alpha = F.interpolate(alpha.float(), size=image.size[::-1], mode="bilinear", align_corners=True)[0, 0]
            mask = Image.fromarray((alpha.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
        image.putalpha(mask)
        return image


__all__ = [
    "BIREFNET_REPO",
    "BIREFNET_REVISION",
    "RMBG2_REPO",
    "RMBG2_REVISION",
    "BiRefNet",
    "BiRefNetBackgroundRemover",
]
