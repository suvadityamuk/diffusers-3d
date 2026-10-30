# Portions of this file reproduce the MoGe and MoGe-2 point-map models (``moge/model/v1.py``, ``moge/model/v2.py``,
# ``moge/model/modules``):
# https://github.com/microsoft/MoGe
# Revisions: a8c37341bc0325ca99b9d57981cc3bb2bd3e255b (MoGe) and 74fbce054ebed49800de42d0ad0e83495065719a (MoGe-2)
# MIT License. Copyright (c) Microsoft Corporation.
#
# The DINOv2 backbone is the Transformers ``Dinov2Model``; only its position-embedding interpolation is replaced, to
# follow the torch-hub DINOv2 that MoGe vendors (facebookresearch/dinov2, Apache License 2.0, Copyright (c) Meta
# Platforms, Inc. and affiliates). Rewritten as plain PyTorch modules with renamed submodules.

"""MoGe (v1) and MoGe-2 monocular geometry models: image to affine point map, mask, depth and camera intrinsics."""

from __future__ import annotations

import contextlib
import math
from numbers import Number

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from torch import nn
from transformers import Dinov2Config, Dinov2Model
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings

from ...execution.models import Object3DModel
from .geometry import depth_to_points, focal_to_intrinsics, normalized_view_plane_uv, recover_focal_shift

# ``dinov2_vitl14`` as MoGe's vendored torch-hub loader builds it: no registers, LayerScale 1.0, bicubic
# position-embedding interpolation with the 0.1 offset.
DINOV2_VITL14_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "mlp_ratio": 4,
    "patch_size": 14,
    "image_size": 518,
    "layerscale_value": 1.0,
    "layer_norm_eps": 1e-6,
}
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


class MoGeDinov2Embeddings(Dinov2Embeddings):
    """``Dinov2Embeddings`` interpolating position embeddings like torch-hub DINOv2 (``interpolate_offset=0.1``).

    Transformers resizes to the token grid with ``size=``; torch-hub DINOv2 passes ``scale_factor=(grid + 0.1) / M``,
    which samples the bicubic kernel at different source coordinates. MoGe picks non-square, non-native grids, so
    the two differ on every input it sees.
    """

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_positions = self.position_embeddings.shape[1] - 1
        if embeddings.shape[1] - 1 == num_positions and height == width:
            return self.position_embeddings
        position_embeddings = self.position_embeddings.float()
        grid = int(math.sqrt(num_positions))
        rows, cols = height // self.patch_size, width // self.patch_size
        patch_embeddings = F.interpolate(
            position_embeddings[:, 1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2),
            scale_factor=(float(rows + 0.1) / grid, float(cols + 0.1) / grid),
            mode="bicubic",
            antialias=False,
        )
        patch_embeddings = patch_embeddings.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat([position_embeddings[:, :1], patch_embeddings], dim=1).to(embeddings.dtype)


# Transformers 5.20 renamed DINOv2's attention weights; checkpoints may use either naming.
_DINOV2_ATTENTION_NAMES = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def match_dinov2_attention_names(state_dict: dict, expected_keys) -> None:
    """Rename DINOv2 attention keys in place to the naming of ``expected_keys`` (the installed transformers)."""

    renamed = any(".attention.q_proj." in key for key in expected_keys)
    for key in list(state_dict):
        target = key
        for legacy, current in _DINOV2_ATTENTION_NAMES:
            target = target.replace(legacy, current) if renamed else target.replace(current, legacy)
        if target != key:
            state_dict[target] = state_dict.pop(key)


class _Dinov2KeyNamesMixin:
    def _fix_state_dict_keys_on_load(self, state_dict) -> None:
        super()._fix_state_dict_keys_on_load(state_dict)
        match_dinov2_attention_names(state_dict, self.state_dict().keys())


def _dinov2_backbone(config: dict) -> Dinov2Model:
    backbone = Dinov2Model(Dinov2Config(**config))
    backbone.embeddings = MoGeDinov2Embeddings(backbone.config)
    return backbone


def _intermediate_layers(backbone: Dinov2Model, pixel_values: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
    """Final-layer-normed tokens (class token first) after each block in ``layers``, as ``get_intermediate_layers``."""

    # The stock layers in order, as ``Dinov2Encoder`` runs them; ``output_hidden_states`` would capture through hooks
    # that ``torch.compile(fullgraph=True)`` cannot trace.
    hidden_states, outputs = backbone.embeddings(pixel_values), []
    for index, layer in enumerate(backbone.encoder.layer[: max(layers) + 1]):
        hidden_states = layer(hidden_states)
        if index in layers:
            outputs.append(backbone.layernorm(hidden_states))
    return outputs


class MoGeResidualConvBlock(nn.Module):
    """Pre-activation residual block: [norm] ReLU conv3x3 [norm] ReLU conv3x3, replicate padding, identity skip."""

    def __init__(
        self, channels: int, hidden_channels: int, in_norm_groups: int | None, hidden_norm_groups: int | None
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(in_norm_groups, channels) if in_norm_groups else nn.Identity()
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(hidden_norm_groups, hidden_channels) if hidden_norm_groups else nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, channels, 3, padding=1, padding_mode="replicate")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.relu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.relu(self.norm2(hidden_states)))
        return hidden_states + residual


class MoGeUpsampler(nn.Module):
    """2x upsampling (stride-2 transposed conv, or bilinear) followed by a 3x3 replicate-padded conv."""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "conv_transpose"):
        super().__init__()
        if mode not in ("conv_transpose", "bilinear"):
            raise ValueError(f"unsupported upsampler {mode!r}")
        self.conv_transpose = (
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2) if mode == "conv_transpose" else None
        )
        self.conv = nn.Conv2d(
            out_channels if self.conv_transpose is not None else in_channels,
            out_channels,
            3,
            padding=1,
            padding_mode="replicate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.conv_transpose is not None:
            hidden_states = self.conv_transpose(hidden_states)
        else:
            hidden_states = F.interpolate(hidden_states, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(hidden_states)


def _with_uv(hidden_states: torch.Tensor, aspect_ratio: float, dtype: torch.dtype) -> torch.Tensor:
    """Concatenate the view-plane UV channels that make the heads aware of the image aspect ratio."""

    batch_size, _, height, width = hidden_states.shape
    uv = normalized_view_plane_uv(width, height, aspect_ratio=aspect_ratio, dtype=dtype, device=hidden_states.device)
    uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    return torch.cat([hidden_states, uv], dim=1)


def _remap_points(points: torch.Tensor, remap_output: str) -> torch.Tensor:
    if remap_output == "linear":
        return points
    if remap_output == "sinh":
        return torch.sinh(points)
    xy, z = points.split([2, 1], dim=-1)
    if remap_output == "exp":
        z = torch.exp(z)
        return torch.cat([xy * z, z], dim=-1)
    if remap_output == "sinh_exp":
        return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
    raise ValueError(f"invalid remap_output {remap_output!r}")


def _num_tokens(num_tokens: int | None, resolution_level: int, num_tokens_range: list[int]) -> int:
    if num_tokens is not None:
        return num_tokens
    min_tokens, max_tokens = num_tokens_range
    return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))


def _focal_and_shift(
    points: torch.Tensor, mask: torch.Tensor, fov_x: Number | torch.Tensor | None, aspect_ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if fov_x is None:
        return recover_focal_shift(points, mask)
    fov_x = torch.as_tensor(fov_x, device=points.device, dtype=points.dtype)
    focal = aspect_ratio / (1 + aspect_ratio**2) ** 0.5 / torch.tan(torch.deg2rad(fov_x / 2))
    if focal.ndim == 0:
        focal = focal[None].expand(points.shape[0])
    return recover_focal_shift(points, mask, focal=focal)


class MoGeModel(_Dinov2KeyNamesMixin, Object3DModel):
    """MoGe (v1): DINOv2 ViT-L/14 and a convolutional head predicting an affine-invariant point map and a mask.

    Matches ``moge.model.v1.MoGeModel`` at MoGe revision ``a8c3734`` with ``Ruicheng/moge-vitl``, the version SAM 3D
    Objects pins. Use :meth:`infer` on RGB images in ``[0, 1]``.
    """

    @register_to_config
    def __init__(
        self,
        backbone_config: dict | None = None,
        intermediate_layers: int | list[int] = 4,
        dim_proj: int = 512,
        dim_upsample: list[int] = (256, 128, 64),
        dim_times_res_block_hidden: int = 2,
        num_res_blocks: int = 2,
        last_conv_channels: int = 32,
        remap_output: str = "exp",
        num_tokens_range: list[int] = (1275, 2551),
        mask_threshold: float = 0.5,
    ):
        super().__init__()
        self.backbone = _dinov2_backbone({**DINOV2_VITL14_CONFIG, **(backbone_config or {})})
        hidden_size = self.backbone.config.hidden_size
        num_layers = self.backbone.config.num_hidden_layers
        self.feature_layers = (
            list(range(num_layers - intermediate_layers, num_layers))
            if isinstance(intermediate_layers, int)
            else list(intermediate_layers)
        )
        dim_upsample = list(dim_upsample)
        self.projections = nn.ModuleList(nn.Conv2d(hidden_size, dim_proj, 1) for _ in self.feature_layers)
        self.upsamplers = nn.ModuleList(
            MoGeUpsampler(in_channels + 2, out_channels)
            for in_channels, out_channels in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(
                *(
                    MoGeResidualConvBlock(
                        channels, dim_times_res_block_hidden * channels, 1, dim_times_res_block_hidden * channels // 32
                    )
                    for _ in range(num_res_blocks)
                )
            )
            for channels in dim_upsample
        )
        # Point (3 channels) and mask (1 channel) output heads.
        self.output_convs_in = nn.ModuleList(
            nn.Conv2d(dim_upsample[-1] + 2, last_conv_channels, 3, padding=1, padding_mode="replicate")
            for _ in range(2)
        )
        self.output_convs_out = nn.ModuleList(nn.Conv2d(last_conv_channels, channels, 1) for channels in (3, 1))
        self.register_buffer("image_mean", torch.tensor(_IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(_IMAGE_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor, num_tokens: int) -> dict[str, torch.Tensor]:
        """Affine point map ``(B, H, W, 3)`` and mask logits ``(B, H, W)`` for ``image`` ``(B, 3, H, W)`` in [0, 1]."""

        original_height, original_width = image.shape[-2:]
        patch_size = self.backbone.config.patch_size
        resize_factor = ((num_tokens * patch_size**2) / (original_height * original_width)) ** 0.5
        width, height = int(original_width * resize_factor), int(original_height * resize_factor)
        image = F.interpolate(image, (height, width), mode="bicubic", align_corners=False, antialias=True)
        image = (image - self.image_mean) / self.image_std
        rows, cols = height // patch_size, width // patch_size
        pixel_values = F.interpolate(
            image, (rows * patch_size, cols * patch_size), mode="bilinear", align_corners=False, antialias=True
        )

        tokens = _intermediate_layers(self.backbone, pixel_values, self.feature_layers)
        hidden_states = torch.stack(
            [
                project(layer_tokens[:, 1:].permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous())
                for project, layer_tokens in zip(self.projections, tokens)
            ],
            dim=1,
        ).sum(dim=1)

        aspect_ratio = width / height
        for upsampler, res_blocks in zip(self.upsamplers, self.res_blocks):
            hidden_states = upsampler(_with_uv(hidden_states, aspect_ratio, hidden_states.dtype))
            hidden_states = res_blocks(hidden_states)
        hidden_states = F.interpolate(hidden_states, (height, width), mode="bilinear", align_corners=False)
        hidden_states = _with_uv(hidden_states, aspect_ratio, hidden_states.dtype)
        points, mask = (
            conv_out(F.relu(conv_in(hidden_states)))
            for conv_in, conv_out in zip(self.output_convs_in, self.output_convs_out)
        )

        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            points = F.interpolate(points, (original_height, original_width), mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, (original_height, original_width), mode="bilinear", align_corners=False)
            points = _remap_points(points.permute(0, 2, 3, 1), self.config.remap_output)
        return {"points": points, "mask": mask.squeeze(1)}

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        fov_x: Number | torch.Tensor | None = None,
        resolution_level: int = 9,
        num_tokens: int | None = None,
        apply_mask: bool = True,
        force_projection: bool = True,
        use_fp16: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Camera-space point map, depth, intrinsics and mask for ``image`` ``(3, H, W)`` or ``(B, 3, H, W)`` in [0, 1].

        Args:
            fov_x: horizontal field of view in degrees; estimated from the point map when None.
            resolution_level: 0-9, picks ``num_tokens`` in ``num_tokens_range``; the output keeps the input size.
            num_tokens: backbone token budget, overriding ``resolution_level``.
            apply_mask: set points and depth outside the mask to ``inf``.
            force_projection: recompute the points by unprojecting the depth through the intrinsics; otherwise only
                shift the predicted points along z.
            use_fp16: run the network under float16 autocast on the image's device (upstream's default).

        Returns:
            ``points`` (..., H, W, 3) in OpenCV camera space (x right, y down, z forward), ``depth`` (..., H, W),
            ``intrinsics`` (..., 3, 3) normalized by the image width and height, ``mask`` (..., H, W) bool, and
            ``mask_prob`` (..., H, W).
        """

        omit_batch_dim = image.dim() == 3
        if omit_batch_dim:
            image = image.unsqueeze(0)
        height, width = image.shape[-2:]
        aspect_ratio = width / height
        num_tokens = _num_tokens(num_tokens, resolution_level, self.config.num_tokens_range)

        with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=use_fp16):
            output = self.forward(image, num_tokens)
        points, mask = output["points"], output["mask"]
        mask_binary = mask > self.config.mask_threshold

        focal, shift = _focal_and_shift(points, mask_binary, fov_x, aspect_ratio)
        intrinsics = focal_to_intrinsics(focal, aspect_ratio)
        depth = points[..., 2] + shift[..., None, None]
        if force_projection:
            points = depth_to_points(depth, intrinsics)
        else:
            points = (
                points
                + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]
            )
        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        result = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "mask_prob": torch.sigmoid(mask),
        }
        return {key: value.squeeze(0) for key, value in result.items()} if omit_batch_dim else result


class MoGeConvStack(nn.Module):
    """MoGe-2 ``ConvStack``: a coarse-to-fine pyramid of 1x1 inputs, residual blocks and 2x upsamplers."""

    def __init__(
        self,
        dim_in: list[int],
        dim_res_blocks: list[int],
        num_res_blocks: list[int],
        resamplers: list[str],
        dim_out: int | None = None,
        fp32_output_projection: bool = False,
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList(nn.Conv2d(cin, cout, 1) for cin, cout in zip(dim_in, dim_res_blocks))
        self.resamplers = nn.ModuleList(
            MoGeUpsampler(cin, cout, mode)
            for cin, cout, mode in zip(dim_res_blocks[:-1], dim_res_blocks[1:], resamplers)
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(*(MoGeResidualConvBlock(channels, channels, None, None) for _ in range(count)))
            for channels, count in zip(dim_res_blocks, num_res_blocks)
        )
        self.output_projection = nn.Conv2d(dim_res_blocks[-1], dim_out, 1) if dim_out is not None else None
        self.fp32_output_projection = fp32_output_projection

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor] | torch.Tensor:
        """Per-level features (no output projection) or the finest level's projection."""

        outputs, hidden_states = [], None
        for level, (input_block, res_blocks) in enumerate(zip(self.input_blocks, self.res_blocks)):
            feature = input_block(features[level])
            hidden_states = feature if hidden_states is None else hidden_states + feature
            hidden_states = res_blocks(hidden_states)
            outputs.append(hidden_states)
            if level < len(self.resamplers):
                hidden_states = self.resamplers[level](hidden_states)
        if self.output_projection is None:
            return outputs
        context = (
            torch.autocast(device_type=outputs[-1].device.type, dtype=torch.float32)
            if self.fp32_output_projection
            else contextlib.nullcontext()
        )
        with context:
            return self.output_projection(outputs[-1])


class MoGe2Model(_Dinov2KeyNamesMixin, Object3DModel):
    """MoGe-2: DINOv2 ViT-L/14, a shared conv neck, point and mask heads, and a metric-scale MLP.

    Matches ``moge.model.v2.MoGeModel`` at MoGe revision ``74fbce05`` with ``Ruicheng/moge-2-vitl``, the version Pixal3D
    installs. Use :meth:`infer` on RGB images in ``[0, 1]``.
    """

    @register_to_config
    def __init__(
        self,
        backbone_config: dict | None = None,
        intermediate_layers: list[int] = (5, 11, 17, 23),
        dim_features: int = 1024,
        dim_res_blocks: list[int] = (1024, 256, 128, 64, 32),
        neck_num_res_blocks: list[int] = (0, 2, 2, 2, 0),
        head_num_res_blocks: list[int] = (0, 1, 1, 1, 0),
        resamplers: list[str] = ("conv_transpose", "conv_transpose", "conv_transpose", "bilinear"),
        scale_head_dims: list[int] = (1024, 1024, 1024, 1),
        remap_output: str = "exp",
        num_tokens_range: list[int] = (1200, 3600),
    ):
        super().__init__()
        self.backbone = _dinov2_backbone({**DINOV2_VITL14_CONFIG, **(backbone_config or {})})
        hidden_size = self.backbone.config.hidden_size
        dim_res_blocks, resamplers = list(dim_res_blocks), list(resamplers)
        self.output_projections = nn.ModuleList(nn.Conv2d(hidden_size, dim_features, 1) for _ in intermediate_layers)
        self.neck = MoGeConvStack(
            [dim_features + 2] + [2] * (len(dim_res_blocks) - 1), dim_res_blocks, list(neck_num_res_blocks), resamplers
        )
        head = {"dim_in": dim_res_blocks, "dim_res_blocks": dim_res_blocks, "resamplers": resamplers}
        self.points_head = MoGeConvStack(
            **head, num_res_blocks=list(head_num_res_blocks), dim_out=3, fp32_output_projection=True
        )
        self.mask_head = MoGeConvStack(**head, num_res_blocks=list(head_num_res_blocks), dim_out=1)
        self.scale_head = nn.ModuleList(
            nn.Linear(cin, cout) for cin, cout in zip(scale_head_dims[:-1], scale_head_dims[1:])
        )
        self.register_buffer("image_mean", torch.tensor(_IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(_IMAGE_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor, num_tokens: int) -> dict[str, torch.Tensor]:
        """Affine point map ``(B, H, W, 3)``, mask probability ``(B, H, W)`` and metric scale ``(B,)``."""

        batch_size, _, height, width = image.shape
        aspect_ratio = width / height
        rows, cols = round((num_tokens / aspect_ratio) ** 0.5), round((num_tokens * aspect_ratio) ** 0.5)
        patch_size = self.backbone.config.patch_size
        pixel_values = F.interpolate(
            image, (rows * patch_size, cols * patch_size), mode="bilinear", align_corners=False, antialias=True
        )
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        tokens = _intermediate_layers(self.backbone, pixel_values, list(self.config.intermediate_layers))
        hidden_states = torch.stack(
            [
                project(layer_tokens[:, 1:].permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous())
                for project, layer_tokens in zip(self.output_projections, tokens)
            ],
            dim=1,
        ).sum(dim=1)
        class_token = tokens[-1][:, 0]

        features = [hidden_states] + [None] * (len(self.neck.input_blocks) - 1)
        for level in range(len(features)):
            uv = normalized_view_plane_uv(
                cols * 2**level, rows * 2**level, aspect_ratio=aspect_ratio, dtype=image.dtype, device=image.device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            features[level] = uv if features[level] is None else torch.cat([features[level], uv], dim=1)
        features = self.neck(features)
        points, mask = self.points_head(features), self.mask_head(features)

        metric_scale = class_token
        for index, linear in enumerate(self.scale_head):
            metric_scale = linear(metric_scale)
            if index < len(self.scale_head) - 1:
                metric_scale = F.relu(metric_scale)

        points = F.interpolate(points, (height, width), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False)
        points = _remap_points(points.permute(0, 2, 3, 1), self.config.remap_output)
        return {"points": points, "mask": mask.squeeze(1).sigmoid(), "metric_scale": metric_scale.squeeze(1).exp()}

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int | None = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Number | torch.Tensor | None = None,
        use_fp16: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Metric camera-space point map, depth, intrinsics and mask for ``image`` ``(3, H, W)`` or ``(B, 3, H, W)``.

        Arguments as :meth:`MoGeModel.infer`. Points and depth are multiplied by the predicted metric scale (meters);
        intrinsics are normalized by the image width and height. Returns ``points``, ``intrinsics``, ``depth`` and
        ``mask``.
        """

        omit_batch_dim = image.dim() == 3
        if omit_batch_dim:
            image = image.unsqueeze(0)
        image = image.to(dtype=self.dtype, device=self.device)
        height, width = image.shape[-2:]
        aspect_ratio = width / height
        num_tokens = _num_tokens(num_tokens, resolution_level, self.config.num_tokens_range)

        enabled = use_fp16 and self.dtype != torch.float16
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=enabled):
            output = self.forward(image, num_tokens=num_tokens)
        points, mask, metric_scale = (output[key].float() for key in ("points", "mask", "metric_scale"))
        if isinstance(fov_x, torch.Tensor):
            fov_x = fov_x.float()

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            mask_binary = mask > 0.5
            focal, shift = _focal_and_shift(points, mask_binary, fov_x, aspect_ratio)
            intrinsics = focal_to_intrinsics(focal, aspect_ratio)
            points[..., 2] += shift[..., None, None]
            mask_binary &= points[..., 2] > 0
            depth = points[..., 2].clone()
            if force_projection:
                points = depth_to_points(depth, intrinsics, homogeneous=True)
            points *= metric_scale[:, None, None, None]
            depth *= metric_scale[:, None, None]
            if apply_mask:
                points = torch.where(mask_binary[..., None], points, torch.inf)
                depth = torch.where(mask_binary, depth, torch.inf)

        result = {"points": points, "intrinsics": intrinsics, "depth": depth, "mask": mask_binary}
        return {key: value.squeeze(0) for key, value in result.items()} if omit_batch_dim else result


__all__ = [
    "DINOV2_VITL14_CONFIG",
    "MoGe2Model",
    "MoGeConvStack",
    "MoGeDinov2Embeddings",
    "MoGeModel",
    "MoGeResidualConvBlock",
    "MoGeUpsampler",
]
