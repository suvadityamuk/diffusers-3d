from __future__ import annotations

import pytest
import torch
from _configs import production_config, tiny_config

from diffusers_3d.families.hunyuan3d_omni.conversion import convert_control_encoder_state_dict
from diffusers_3d.families.hunyuan3d_omni.models import Hunyuan3DOmniControlEncoder

CONTROLS = {
    "skeleton": torch.linspace(-0.5, 0.5, 2 * 4 * 6).reshape(2, 4, 6),
    "bbox": torch.tensor([[[0.8, 0.6, 1.0]], [[0.3, 1.0, 0.5]]]),
    "voxels": torch.linspace(-1, 1, 2 * 9 * 3).reshape(2, 9, 3),
    "point_cloud": torch.linspace(-0.9, 0.9, 2 * 5 * 3).reshape(2, 5, 3),
}


def _encoder() -> Hunyuan3DOmniControlEncoder:
    torch.manual_seed(0)
    return Hunyuan3DOmniControlEncoder(**tiny_config(Hunyuan3DOmniControlEncoder)).eval()


@pytest.mark.parametrize("control_type", list(CONTROLS))
def test_each_control_gives_row_tokens_then_repeated_type_tokens(control_type):
    encoder = _encoder()
    control = CONTROLS[control_type]
    with torch.no_grad():
        tokens = encoder(control, control_type).embeddings
        others = [encoder(CONTROLS[other], other).embeddings[0, -1] for other in CONTROLS if other != control_type]
    rows = encoder.voxelize(control).shape[1] if control_type == "voxels" else control.shape[1]
    assert tokens.shape == (2, rows + 10, 24)
    assert torch.equal(tokens[:, -10:], tokens[:1, -1:].expand(2, 10, -1))
    assert all(not torch.equal(tokens[0, -1], other) for other in others)


def test_three_channel_controls_embed_their_points_twice():
    encoder = _encoder()
    points = CONTROLS["point_cloud"]
    with torch.no_grad():
        as_points = encoder(points, "point_cloud").embeddings[:, :-10]
        as_bones = encoder(points.repeat(1, 1, 2), "skeleton").embeddings[:, :-10]
    torch.testing.assert_close(as_points, as_bones)


def test_voxelize_keeps_occupied_cell_centers_in_linear_order_and_zero_pads():
    encoder = _encoder()  # 4 cells per axis of width 0.5
    points = torch.tensor(
        [[[0.9, 0.9, 0.9], [-1.0, -1.0, -1.0], [0.1, -0.6, -1.0], [-0.99, -0.8, -0.9]], [[1.0, 1.0, -1.0]] * 4]
    )
    centers = encoder.voxelize(points)
    expected = torch.tensor(
        [
            [[-0.75, -0.75, -0.75], [0.25, -0.75, -0.75], [0.75, 0.75, 0.75]],
            [[0.75, 0.75, -0.75], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(centers, expected)


def test_rejects_unknown_types_and_wrong_channels():
    encoder = _encoder()
    with pytest.raises(ValueError, match="control_type"):
        encoder(CONTROLS["point_cloud"], "pose")
    with pytest.raises(ValueError, match="skeleton control must have shape"):
        encoder(CONTROLS["point_cloud"], "skeleton")


def test_save_load_round_trip(tmp_path):
    encoder = _encoder()
    encoder.save_pretrained(tmp_path)
    loaded = Hunyuan3DOmniControlEncoder.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(CONTROLS["voxels"], "voxels").embeddings, encoder(CONTROLS["voxels"], "voxels").embeddings
        )


def test_production_layout_matches_the_released_checkpoint():
    # Keys and shapes of the published cond_encoder/pytorch_model.bin.
    released = {
        "cond_signal_embedding.weight": (4, 8),
        "cond_signal_linear.weight": (1024, 8),
        "cond_signal_linear.bias": (1024,),
        "linear.0.weight": (1024, 102),
        "linear.0.bias": (1024,),
        "linear.1.weight": (1024,),
    }
    with torch.device("meta"):
        encoder = Hunyuan3DOmniControlEncoder(**production_config(Hunyuan3DOmniControlEncoder))
        state_dict = convert_control_encoder_state_dict({key: torch.empty(shape) for key, shape in released.items()})
    encoder.load_state_dict(state_dict, strict=True, assign=True)
    assert {key: tuple(value.shape) for key, value in encoder.state_dict().items()} == {
        key: tuple(value.shape) for key, value in state_dict.items()
    }
