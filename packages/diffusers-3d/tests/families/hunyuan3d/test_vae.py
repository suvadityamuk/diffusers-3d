from __future__ import annotations

import pytest
import torch

from diffusers_3d import Hunyuan3DShapeFieldOutput, Hunyuan3DShapeVAE, MeshAsset


def test_tiny_decode_field_chunking_and_save_load(tmp_path):
    torch.manual_seed(0)
    vae = Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config()).eval()
    latents = torch.randn(1, 4, 8)
    decoded = vae.decode(latents).sample
    queries = torch.randn(1, 11, 3)
    with torch.no_grad():
        chunked = vae.evaluate_field(decoded, queries, query_chunk_size=3)
        unchunked = vae.evaluate_field(decoded, queries, query_chunk_size=32)
        dense = vae.decoded_latents_to_field(decoded, resolution=3, query_chunk_size=5)
    torch.testing.assert_close(chunked, unchunked)
    assert dense.field.shape == (1, 4, 4, 4)

    vae.save_pretrained(tmp_path)
    loaded = Hunyuan3DShapeVAE.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded.decode(latents).sample, decoded)


@pytest.mark.portable
def test_tiny_sphere_field_returns_mesh_asset():
    vae = Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config())
    axis = torch.linspace(-1.0, 1.0, 9)
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    field = 0.55 - xyz.square().sum(dim=-1).sqrt()
    output = Hunyuan3DShapeFieldOutput(
        field=field.unsqueeze(0),
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )
    meshes = vae.extract_meshes(output)
    assert len(meshes) == 1
    assert type(meshes[0]) is MeshAsset
    assert meshes[0].vertices.shape[1] == 3
    assert meshes[0].faces.shape[1] == 3
    assert float(meshes[0].vertices.abs().max()) <= 0.75


def test_decoder_state_dict_names_are_reference_compatible():
    vae = Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config())
    keys = set(vae.state_dict())
    assert "post_kl.weight" in keys
    assert "transformer.resblocks.0.attn.c_qkv.weight" in keys
    assert "transformer.resblocks.0.mlp.c_fc.weight" in keys
    assert "geo_decoder.query_proj.weight" in keys
    assert "geo_decoder.cross_attn_decoder.attn.c_kv.weight" in keys
    assert "geo_decoder.output_proj.weight" in keys
    assert not any(key.startswith(("encoder.", "pre_kl.")) for key in keys)


def test_unsupported_encode_flashvdm_and_diso_are_explicit():
    vae = Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config())
    with pytest.raises(NotImplementedError, match="decode-only"):
        vae.encode(torch.zeros(1, 4, 3))
    with pytest.raises(NotImplementedError, match="FlashVDM"):
        vae.enable_flashvdm_decoder()
    with pytest.raises(NotImplementedError, match="FlashVDM"):
        Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config(), decoder_type="flashvdm")
