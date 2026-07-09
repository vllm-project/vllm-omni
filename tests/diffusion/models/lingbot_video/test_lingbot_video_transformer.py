# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_transformer(**overrides):
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    config = {
        "patch_size": (1, 1, 1),
        "in_channels": 2,
        "out_channels": 2,
        "hidden_size": 16,
        "num_attention_heads": 1,
        "depth": 0,
        "intermediate_size": 32,
        "text_dim": 8,
        "freq_dim": 8,
        "axes_dims": (4, 4, 8),
        "axes_lens": (32, 32, 32),
    }
    config.update(overrides)
    return LingBotVideoTransformer3DModel(**config)


def test_joint_position_ids_video_then_text_order():
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import make_joint_position_ids

    positions = make_joint_position_ids(text_len=3, grid_t=1, grid_h=2, grid_w=2, device=torch.device("cpu"))

    assert positions.shape == (7, 3)
    assert positions[:4, 0].tolist() == [4, 4, 4, 4]
    assert positions[:4, 1:].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert positions[4:].tolist() == [[1, 0, 0], [2, 0, 0], [3, 0, 0]]


def test_tiny_transformer_depth_zero_forward_shape():
    model = _tiny_transformer()
    hidden_states = torch.randn(1, 2, 1, 2, 2)
    timestep = torch.tensor([300.0])
    encoder_hidden_states = torch.randn(1, 3, 8)
    encoder_attention_mask = torch.ones(1, 3, dtype=torch.long)

    with torch.no_grad():
        out = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def test_packed_attention_uses_sdpa_fallback_without_flash_varlen(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    monkeypatch.setattr(module, "flash_attn_varlen_func_v3", None)
    attn = module.LingBotVideoAttention(
        hidden_size=8,
        num_heads=2,
        norm_eps=1e-6,
        qkv_bias=False,
        out_bias=False,
    )
    captured = {}

    def fake_sdpa_forward(query, key, value, attn_metadata):
        captured["mask"] = attn_metadata.attn_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(attn.attn.sdpa_fallback, "forward", fake_sdpa_forward)
    x = torch.randn(1, 5, 8)
    rotary = torch.ones(1, 5, 2, dtype=torch.complex64)
    packed_indices = {
        "cu_seqlens_kv": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_seqlen_in_batch_kv": 3,
        "attention_mask": module._packed_block_attention_mask([2, 3], x.device),
    }

    out = attn(x, rotary, packed_indices=packed_indices)

    assert out.shape == x.shape
    mask = captured["mask"]
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, :2, :2].all()
    assert mask[0, 0, 2:, 2:].all()
    assert not mask[0, 0, :2, 2:].any()
    assert not mask[0, 0, 2:, :2].any()


def test_tiny_transformer_rejects_invalid_rope_dims():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    with pytest.raises(AssertionError, match="head_dim"):
        LingBotVideoTransformer3DModel(
            hidden_size=16,
            num_attention_heads=1,
            axes_dims=(4, 4, 4),
            depth=0,
        )


def test_transformer_to_keeps_sensitive_modules_in_fp32():
    model = _tiny_transformer()

    model.to(dtype=torch.bfloat16)

    assert model.patch_embedder.weight.dtype == torch.bfloat16
    assert model.time_embedder.linear_1.weight.dtype == torch.float32
    assert model.norm_out_modulation[1].weight.dtype == torch.float32
