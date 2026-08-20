# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the local MiniMax H3 remote-VAE operator injection."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.minimax_h3.vae_ops import (
    _apply_h3_omni_rope,
    patch_minimax_h3_video_vae,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def all_to_all_4D(tensor, *_args, **_kwargs):
    """Single-rank spatial-parallel stand-in exposed like the remote helper."""
    return tensor


def get_parallel_state():
    return {"sp_process_group": None}


def _vit_norm_input(_module, hidden_states):
    return hidden_states.float()


def _apply_h3_rope(tensor: torch.Tensor, rotary_pos_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = (value.to(tensor.dtype) for value in rotary_pos_emb)
    rotary_dim = cos.shape[-1]
    rotated, passed = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, passed), dim=-1)


class _RemoteAttention(nn.Module):
    """Small faithful model of the remote H3 attention surface."""

    def __init__(self, *, qk_affine: bool) -> None:
        super().__init__()
        self.dim_head = 8
        self.heads = 2
        self.to_qkv = nn.Linear(16, 48, bias=False)
        self.to_out = nn.Linear(16, 16, bias=False)
        self.norm_q = nn.RMSNorm(8, eps=1e-5, elementwise_affine=qk_affine)
        self.norm_k = nn.RMSNorm(8, eps=1e-5, elementwise_affine=qk_affine)
        self.spatial_parallel = False

    def perform_attention(self, query, key, value, _pack_info):
        return query + key + value

    def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
        if pack_info is None:
            pack_info = {}
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.to_qkv(hidden_states).view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = self.norm_q(_vit_norm_input(self.norm_q, query)).to(query.dtype)
        key = self.norm_k(_vit_norm_input(self.norm_k, key)).to(key.dtype)
        if rotary_pos_emb is not None:
            query = _apply_h3_rope(query, rotary_pos_emb)
            key = _apply_h3_rope(key, rotary_pos_emb)
        return self.to_out(self.perform_attention(query, key, value, pack_info).reshape(batch_size, seq_len, -1))


class _RemoteBlock(nn.Module):
    def __init__(self, *, qk_affine: bool) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(16, eps=1e-5)
        self.norm2 = nn.RMSNorm(16, eps=1e-5)
        self.layer_norm = nn.LayerNorm(16, eps=1e-5)
        self.attn = _RemoteAttention(qk_affine=qk_affine)


class _RemoteVAE(nn.Module):
    def __init__(self, *, qk_affine: bool) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.transformer_blocks = nn.ModuleList([_RemoteBlock(qk_affine=qk_affine)])


def _h3_rotary_embedding(*, batch: int, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    # H3's complete layout repeats the half rotation frequencies.
    half_cos = torch.randn(batch, seq_len, 1, 3, dtype=dtype)
    half_sin = torch.randn(batch, seq_len, 1, 3, dtype=dtype)
    return torch.cat((half_cos, half_cos), dim=-1), torch.cat((half_sin, half_sin), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_h3_vae_patch_matches_remote_rmsnorm_and_full_dim_rope(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    reference = _RemoteVAE(qk_affine=True).to(dtype=dtype)
    patched = copy.deepcopy(reference)
    patch_minimax_h3_video_vae(patched)

    reference_block = reference.decoder.transformer_blocks[0]
    patched_block = patched.decoder.transformer_blocks[0]
    assert isinstance(patched_block.norm1, RMSNorm)
    assert isinstance(patched_block.norm2, RMSNorm)
    assert isinstance(patched_block.attn.norm_q, RMSNorm)
    assert isinstance(patched_block.attn.norm_k, RMSNorm)
    assert isinstance(patched_block.layer_norm, nn.LayerNorm)
    assert isinstance(patched_block.attn.omni_rope, RotaryEmbedding)

    # Remote H3 always feeds FP32 into decoder RMSNorm and casts back.
    norm_input = torch.randn(2, 5, 16, dtype=torch.float32)
    for reference_norm, patched_norm in (
        (reference_block.norm1, patched_block.norm1),
        (reference_block.norm2, patched_block.norm2),
    ):
        expected_norm = reference_norm(norm_input).to(dtype)
        actual_norm = patched_norm(norm_input).to(dtype)
        torch.testing.assert_close(actual_norm, expected_norm, atol=3e-3, rtol=3e-3)

    hidden_states = norm_input.to(dtype)
    rotary_pos_emb = _h3_rotary_embedding(batch=2, seq_len=5, dtype=dtype)
    expected = reference_block.attn(hidden_states, rotary_pos_emb)
    actual = patched_block.attn(hidden_states, rotary_pos_emb)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)

    # Running a second time keeps the injected RoPE and parameters intact.
    rope = patched_block.attn.omni_rope
    patch_minimax_h3_video_vae(patched)
    assert patched_block.attn.omni_rope is rope


def test_video_vae_enables_local_ops_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class _RemoteWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _RemoteVAE(qk_affine=False)

    monkeypatch.setenv("MINIMAX_H3_VAE_USE_OMNI_OPS", "1")
    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]},
    )
    monkeypatch.setattr(vae_module, "_load_remote_component", lambda _path, _config: _RemoteWrapper())

    video_vae = vae_module.MiniMaxH3VideoVAE("unused", device=torch.device("cpu"))
    assert isinstance(video_vae.model.decoder.transformer_blocks[0].norm1, RMSNorm)


def test_h3_vae_patch_leaves_non_affine_rmsnorm_parameter_contract_unchanged() -> None:
    model = _RemoteVAE(qk_affine=False)
    block = model.decoder.transformer_blocks[0]
    original_norm_q = block.attn.norm_q
    original_norm_k = block.attn.norm_k
    original_state_keys = set(model.state_dict())

    patch_minimax_h3_video_vae(model)

    assert block.attn.norm_q is original_norm_q
    assert block.attn.norm_k is original_norm_k
    assert block.attn.norm_q.weight is None
    assert block.attn.norm_k.weight is None
    assert set(model.state_dict()) == original_state_keys


def test_rotary_embedding_accepts_h3_four_dimensional_full_rotary_layout() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 7, 3, 8)
    cos, sin = _h3_rotary_embedding(batch=2, seq_len=7, dtype=x.dtype)
    rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)

    actual = _apply_h3_omni_rope(rope, x, cos, sin)
    expected = _apply_h3_rope(x, (cos, sin))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("is_npu", [False, True])
def test_h3_rope_only_passes_the_rotary_prefix_to_omni(
    monkeypatch: pytest.MonkeyPatch,
    is_npu: bool,
) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import vae_ops

    class _RecordingRope(nn.Module):
        def forward(self, tensor, cos, sin):
            self.tensor = tensor
            self.cos = cos
            self.sin = sin
            return tensor + 1

    class _Platform:
        @staticmethod
        def is_npu() -> bool:
            return is_npu

    monkeypatch.setattr(vae_ops, "current_omni_platform", _Platform())
    rope = _RecordingRope()
    x = torch.randn(2, 7, 3, 64)
    cos = torch.randn(2, 7, 1, 48)
    sin = torch.randn(2, 7, 1, 48)

    actual = _apply_h3_omni_rope(rope, x, cos, sin)

    assert rope.tensor.shape == (2, 7, 3, 48)
    expected_cos_shape = (2, 7, 1, 48) if is_npu else (2, 7, 48)
    assert rope.cos.shape == expected_cos_shape
    assert rope.sin.shape == expected_cos_shape
    torch.testing.assert_close(actual[..., :48], x[..., :48] + 1, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 48:], x[..., 48:], atol=0, rtol=0)
