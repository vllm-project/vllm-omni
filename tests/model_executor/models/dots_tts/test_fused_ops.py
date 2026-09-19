# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Parity and loading regressions for dots.tts fused operator changes."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

from vllm_omni.model_executor.models.dots_tts.dots_tts_dit import (
    DiTBlock,
    FinalLayer,
    _qkv_qk_norm_rope,
    modulate,
)
from vllm_omni.model_executor.models.dots_tts.dots_tts_talker import (
    _load_stacked_qkv_weights,
)
from vllm_omni.model_executor.models.dots_tts.fused_adaln_kernel import (
    indexed_gate_layer_norm_scale_shift,
    layer_norm_indexed_scale_shift,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _indices(batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.arange(batch_size).repeat_interleave(sequence_length)


def test_layer_norm_indexed_scale_shift_matches_reference() -> None:
    torch.manual_seed(0)
    batch_size, sequence_length, hidden_size = 3, 5, 16
    x = torch.randn(batch_size * sequence_length, hidden_size)
    weight = torch.randn(hidden_size)
    shift = torch.randn(batch_size, hidden_size)
    scale = torch.randn(batch_size, hidden_size)
    indices = _indices(batch_size, sequence_length)
    eps = 1e-5

    actual = layer_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
    expected = F.layer_norm(x, (hidden_size,), weight, None, eps)
    expected = expected * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    torch.testing.assert_close(actual, expected)


def test_indexed_gate_layer_norm_scale_shift_matches_reference() -> None:
    torch.manual_seed(1)
    batch_size, sequence_length, hidden_size = 2, 4, 16
    residual = torch.randn(batch_size * sequence_length, hidden_size)
    branch = torch.randn_like(residual)
    gate = torch.randn(batch_size, hidden_size)
    weight = torch.randn(hidden_size)
    shift = torch.randn(batch_size, hidden_size)
    scale = torch.randn(batch_size, hidden_size)
    indices = _indices(batch_size, sequence_length)
    eps = 1e-5

    actual_residual, actual_modulated = indexed_gate_layer_norm_scale_shift(
        residual, gate, branch, weight, shift, scale, indices, eps
    )
    expected_residual = residual + gate.index_select(0, indices) * branch
    expected_modulated = F.layer_norm(expected_residual, (hidden_size,), weight, None, eps)
    expected_modulated = expected_modulated * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(actual_modulated, expected_modulated)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_layer_norm_kernels_match_bf16_cuda_reference() -> None:
    torch.manual_seed(4)
    batch_size, sequence_length, hidden_size = 2, 3, 1024
    x = torch.randn(batch_size * sequence_length, hidden_size, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    indices = _indices(batch_size, sequence_length).to("cuda")
    eps = 1e-5

    actual_first = layer_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
    actual_residual, actual_second = indexed_gate_layer_norm_scale_shift(
        x, gate, branch, weight, shift, scale, indices, eps
    )

    expected_first = F.layer_norm(x, (hidden_size,), weight, None, eps)
    expected_first = expected_first * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    expected_residual = x + gate.index_select(0, indices) * branch
    expected_second = F.layer_norm(expected_residual, (hidden_size,), weight, None, eps)
    expected_second = expected_second * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    torch.testing.assert_close(actual_first, expected_first, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_residual, expected_residual, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_second, expected_second, atol=3e-2, rtol=3e-2)


def test_dots_qkv_qk_norm_rope_layout_matches_reference() -> None:
    torch.manual_seed(5)
    batch_size, sequence_length, num_heads, head_dim = 2, 4, 4, 8
    qkv = torch.randn(batch_size, sequence_length, 3 * num_heads * head_dim)
    q_weight = torch.randn(head_dim)
    k_weight = torch.randn(head_dim)
    freqs = torch.randn(sequence_length, head_dim // 2)
    rope_table = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    eps = 1e-5

    actual_q, actual_k, actual_v = _qkv_qk_norm_rope(
        qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        rope_table=rope_table,
        eps=eps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    q, k, v = qkv.chunk(3, dim=-1)
    q = F.rms_norm(q.reshape(batch_size, sequence_length, num_heads, head_dim), (head_dim,), q_weight, eps)
    k = F.rms_norm(k.reshape(batch_size, sequence_length, num_heads, head_dim), (head_dim,), k_weight, eps)
    half = head_dim // 2
    cos = rope_table[:, :half].unsqueeze(0).unsqueeze(2)
    sin = rope_table[:, half:].unsqueeze(0).unsqueeze(2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[..., :half] * cos - x[..., half:] * sin, x[..., half:] * cos + x[..., :half] * sin], dim=-1)

    expected_q = rotate(q).permute(0, 2, 1, 3).contiguous()
    expected_k = rotate(k).permute(0, 2, 1, 3).contiguous()
    expected_v = v.reshape(batch_size, sequence_length, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)


class _IdentityBranch(nn.Module):
    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        return x


def test_dit_block_fused_modulation_matches_original_module_math() -> None:
    torch.manual_seed(2)
    hidden_size = 16
    block = DiTBlock(_IdentityBranch(), _IdentityBranch(), hidden_size=hidden_size, modulation=True)
    with torch.no_grad():
        block.adaLN_modulation[-1].weight.normal_()
        block.adaLN_modulation[-1].bias.normal_()

    x = torch.randn(2, 3, hidden_size)
    condition = torch.randn(2, hidden_size)
    actual = block(x, condition=condition)

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = block.adaLN_modulation(condition).chunk(6, dim=1)
    attn_branch = modulate(block.norm1(x), shift_a, scale_a)
    residual = x + gate_a.unsqueeze(1) * attn_branch
    expected = residual + gate_f.unsqueeze(1) * modulate(block.norm2(residual), shift_f, scale_f)

    torch.testing.assert_close(actual, expected)


def test_final_layer_fused_modulation_matches_original_module_math() -> None:
    torch.manual_seed(3)
    hidden_size, output_size = 16, 8
    layer = FinalLayer(hidden_size, output_size)
    with torch.no_grad():
        layer.adaLN_modulation[-1].weight.normal_()
        layer.adaLN_modulation[-1].bias.normal_()
        layer.linear.weight.normal_()
        layer.linear.bias.normal_()

    x = torch.randn(2, 3, hidden_size)
    condition = torch.randn(2, hidden_size)
    actual = layer(x, condition)

    shift, scale = layer.adaLN_modulation(condition).chunk(2, dim=1)
    expected = layer.linear(modulate(layer.norm(x), shift, scale))
    torch.testing.assert_close(actual, expected)


class _PackedQKV(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

        def load_shard(param: nn.Parameter, tensor: torch.Tensor, shard_id: str) -> None:
            offsets = {"q": 0, "k": hidden_size, "v": 2 * hidden_size}
            param.data[offsets[shard_id] : offsets[shard_id] + hidden_size].copy_(tensor)

        self.weight.weight_loader = load_shard


class _AttentionContainer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.qkv_proj = _PackedQKV(hidden_size)


class _TinyQKVModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = _AttentionContainer(hidden_size)


def _qkv_shards(hidden_size: int) -> list[tuple[str, torch.Tensor]]:
    return [
        ("blocks.0.attn.q_proj.weight", torch.full((hidden_size, hidden_size), 1.0)),
        ("blocks.0.attn.k_proj.weight", torch.full((hidden_size, hidden_size), 2.0)),
        ("blocks.0.attn.v_proj.weight", torch.full((hidden_size, hidden_size), 3.0)),
    ]


def test_split_qkv_checkpoint_weights_are_packed_in_qkv_order() -> None:
    hidden_size = 8
    model = _TinyQKVModel(hidden_size)
    loaded = _load_stacked_qkv_weights(model, _qkv_shards(hidden_size), component="test")

    assert loaded == {"blocks.0.attn.qkv_proj.weight"}
    expected = torch.cat([weight for _, weight in _qkv_shards(hidden_size)], dim=0)
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight, expected)


@pytest.mark.parametrize("dropped", [0, 1, 2])
def test_incomplete_qkv_checkpoint_weights_raise(dropped: int) -> None:
    hidden_size = 8
    model = _TinyQKVModel(hidden_size)
    shards = _qkv_shards(hidden_size)
    del shards[dropped]

    with pytest.raises(ValueError, match="incomplete QKV shards"):
        _load_stacked_qkv_weights(model, shards, component="test")
