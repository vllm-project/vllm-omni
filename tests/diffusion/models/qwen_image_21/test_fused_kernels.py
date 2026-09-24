# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer
from vllm_omni.diffusion.models.qwen_image_21.ops.qk_norm_rope import apply_qk_norm_rope
from vllm_omni.diffusion.models.qwen_image_21.ops.swiglu import fused_silu_mul

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

HEAD_DIM = 128


def _single_rank(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )


def _eager_qk_norm_rope(qk, q_weight, k_weight, freqs, eps, num_q_heads):
    """Reference chain with the eager rounding: unit -> BF16, then the learned scale."""
    q, k = qk.split([num_q_heads, qk.shape[2] - num_q_heads], dim=-2)

    def run(x, weight):
        unit = (x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
        paired = torch.view_as_complex(unit.float().reshape(*unit.shape[:-1], -1, 2))
        return torch.view_as_real(paired * freqs.unsqueeze(1)).flatten(-2).to(x.dtype)

    return torch.cat([run(q, q_weight), run(k, k_weight)], dim=-2)


def _freqs(seq, device, dtype=torch.complex64):
    index = torch.arange(seq, device=device, dtype=torch.float32)
    angles = torch.outer(index, torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM)
    return torch.polar(torch.ones_like(angles), angles).to(dtype)


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu])],
)
@pytest.mark.parametrize("batch,seq", [(1, 1), (1, 17), (2, 255), (1, 256)])
def test_fused_qk_norm_rope_matches_eager_rounding(monkeypatch, device, batch, seq):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    num_q_heads, num_kv_heads = 4, 4
    total = (num_q_heads + num_kv_heads) * HEAD_DIM
    packed = (torch.randn(batch, seq, total, device=device) * 3.0).to(torch.bfloat16)
    # A strided view over the Q-then-K region of a packed QKV projection, as in the block.
    qk = packed.unflatten(-1, (num_q_heads + num_kv_heads, HEAD_DIM))
    q_weight = torch.linspace(0.5, 1.5, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.linspace(1.5, 0.5, HEAD_DIM, device=device, dtype=torch.bfloat16)
    freqs = _freqs(seq, device)

    q, k = apply_qk_norm_rope(qk, q_weight, k_weight, freqs, 1e-6, num_q_heads)
    want = _eager_qk_norm_rope(qk, q_weight, k_weight, freqs, 1e-6, num_q_heads)
    torch.testing.assert_close(torch.cat([q, k], dim=-2), want, rtol=0, atol=0)
    for tensor in (q, k):
        assert tensor.data_ptr() != packed.data_ptr()


@pytest.mark.cuda
@pytest.mark.gpu
def test_fused_qk_norm_rope_leaves_v_untouched(monkeypatch):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    num_q_heads, num_kv_heads, batch, seq = 4, 4, 1, 64
    q_size, kv_size = num_q_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    packed = (torch.randn(batch, seq, q_size + 2 * kv_size, device="cuda") * 3.0).to(torch.bfloat16)
    value_before = packed[..., q_size + kv_size :].clone()
    qk = packed[..., : q_size + kv_size].unflatten(-1, (num_q_heads + num_kv_heads, HEAD_DIM))
    apply_qk_norm_rope(
        qk,
        torch.ones(HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        torch.ones(HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        _freqs(seq, "cuda"),
        1e-6,
        num_q_heads,
    )
    torch.testing.assert_close(packed[..., q_size + kv_size :], value_before, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
def test_fused_ops_fall_back_for_ineligible_inputs(monkeypatch):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    num_heads, batch, seq = 16, 1, 32
    packed = (torch.randn(batch, seq, num_heads * HEAD_DIM, device="cuda") * 3.0).to(torch.bfloat16)
    q_weight = torch.linspace(0.5, 1.5, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.linspace(1.5, 0.5, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    freqs = _freqs(seq, "cuda")

    # A head slice is not contiguous, so the op must stay on the eager chain.
    sliced = packed.unflatten(-1, (num_heads, HEAD_DIM))[:, :, 2:6]
    assert not sliced.is_contiguous()
    q, k = apply_qk_norm_rope(sliced, q_weight, k_weight, freqs, 1e-6, 2)
    want = _eager_qk_norm_rope(sliced, q_weight, k_weight, freqs, 1e-6, 2)
    torch.testing.assert_close(torch.cat([q, k], dim=-2), want, rtol=0, atol=0)

    # FP16 has no fused kernel.
    half = (torch.randn(batch, seq, 8 * HEAD_DIM, device="cuda") * 3.0).to(torch.float16)
    qk_half = half.unflatten(-1, (8, HEAD_DIM))
    q16, k16 = apply_qk_norm_rope(qk_half, q_weight.half(), k_weight.half(), freqs, 1e-6, 4)
    want16 = _eager_qk_norm_rope(qk_half, q_weight.half(), k_weight.half(), freqs, 1e-6, 4)
    torch.testing.assert_close(torch.cat([q16, k16], dim=-2), want16, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu])],
)
@pytest.mark.parametrize("elements", [1, 4095, 4096, 12288, 100000])
def test_fused_silu_mul_matches_eager(device, elements):
    torch.manual_seed(0)
    gate = (torch.randn(elements, device=device) * 4.0).to(torch.bfloat16)
    up = (torch.randn(elements, device=device) * 4.0).to(torch.bfloat16)
    got = fused_silu_mul(gate, up)
    want = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.data_ptr() != gate.data_ptr() and got.data_ptr() != up.data_ptr()


def _randomize(module):
    """vLLM linear layers start uninitialized; the checkpoint loader normally fills them."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)


class _RecordingAttention(torch.nn.Module):
    """Deterministic stand-in for the real attention layer that depends on Q, K and V."""

    def forward(self, query, key, value, metadata=None):  # noqa: D102
        del metadata
        return query + key.mean(dim=1, keepdim=True) + value.mean(dim=1, keepdim=True)


def _iterative_qk_norm_rope(qk, q_weight, k_weight, freqs, eps, num_q_heads):
    return tuple(
        apply_qk_norm_rope(qk, q_weight, k_weight, freqs, eps, num_q_heads)
    )


@pytest.mark.cuda
@pytest.mark.gpu
def test_attention_forward_is_unchanged_by_fusion(monkeypatch):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    attention = transformer.QwenImage21Attention(dim=512, heads=4, dim_head=HEAD_DIM, eps=1e-6).to(
        device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        attention.norm_q.weight.copy_(torch.linspace(0.5, 1.5, HEAD_DIM, device="cuda", dtype=torch.bfloat16))
        attention.norm_k.weight.copy_(torch.linspace(1.5, 0.5, HEAD_DIM, device="cuda", dtype=torch.bfloat16))
    attention.attn = _RecordingAttention()
    _randomize(attention)

    batch, seq = 2, 48
    hidden = (torch.randn(batch, seq, 512, device="cuda") * 2.0).to(torch.bfloat16)
    freqs = _freqs(seq, "cuda")

    fused = attention(hidden, freqs)

    original = transformer.apply_qk_norm_rope
    monkeypatch.setattr(transformer, "apply_qk_norm_rope", _iterative_qk_norm_rope)
    try:
        unfused = attention(hidden, freqs)
    finally:
        monkeypatch.setattr(transformer, "apply_qk_norm_rope", original)
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
def test_mlp_forward_is_unchanged_by_fusion(monkeypatch):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    mlp = transformer.QwenImage21SwiGLUFeedForward(hidden_size=512, mlp_hidden_size=1536).to(
        device="cuda", dtype=torch.bfloat16
    )
    _randomize(mlp)
    hidden = (torch.randn(2, 48, 512, device="cuda") * 2.0).to(torch.bfloat16)
    fused = mlp(hidden)
    want = mlp.out(torch.nn.functional.silu(mlp.gate_layer(hidden)) * mlp.proj(hidden))
    torch.testing.assert_close(fused, want, rtol=0, atol=0)
