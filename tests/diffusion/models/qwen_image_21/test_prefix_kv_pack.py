# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer
from vllm_omni.diffusion.models.qwen_image_21.ops.prefix_kv import concat_prefix_kv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

HEAD_DIM = 128
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _single_rank(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )


def _quantize(t):
    scale = t.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) / FP8_MAX
    return (t / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def _eager_concat(prefix, prefix_scale, target):
    if prefix_scale is not None:
        prefix = (prefix.float() * prefix_scale).to(target.dtype)
    return torch.cat([prefix, target], dim=1)


def _freqs(seq, device):
    index = torch.arange(seq, device=device, dtype=torch.float32)
    angles = torch.outer(index, torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM)
    return torch.polar(torch.ones_like(angles), angles)


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu])],
)
@pytest.mark.parametrize("batch,prefix_len,target_len", [(1, 512, 256), (1, 1, 64), (2, 33, 7)])
@pytest.mark.parametrize("heads", [4, 3])
def test_concat_prefix_kv_matches_eager_dequant(device, batch, prefix_len, target_len, heads):
    torch.manual_seed(0)
    prefix = (torch.randn(batch, prefix_len, heads, HEAD_DIM, device=device) * 3.0).to(torch.bfloat16)
    target = (torch.randn(batch, target_len, heads, HEAD_DIM, device=device) * 3.0).to(torch.bfloat16)
    fp8, scale = _quantize(prefix)

    for payload, prefix_scale in ((fp8, scale), (prefix, None)):
        got = concat_prefix_kv(payload, prefix_scale, target)
        want = _eager_concat(payload, prefix_scale, target)
        assert got.shape == (batch, prefix_len + target_len, heads, HEAD_DIM)
        assert got.data_ptr() != target.data_ptr()
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
def test_fused_path_accepts_the_strided_v_layout(monkeypatch):
    """V is a strided view into the packed qkv output; the fused kernel must still take it."""
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    from vllm_omni.diffusion.models.qwen_image_21.ops import prefix_kv

    torch.manual_seed(0)
    batch, seq, heads, head_dim = 1, 64, 4, HEAD_DIM
    qkv = (torch.randn(batch, seq, 3 * heads * head_dim, device="cuda") * 3.0).to(torch.bfloat16)
    value = qkv[..., 2 * heads * head_dim :].unflatten(-1, (heads, head_dim))
    assert not value.is_contiguous() and value.stride(-1) == 1

    prefix = (torch.randn(batch, 16, heads, head_dim, device="cuda") * 3.0).to(torch.bfloat16)
    fp8, scale = _quantize(prefix)
    assert prefix_kv._supported(fp8, scale, value)
    torch.testing.assert_close(concat_prefix_kv(fp8, scale, value), _eager_concat(fp8, scale, value), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
def test_concat_prefix_kv_falls_back_for_ineligible_inputs():
    torch.manual_seed(0)
    prefix = (torch.randn(1, 8, 4, HEAD_DIM, device="cuda") * 3.0).to(torch.bfloat16)
    target = (torch.randn(1, 8, 4, HEAD_DIM, device="cuda") * 3.0).to(torch.bfloat16)
    # FP16 targets have no fused kernel.
    torch.testing.assert_close(
        concat_prefix_kv(prefix.half(), None, target.half()),
        _eager_concat(prefix.half(), None, target.half()),
        rtol=0,
        atol=0,
    )
    # A 3-D payload is rejected rather than misinterpreted.
    with pytest.raises(ValueError):
        concat_prefix_kv(prefix[0], None, target)
    # Mismatched head counts fall back to the eager chain, which reports the mismatch.
    with pytest.raises(RuntimeError):
        concat_prefix_kv(prefix[:, :, :2], None, target)


class _RecordingAttention(torch.nn.Module):
    """Deterministic stand-in for the real attention layer."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, query, key, value, metadata=None):  # noqa: D102
        del metadata
        self.seen = (key.shape, value.shape)
        return query + key.mean(dim=1, keepdim=True) + value.mean(dim=1, keepdim=True)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cache_dtype", ["fp8_e4m3", "fp8_e4m3_v", None])
def test_attention_decode_is_unchanged_by_prefix_pack_fusion(monkeypatch, cache_dtype):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    attention = transformer.QwenImage21Attention(
        dim=512, heads=4, dim_head=HEAD_DIM, eps=1e-6, prefix_kv_cache_dtype=cache_dtype
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    attention.attn = _RecordingAttention()

    prefix_len, target_len, batch = 48, 16, 2
    prefill = (torch.randn(batch, prefix_len, 512, device="cuda") * 2.0).to(torch.bfloat16)
    decode = (torch.randn(batch, target_len, 512, device="cuda") * 2.0).to(torch.bfloat16)

    def run_pair():
        # Per-block cache: a dict keyed by CFG branch.
        cache = {}
        attention(prefill, _freqs(prefix_len, "cuda"), kv_cache=cache, cache_write_len=prefix_len)
        out = attention(decode, _freqs(target_len, "cuda"), kv_cache=cache, cache_write_len=None)
        return out, cache

    fused, fused_cache = run_pair()
    packed_shape = (batch, prefix_len + target_len, 4, HEAD_DIM)
    assert attention.attn.seen == (packed_shape, packed_shape)
    # The cache keeps only the prefix; the packed tensor is the input the backend sees.
    assert fused_cache["cond"]["key"].shape[1] == prefix_len

    original = transformer.concat_prefix_kv
    monkeypatch.setattr(transformer, "concat_prefix_kv", _eager_concat)
    try:
        unfused, _ = run_pair()
    finally:
        monkeypatch.setattr(transformer, "concat_prefix_kv", original)
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
def test_prefix_cache_layout_is_unchanged(monkeypatch):
    """The FP8 cache must keep the same scale schema and stay out of the CUDA graph path."""
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    attention = transformer.QwenImage21Attention(
        dim=512, heads=4, dim_head=HEAD_DIM, eps=1e-6, prefix_kv_cache_dtype="fp8_e4m3"
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    attention.attn = _RecordingAttention()
    prefix_len = 32
    hidden = (torch.randn(1, prefix_len, 512, device="cuda") * 2.0).to(torch.bfloat16)
    cache = {}
    attention(hidden, _freqs(prefix_len, "cuda"), kv_cache=cache, cache_write_len=prefix_len)
    branch = cache["cond"]
    assert set(branch) == {"key", "value", "key_scale", "value_scale"}
    assert branch["key"].dtype is torch.float8_e4m3fn and branch["value"].dtype is torch.float8_e4m3fn
    assert branch["key_scale"].shape == (1, prefix_len, 4, 1)
