# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU dense fallback numerics on CPU, plus dispatch-only kernel contracts.

The accuracy tests execute real SDPA against a float64 einsum/softmax reference.
They do not substitute an attention kernel. Native NPU accuracy is covered by
``test_flash_attn_npu.py`` and requires Ascend hardware.
"""

import sys
from types import ModuleType

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _reference(q, k, v, mask, causal, scale):
    # Explicitly map each Q head to its KV group, independently of SDPA.
    head_indices = torch.arange(q.shape[2]) // (q.shape[2] // k.shape[2])
    scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), k.double()[:, :, head_indices]) * scale
    if mask is not None:
        if mask.ndim == 2 and mask.shape == (q.shape[0], k.shape[1]):
            mask = mask.bool()[:, None, None, :]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -torch.inf)
        else:
            scores = scores + mask.double()
    if causal:
        keep = torch.arange(k.shape[1])[None, :] <= torch.arange(q.shape[1])[:, None] + k.shape[1] - q.shape[1]
        scores = scores.masked_fill(~keep, -torch.inf)
    # Empty causal/prefix rows have zero output, rather than NaN.
    probs = torch.softmax(scores, dim=-1).nan_to_num()
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.double()[:, :, head_indices])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("q_len,kv_len", [(1, 7), (3, 7), (7, 3)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask_kind", ["prefix", "bias", "key_keep", "none"])
def test_dense_fallback_matches_independent_reference(dtype, q_len, kv_len, causal, mask_kind):
    torch.manual_seed(91)
    # An unqualified head size also exercises the mask-free/Boolean fallback.
    q = torch.randn(2, q_len, 8, 16).to(dtype)
    k = torch.randn(2, kv_len, 2, 16).to(dtype)
    v = torch.randn_like(k)
    mask = None
    if mask_kind in ("prefix", "bias"):
        mask = torch.zeros(2, 1, q_len, kv_len, dtype=dtype)
        mask[..., 1] = -torch.inf
        mask[0, :, 0, :] = -torch.inf
        if mask_kind == "bias":
            mask[..., -1] = 0.375
    elif mask_kind == "key_keep":
        mask = torch.ones(2, kv_len, dtype=torch.bool)
        mask[:, 1] = False
        mask[0] = False
    impl = FlashAttentionImpl(num_heads=8, num_kv_heads=2, head_size=16, softmax_scale=0.37, causal=causal)
    got = impl.forward_fa_npu(q, k, v, AttentionMetadata(attn_mask=mask))
    expected = _reference(q, k, v, mask, causal, 0.37)
    assert got.dtype == dtype and got.shape == q.shape
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.double(), expected, atol=2e-2 if dtype == torch.bfloat16 else 2e-6, rtol=2e-2)


def test_additive_mask_does_not_import_npu_runtimes(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    q = torch.randn(1, 3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 7, 2, 64, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 3, 7, dtype=torch.bfloat16)
    mask[..., :2] = -torch.inf
    impl = FlashAttentionImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.23)
    got = impl.forward_fa_npu(q, k, k, AttentionMetadata(attn_mask=mask))
    torch.testing.assert_close(got.double(), _reference(q, k, k, mask, False, 0.23), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_kernel_execution_errors_propagate(monkeypatch, causal):
    """Dispatch contract only: this test does not claim native kernel accuracy."""
    runtime = ModuleType("torch_npu" if causal else "mindiesd")
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("device kernel failure")

    setattr(runtime, "npu_fusion_attention" if causal else "attention_forward", execute)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    q = torch.randn(1, 3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 7, 2, 64, dtype=torch.bfloat16)
    impl = FlashAttentionImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.37, causal=causal)
    with pytest.raises(RuntimeError, match="device kernel failure"):
        impl.forward_fa_npu(q, k, k)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs["scale"] == 0.37
    assert args[1].shape[2] == 2


@pytest.mark.parametrize("q_len,kv_len", [(0, 7), (3, 0)])
@pytest.mark.parametrize("causal", [False, True])
def test_empty_sequences_use_sdpa_without_npu_kernels(monkeypatch, q_len, kv_len, causal):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    q = torch.randn(2, q_len, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, kv_len, 2, 64, dtype=torch.bfloat16)
    impl = FlashAttentionImpl(num_heads=4, num_kv_heads=2, head_size=64, softmax_scale=0.23, causal=causal)
    result = impl.forward_fa_npu(q, k, k)
    assert result.shape == q.shape
    torch.testing.assert_close(result, torch.zeros_like(q), rtol=0, atol=0)


@pytest.mark.parametrize("q_len", [1, 3])
@pytest.mark.parametrize("mask_dtype", [torch.bfloat16, torch.float32])
def test_cached_prefix_mask_preserves_all_prefix_keys(q_len, mask_dtype):
    torch.manual_seed(93)
    prefix_len = 4
    kv_len = prefix_len + q_len
    q = torch.randn(1, q_len, 8, 64, dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 2, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.zeros(1, 1, q_len, kv_len, dtype=mask_dtype)
    # Cached prefix keys remain visible to every query; only new keys are causal.
    for row in range(q_len):
        mask[..., row, prefix_len + row + 1 :] = -torch.inf
    impl = FlashAttentionImpl(num_heads=8, num_kv_heads=2, head_size=64, softmax_scale=0.23, causal=False)
    got = impl.forward_fa_npu(q, k, v, AttentionMetadata(attn_mask=mask))
    expected = _reference(q, k, v, mask, False, 0.23)
    torch.testing.assert_close(got.double(), expected, atol=2e-2, rtol=2e-2)
