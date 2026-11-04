# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, AttentionRanges
from vllm_omni.diffusion.attention.backends.flex_attention import FlexAttentionBackend
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.attention import ARDiffusionKVLayerContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@torch.no_grad()
def test_registered_backend_runs_exact_ranges_through_shared_attention():
    assert DiffusionAttentionBackendEnum.FLEX_ATTENTION.get_class() is FlexAttentionBackend
    attention = Attention(2, 8, causal=False, softmax_scale=8**-0.5, default_backend="FLEX_ATTENTION")
    assert not attention._has_custom_attention
    q, k, v = [torch.randn(1, length, 2, 8) for length in (3, 5, 5)]
    ranges = [[(0, 2, 0, 3), (2, 3, 3, 5)]]
    qm, km = torch.tensor([[True, False, True]]), torch.tensor([[True, False, True, True, True]])
    mask = torch.tensor([[[[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]]], dtype=torch.bool)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    actual = attention(q, k, v, AttentionMetadata(range_mask=AttentionRanges(ranges, qm, km)))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[:, 1]) == 0


def test_explicit_backend_overrides_model_default(monkeypatch):
    calls = []

    def resolve(name, head_size):
        calls.append((name, head_size))
        return SDPABackend

    monkeypatch.setattr("vllm_omni.diffusion.attention.selector._cached_get_backend_cls", resolve)
    backend, spec = get_attn_backend_for_role(
        "test",
        8,
        attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        default_backend="FLEX_ATTENTION",
    )
    assert backend is SDPABackend and spec.backend == "TORCH_SDPA"
    assert calls == [("TORCH_SDPA", 8)]


@torch.no_grad()
def test_unsupported_backend_cannot_ignore_exact_ranges():
    attention = Attention(2, 8, causal=False, softmax_scale=8**-0.5, default_backend="TORCH_SDPA")
    q = torch.randn(1, 3, 2, 8)
    with pytest.raises(NotImplementedError, match="supporting range_mask"):
        attention(q, q, q, AttentionMetadata(range_mask=AttentionRanges()))


@torch.no_grad()
@pytest.mark.parametrize("cache_conditioning", [False, True])
@pytest.mark.parametrize("update_cache", [False, True])
def test_shared_kv_assembly_reuses_history_and_preserves_media_order(cache_conditioning, update_cache):
    keys, values = torch.zeros(1, 16, 2, 8), torch.zeros(1, 16, 2, 8)
    history = torch.randn(1, 3, 2, 8)
    keys[:, :3], values[:, :3] = history, -history
    current = torch.randn(1, 5, 2, 8)
    context = ARDiffusionKVLayerContext(0, 3, 4, cache_conditioning, update_cache, keys, values)
    full_key, full_value = context.assemble(current, -current)
    count = 5 if cache_conditioning or update_cache else 4
    torch.testing.assert_close(full_key, torch.cat((history, current[:, :count]), 1), rtol=0, atol=0)
    torch.testing.assert_close(full_value, -full_key, rtol=0, atol=0)
    assert torch.equal(keys[:, :3], history)
    if cache_conditioning or not update_cache:
        assert full_key.untyped_storage().data_ptr() == keys.untyped_storage().data_ptr()


@torch.no_grad()
def test_shared_attention_consumes_kv_context_before_backend():
    attention = Attention(2, 8, causal=False, softmax_scale=8**-0.5, default_backend="FLEX_ATTENTION")
    keys, values = torch.zeros(1, 16, 2, 8), torch.zeros(1, 16, 2, 8)
    keys[:, :3], values[:, :3] = torch.randn(1, 3, 2, 8), torch.randn(1, 3, 2, 8)
    q, k, v = [torch.randn(1, 2, 2, 8) for _ in range(3)]
    context = ARDiffusionKVLayerContext(0, 3, 2, False, False, keys, values)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        torch.cat((keys[:, :3], k), 1).transpose(1, 2),
        torch.cat((values[:, :3], v), 1).transpose(1, 2),
    ).transpose(1, 2)
    actual = attention(q, k, v, AttentionMetadata(kv_context=context))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_shared_flex_consumes_prebuilt_block_mask_with_compact_gqa():
    attention = Attention(4, 8, causal=False, softmax_scale=8**-0.5, num_kv_heads=2, default_backend="FLEX_ATTENTION")
    q, k, v = torch.randn(1, 4, 4, 8), torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8)

    def predicate(batch, head, query, key):
        return query // 2 == key // 2

    block_mask = create_block_mask(predicate, None, None, 4, 4, device="cpu")
    metadata = AttentionMetadata(range_mask=AttentionRanges(), extra={"block_mask": block_mask})
    actual = attention(q, k, v, metadata)
    mask = (torch.arange(4)[:, None] // 2 == torch.arange(4)[None] // 2)[None, None]
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(2, 2).transpose(1, 2),
        v.repeat_interleave(2, 2).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
