# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the Ascend image-prefix adapter with an explicit joint-full mask."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the device-independent helper without importing accelerator plugins.
_PATH = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/models/hunyuan_image3.py"
_SPEC = importlib.util.spec_from_file_location("hy3_npu_image_prefix", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("cached", [0, 5, 9])
def test_image_prefix_matches_masked_sdpa_with_paged_context(cached):
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(10, 4, 8, generator=generator)[cached:]
    key = torch.randn(10, 2, 8, generator=generator)
    value = torch.randn(10, 2, 8, generator=generator)
    table = torch.tensor([[2, 0, 1]])
    key_cache = torch.full((3, 4, 2, 8), 1000.0)
    value_cache = key_cache.clone()
    for position in range(10):
        block = table[0, position // 4]
        key_cache[block, position % 4] = key[position]
        value_cache[block, position % 4] = value[position]
    original_key, original_value = key_cache.clone(), value_cache.clone()
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[len(query)],
        seq_lens_list=[10],
        block_tables=table,
        mm_prefix_range={0: [(1, 3), (6, 8)]},
    )

    def attend(q, k, v, mask=None):
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        return scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
        ).transpose(0, 1)

    key_positions = torch.arange(10)
    query_positions = torch.arange(cached, 10)
    causal = key_positions[None, :] <= query_positions[:, None]
    expected_mask = causal.clone()
    for start, end in metadata.mm_prefix_range[0]:
        same_image = (query_positions >= start) & (query_positions <= end)
        expected_mask |= same_image[:, None] & (key_positions[None, :] <= end)
    expected = attend(query, key, value, expected_mask)
    output = attend(query, key, value, causal)
    calls = []

    def fia(**kwargs):
        calls.append(kwargs["actual_seq_lengths_kv"][0])
        length = calls[-1]
        pages = kwargs["block_table"][0]
        k = kwargs["key"].reshape_as(key_cache)[pages].reshape(-1, 2, 8)[:length]
        v = kwargs["value"].reshape_as(value_cache)[pages].reshape(-1, 2, 8)[:length]
        return attend(kwargs["query"], k, v), None

    actual = _MODULE.correct_image_queries(
        query,
        key_cache,
        value_cache,
        metadata,
        output,
        num_heads=4,
        num_kv_heads=2,
        scale=8**-0.5,
        fia=fia,
    )
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(key_cache, original_key) and torch.equal(value_cache, original_value)
    assert calls == {0: [4, 9], 5: [9], 9: []}[cached]


def test_incomplete_bidirectional_image_is_rejected():
    metadata = SimpleNamespace(actual_seq_lengths_q=[7], seq_lens_list=[7], mm_prefix_range={0: [(1, 8)]})
    with pytest.raises(ValueError, match="incomplete bidirectional image"):
        list(_MODULE.image_segments(metadata))


@pytest.fixture
def native_backend(monkeypatch):
    # Check the backend contract on CPU; loading Ascend ops can query a device.
    class Impl:
        def __init__(self, *, pcp=1, dcp=1):
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp)
            )

    class Backend:
        @staticmethod
        def get_impl_cls():
            return Impl

    native = SimpleNamespace(
        AscendAttentionBackend=Backend,
        AscendAttentionBackendImpl=Impl,
        AscendAttentionState=SimpleNamespace(DecodeOnly=object(), PrefillCacheHit=object()),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", native)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=Mock()))
    _MODULE.get_image_prefix_backend.cache_clear()
    yield native
    _MODULE.get_image_prefix_backend.cache_clear()


@pytest.mark.parametrize("decode,spans", [(True, True), (False, False), (False, True)])
def test_explicit_backend_preserves_native_forward_and_corrects_only_image_prefill(
    monkeypatch, native_backend, decode, spans
):
    native = native_backend
    backend = _MODULE.get_image_prefix_backend()
    assert backend is _MODULE.get_image_prefix_backend()
    assert issubclass(backend, native.AscendAttentionBackend)
    assert native.AscendAttentionBackend.get_impl_cls() is native.AscendAttentionBackendImpl
    impl = object.__new__(backend.get_impl_cls())
    impl.key_cache = impl.value_cache = torch.empty(1)
    impl.num_heads = impl.num_kv_heads = 1
    impl.scale = 1.0
    original = Mock(return_value=torch.empty(1))
    correct = Mock(return_value=original.return_value)
    monkeypatch.setattr(native.AscendAttentionBackendImpl, "forward_fused_infer_attention", original, raising=False)
    monkeypatch.setattr(_MODULE, "correct_image_queries", correct)
    metadata = SimpleNamespace(
        attn_state=native.AscendAttentionState.DecodeOnly if decode else native.AscendAttentionState.PrefillCacheHit,
        mm_prefix_range={0: [(0, 0)]} if spans else {},
    )
    args = (torch.empty(1), torch.empty(1), torch.empty(1), metadata, torch.empty(1))
    assert impl.forward_fused_infer_attention(*args) is original.return_value
    original.assert_called_once_with(*args, None)
    assert correct.call_count == int(spans and not decode)


@pytest.mark.parametrize("pcp,dcp", [(1, 1), (2, 1), (1, 2)])
def test_explicit_backend_rejects_unsupported_context_parallelism(native_backend, pcp, dcp):
    impl = _MODULE.get_image_prefix_backend().get_impl_cls()
    if pcp == dcp == 1:
        impl(pcp=pcp, dcp=dcp)
    else:
        with pytest.raises(ValueError, match="PCP=1 and DCP=1"):
            impl(pcp=pcp, dcp=dcp)
