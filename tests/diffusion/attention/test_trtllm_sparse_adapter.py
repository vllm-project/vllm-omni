# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.block_sparse import (
    check_dynamic_selected_execution,
    check_dynamic_sparse_owners,
    make_attention_inputs,
    selected_attention_reference,
)
from vllm_omni.diffusion.attention.backends import trtllm_attn as provider
from vllm_omni.diffusion.attention.block_selection.abstract import BlockSelection

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
def test_future_provider_owns_kernel_compatibility(monkeypatch):
    plans = []

    class FutureWrapper:
        def plan(self, **kwargs):
            plans.append(kwargs)

        def run(self, q, k, v, **kwargs):
            return torch.zeros_like(q)

    monkeypatch.setattr(provider.TrtllmSparseAdapter, "_load_api", staticmethod(lambda: (FutureWrapper, "future")))
    adapter = provider.TrtllmSparseAdapter()
    adapter.prepare("auto", 160, 2, 2, torch.device("cuda"), (7, 96))
    q = torch.zeros(1, 14, 2, 160, dtype=torch.float32)
    k = torch.zeros(1, 100, 2, 160, dtype=torch.float32)
    selection = BlockSelection(torch.zeros(1, 2, 2, 1, dtype=torch.int32), torch.ones(1, 2, 2, dtype=torch.int32))
    result = adapter.execute(q, k, k, selection, 0.1, (7, 96))
    assert result.shape == q.shape
    assert plans[0]["head_dim"] == 160
    assert plans[0]["q_block_size"] == 7
    assert plans[0]["kv_block_size"] == 96
    assert plans[0]["q_data_type"] == torch.float32


@pytest.mark.cpu
def test_exact_bit_words_ignore_padding_and_preserve_head_rows():
    indices = torch.tensor([[[[0, 31, 32, 64], [1, 63, -99, 999]]]], dtype=torch.int32).expand(2, 3, 2, 4)
    counts = torch.tensor([[[4, 2]]], dtype=torch.int32).expand(2, 3, 2)
    bits = provider._selected_block_bits(BlockSelection(indices, counts), 65)
    assert bits.dtype == torch.uint32
    assert bits.is_contiguous()
    assert bits.shape == (2, 3, 2, 3)
    expected = torch.tensor([1 + 2**31, 1, 1, 2, 2**31, 0], dtype=torch.int64).reshape(1, 1, 2, 3)
    torch.testing.assert_close(bits.long(), expected.expand_as(bits))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "head_size,heads,kv_heads,device,blocks,message",
    [
        (128, 4, 2, "cuda", (64, 64), "requires MHA"),
        (128, 2, 2, "cpu", (64, 64), "requires CUDA"),
    ],
)
def test_rejects_invalid_contract_before_import(monkeypatch, head_size, heads, kv_heads, device, blocks, message):
    monkeypatch.setattr(provider.TrtllmSparseAdapter, "_load_api", staticmethod(lambda: pytest.fail("loaded provider")))
    with pytest.raises(ValueError, match=message):
        provider.TrtllmSparseAdapter().prepare("auto", head_size, heads, kv_heads, torch.device(device), blocks)


@pytest.mark.cpu
def test_missing_dependency(monkeypatch):
    adapter = provider.TrtllmSparseAdapter()
    error = ImportError("BlockSparseTSWrapper unavailable")

    def fail():
        raise error

    monkeypatch.setattr(adapter, "_load_api", fail)
    with pytest.raises(ImportError) as result:
        adapter.prepare("auto", 128, 2, 2, torch.device("cuda"), (64, 64))
    assert result.value is error


@pytest.mark.cpu
def test_existing_registry_resolves_adapter_without_dense_probe(monkeypatch):
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.data import AttentionConfig

    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *args: pytest.fail("dense probe"))
    config = AttentionConfig(default={"name": "block_sparse", "config": {"backend": {"require": "TRTLLM_ATTN"}}})
    backend, _ = selector.get_attn_backend_for_role("minimax_h3.dit", 128, config)
    assert backend.get_block_sparse_adapter() is provider.TrtllmSparseAdapter
    with pytest.raises(ValueError, match="no kernel-ID interface"):
        provider.TrtllmSparseAdapter.validate_selection("unknown", 128)


@pytest.mark.cpu
def test_provider_error_propagates_without_fallback(monkeypatch):
    error = RuntimeError("provider plan error")

    class FailingWrapper:
        def plan(self, **kwargs):
            raise error

    monkeypatch.setattr(provider.TrtllmSparseAdapter, "_load_api", staticmethod(lambda: (FailingWrapper, "test")))
    q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    selection = BlockSelection(torch.zeros(1, 1, 1, 1, dtype=torch.int32), torch.ones(1, 1, 1, dtype=torch.int32))
    with pytest.raises(RuntimeError) as result:
        provider.TrtllmSparseAdapter._run(q, q, q, selection, 0.1, (64, 64))
    assert result.value is error


def _check_selected_execution():
    q, k, v = make_attention_inputs(heads=2, kv_heads=2, q_len=129, kv_len=2113)
    indices = torch.tensor([0, 31, 33, -99], device=q.device, dtype=torch.int32).expand(2, 2, 3, 4).contiguous()
    counts = torch.full((2, 2, 3), 3, device=q.device, dtype=torch.int32)
    adapter = provider.TrtllmSparseAdapter()
    adapter.prepare("auto", 128, 2, 2, q.device, (64, 64))
    compiled = torch.compile(adapter.execute, fullgraph=True)
    for run, first in [(adapter.execute, 0), (compiled, 1), (compiled, 2)]:
        current = indices.clone()
        current[..., 0] = first
        # Different batch/head patterns must remain independent.
        current[1, 1, :, 0] = first + 3
        selection = BlockSelection(current, counts)
        expected = selected_attention_reference(q, k, v, selection, 0.1, (64, 64))
        actual = run(q, k, v, selection, 0.1, (64, 64))
        torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)
    torch.compiler.reset()


@pytest.fixture
def mock_sparse_provider(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for metadata/compile test")
    instances = []

    class ReferenceWrapper:
        def __init__(self):
            instances.append(self)

        def plan(self, **kwargs):
            assert kwargs["sparse_format"] == "bitmask"
            assert not kwargs["use_proxy_routes"]
            assert not kwargs["use_kv_valid_bits"]
            self.options = kwargs

        def run(self, q, k, v, *, exact_block_bits, sm_scale):
            assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            assert exact_block_bits.dtype == torch.uint32
            ids = torch.arange((k.shape[1] + 63) // 64, device=k.device)
            mask = ((exact_block_bits.long()[..., ids // 32] >> (ids % 32)) & 1).bool()
            mask = mask.repeat_interleave(64, 2).repeat_interleave(64, 3)[..., : q.shape[1], : k.shape[1]]
            scores = (q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)) * sm_scale
            weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            return (weights @ v.float().transpose(1, 2)).transpose(1, 2).to(q.dtype).contiguous()

    monkeypatch.setattr(provider.TrtllmSparseAdapter, "_load_api", staticmethod(lambda: (ReferenceWrapper, "mock")))
    return instances


@pytest.mark.cuda
@torch.inference_mode()
def test_mock_provider_conversion_and_compiled_boundary(mock_sparse_provider):
    _check_selected_execution()
    assert len(mock_sparse_provider) == 3  # Fake tracing does not plan; live calls own separate workspaces.


@pytest.mark.cuda
@torch.inference_mode()
def test_blackwell_kernel_selected_attention():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Actual PrimTS kernel validation requires SM100/SM103")
    pytest.importorskip("flashinfer.attention.prims_ts")
    _check_selected_execution()


@pytest.mark.cuda
@pytest.mark.parametrize("native", [False, True], ids=["mock-provider", "native-blackwell"])
@torch.inference_mode()
def test_dynamic_fullgraph_patterns_and_owners(request, native):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if native:
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("Actual PrimTS kernel validation requires SM100/SM103")
        pytest.importorskip("flashinfer.attention.prims_ts")
    else:
        request.getfixturevalue("mock_sparse_provider")
    torch.compiler.reset()
    try:
        check_dynamic_selected_execution(provider.TrtllmSparseAdapter, "auto", 4)
        check_dynamic_sparse_owners(provider.TrtllmSparseAdapter, "auto", 4)
    finally:
        torch.compiler.reset()
