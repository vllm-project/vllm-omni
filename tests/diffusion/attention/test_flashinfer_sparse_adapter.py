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
from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferSparseAdapter
from vllm_omni.diffusion.attention.block_selection.abstract import BlockSelection

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
def test_future_provider_owns_kernel_compatibility(monkeypatch):
    plans = []

    class FutureWrapper:
        def __init__(self, workspace, backend):
            assert backend == "future-kernel"

        def plan(self, mask, rows, columns, qo_heads, kv_heads, head_dim, **kwargs):
            plans.append((rows, columns, head_dim, kwargs))

        def run(self, q, k, v):
            return torch.zeros_like(q)

    monkeypatch.setattr(FlashInferSparseAdapter, "_load_api", staticmethod(lambda: (FutureWrapper, "future")))
    adapter = FlashInferSparseAdapter()
    adapter.prepare("future-kernel", 160, 2, 2, torch.device("cuda"), (7, 96))
    q = torch.zeros(1, 14, 2, 160, dtype=torch.float32)
    k = torch.zeros(1, 100, 2, 160, dtype=torch.float32)
    selection = BlockSelection(torch.zeros(1, 2, 2, 1, dtype=torch.int32), torch.ones(1, 2, 2, dtype=torch.int32))
    result = adapter.execute(q, k, k, selection, 0.1, (7, 96))
    assert result.shape == q.shape
    rows, columns, head_dim, options = plans[0]
    assert rows[0].tolist() == [7, 7]
    assert columns[0].tolist() == [96, 4]
    assert head_dim == 160
    assert options["q_data_type"] == torch.float32


@pytest.mark.cpu
def test_rejects_unsupported_heads_before_dependency_loading(monkeypatch):
    def unexpected_load():
        pytest.fail("Unsupported head mapping must fail before loading kernels")

    monkeypatch.setattr(FlashInferSparseAdapter, "_load_api", staticmethod(unexpected_load))
    with pytest.raises(ValueError, match="requires MHA"):
        FlashInferSparseAdapter().prepare("auto", 128, 4, 2, torch.device("cuda"), (64, 64))


@pytest.mark.cpu
def test_dependency_failure_propagates(monkeypatch):
    error = ImportError("missing sparse provider")

    def fail():
        raise error

    monkeypatch.setattr(FlashInferSparseAdapter, "_load_api", staticmethod(fail))
    with pytest.raises(ImportError) as result:
        FlashInferSparseAdapter().prepare("auto", 128, 4, 4, torch.device("cuda"), (64, 64))
    assert result.value is error


@pytest.mark.cpu
def test_runtime_failure_propagates_and_opaque_id_is_forwarded(monkeypatch):
    error = RuntimeError("provider plan failed")
    seen = []

    class FailingWrapper:
        def __init__(self, workspace, backend):
            seen.append(backend)

        def plan(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(FlashInferSparseAdapter, "_load_api", staticmethod(lambda: (FailingWrapper, "test")))
    q = torch.zeros(1, 2, 1, 4, dtype=torch.float16)
    indices = torch.zeros(1, 1, 1, 1, dtype=torch.int32)
    counts = torch.ones(1, 1, 1, dtype=torch.int32)
    with pytest.raises(RuntimeError) as result:
        FlashInferSparseAdapter._run(q, q, q, indices, counts, 0.5, (2, 2), "future-provider-id")
    assert result.value is error
    assert seen == ["future-provider-id"]


@pytest.mark.cuda
@pytest.mark.parametrize("implementation", ["fa2", "fa3"])
@torch.inference_mode()
def test_padding_tails_and_preparation_through_shared_dispatch(implementation):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if implementation == "fa3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This FA3 case requires Hopper")
    pytest.importorskip("flashinfer.sparse")
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention
    from vllm_omni.diffusion.data import BlockSparseAttentionSpec

    q, k, v = make_attention_inputs(heads=2, kv_heads=2, q_len=65, kv_len=130, head_size=64)
    # Direct contract: poisoned inactive entries must be ignored, and the last
    # selected KV block contains only two real tokens.
    indices = torch.tensor([0, 2, -999], device=q.device, dtype=torch.int32).expand(2, 2, 2, 3).contiguous()
    counts = torch.full((2, 2, 2), 2, device=q.device, dtype=torch.int32)
    selection = BlockSelection(indices, counts)
    adapter = FlashInferSparseAdapter()
    adapter.prepare(implementation, 64, 2, 2, q.device, (64, 64))
    expected = selected_attention_reference(q, k, v, selection, 0.125, (64, 64))
    actual = adapter.execute(q, k, v, selection, 0.125, (64, 64))
    torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)

    spec = BlockSparseAttentionSpec(
        name="block_sparse",
        config={
            "backend": {"require": "FLASHINFER_ATTN", "implementation": implementation},
        },
    )
    layer = BlockSparseAttention(2, 2, 64, 0.125, False, "BSHD", spec, adapter=FlashInferSparseAdapter())
    metadata = AttentionMetadata(extra={"protected_kv_prefix": 64})
    eager = layer.forward(q, k, v, metadata)
    compiled = torch.compile(layer.forward, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, metadata), eager, atol=0, rtol=0)
    padded = [torch.cat((t[:1], torch.full_like(t[:1, :5], 999)), dim=1) for t in (q, k, v)]
    packed_metadata = AttentionMetadata(
        extra={"protected_kv_prefix": 64},
        packed_padding=PackedPaddingMetadata(
            q.shape[1],
            k.shape[1],
            torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device),
            torch.tensor([0, k.shape[1]], dtype=torch.int32, device=q.device),
        ),
    )
    padded_result = compiled(*padded, packed_metadata)
    torch.testing.assert_close(padded_result[:, : q.shape[1]], eager[:1], atol=0.004, rtol=0.02)
    assert torch.count_nonzero(padded_result[:, q.shape[1] :]) == 0
    torch.compiler.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("implementation", ["auto", "fa2", "fa3"])
@torch.inference_mode()
def test_dynamic_fullgraph_patterns_and_owners(implementation):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if implementation == "fa3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This FA3 case requires Hopper")
    pytest.importorskip("flashinfer.sparse")
    torch.compiler.reset()
    try:
        check_dynamic_selected_execution(FlashInferSparseAdapter, implementation, 4)
        check_dynamic_sparse_owners(FlashInferSparseAdapter, implementation, 4)
    finally:
        torch.compiler.reset()
