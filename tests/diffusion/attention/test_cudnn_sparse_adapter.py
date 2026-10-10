# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.block_sparse import (
    check_dynamic_selected_execution,
    check_dynamic_sparse_owners,
    make_attention_inputs,
    selected_attention_reference,
)
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionBackend, CuDNNSparseAdapter
from vllm_omni.diffusion.attention.block_selection.abstract import BlockSelection

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
def test_future_kernel_requirements_are_not_duplicated(monkeypatch):
    calls = []

    def future_kernel(q, k, v, indices, **kwargs):
        calls.append(kwargs)
        return q.new_zeros((*q.shape[:-1], v.shape[-1])), None

    monkeypatch.setattr(CuDNNSparseAdapter, "_load_api", staticmethod(lambda: (future_kernel, "future")))
    adapter = CuDNNSparseAdapter()
    # Deliberately outside today's kernel table: provider code, not a local
    # dtype/head-size/block-size allowlist, decides whether these work.
    adapter.prepare("auto", 160, 4, 2, torch.device("cuda"), (96, 96))
    q = torch.zeros(1, 97, 4, 160, dtype=torch.float32)
    k = torch.zeros(1, 101, 2, 160, dtype=torch.float32)
    v = torch.zeros(1, 101, 2, 80, dtype=torch.float32)
    indices = torch.zeros(1, 4, 2, 1, dtype=torch.int32)
    counts = torch.ones(1, 4, 2, dtype=torch.int32)
    result = adapter.execute(q, k, v, BlockSelection(indices, counts), 0.1, (96, 96))
    assert result.shape == (1, 97, 4, 80)
    assert calls[0]["sparse_block_size"] == 96
    assert calls[0]["pack_gqa"] is False
    assert calls[0]["q2k_block_nums"] is counts
    assert calls[0]["block_sizes"].tolist() == [96, 5]


@pytest.mark.cpu
def test_api_representation_constraints_fail_before_import(monkeypatch):
    monkeypatch.setattr(CuDNNSparseAdapter, "_load_api", staticmethod(lambda: pytest.fail("loaded provider")))
    adapter = CuDNNSparseAdapter()
    with pytest.raises(ValueError, match="one block size"):
        adapter.prepare("auto", 128, 4, 2, torch.device("cuda"), (64, 128))
    with pytest.raises(ValueError, match="no kernel-ID"):
        adapter.prepare("invented", 128, 4, 2, torch.device("cuda"), (64, 64))


@pytest.mark.cpu
def test_provider_errors_propagate(monkeypatch):
    error = ImportError("missing BSA")

    def missing():
        raise error

    monkeypatch.setattr(CuDNNSparseAdapter, "_load_api", staticmethod(missing))
    with pytest.raises(ImportError) as result:
        CuDNNSparseAdapter().prepare("auto", 128, 4, 2, torch.device("cuda"), (64, 64))
    assert result.value is error

    execution_error = NotImplementedError("installed kernel rejects this request")

    def unsupported(*args, **kwargs):
        raise execution_error

    monkeypatch.setattr(CuDNNSparseAdapter, "_load_api", staticmethod(lambda: (unsupported, "test")))
    q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, 1, 1, dtype=torch.int32)
    counts = torch.ones(1, 1, 1, dtype=torch.int32)
    with pytest.raises(NotImplementedError) as result:
        CuDNNSparseAdapter().execute(q, q, q, BlockSelection(indices, counts), 0.1, (64, 64))
    assert result.value is execution_error


@pytest.fixture
def hopper_bsa():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("These native BSA tests target Hopper")
    pytest.importorskip("cudnn.block_sparse_attention")
    yield
    torch.compiler.reset()


@pytest.mark.cuda
@torch.inference_mode()
def test_native_ignored_selection_padding(hopper_bsa):
    q, k, v = make_attention_inputs(heads=4, kv_heads=1, q_len=128, kv_len=193)
    indices = torch.tensor([0, 3, -999], device=q.device, dtype=torch.int32).expand(2, 4, 2, 3).contiguous()
    counts = torch.full((2, 4, 2), 2, device=q.device, dtype=torch.int32)
    selection = BlockSelection(indices, counts)
    adapter = CuDNNSparseAdapter()
    adapter.prepare("auto", 128, 4, 1, q.device, (64, 64))
    expected = selected_attention_reference(q, k, v, selection, 0.1, (64, 64))
    actual = adapter.execute(q, k, v, selection, 0.1, (64, 64))
    torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)


@pytest.mark.cuda
@torch.inference_mode()
def test_native_role_dispatch_request_preparation_and_fullgraph(hopper_bsa, monkeypatch):
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig

    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    config = AttentionConfig(
        per_role={
            "cosmos3.gen": {
                "name": "block_sparse",
                "config": {"backend": {"require": "CUDNN_ATTN"}},
            }
        }
    )
    cfg = SimpleNamespace(
        diffusion_attention_config=config,
        parallel_config=SimpleNamespace(ring_degree=1),
        dtype=torch.bfloat16,
        diffusion_kv_cache_dtype="float",
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices=None,
    )
    with set_current_diffusion_config(cfg):
        layer = layer_mod.Attention(4, 128, False, 0.1, num_kv_heads=2, role="cosmos3.gen", role_category="self")
    assert layer.attention.adapter.provider == CuDNNAttentionBackend.get_name()
    q, k, v = (t[:1] for t in make_attention_inputs(q_len=128, kv_len=1089))
    metadata = AttentionMetadata(extra={"protected_kv_prefix": 65})
    impl = layer.attention
    context = ExecutionContext(platform="cuda")
    assert impl.resolve_execution_path(context, q, k, v, metadata).support.status is SupportStatus.UNSUPPORTED
    selection = impl.selector.select(q, k, 0.1, 65)
    expected = selected_attention_reference(q, k, v, selection, 0.1, (64, 64))
    eager = layer(q, k, v, metadata)
    torch.testing.assert_close(eager.float(), expected, atol=0.004, rtol=0.02)
    assert impl.resolve_execution_path(context, q, k, v, metadata).support.status is SupportStatus.SUPPORTED
    compiled = torch.compile(layer, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, metadata), eager, atol=0, rtol=0)
    torch.testing.assert_close(compiled(q, k, -v, metadata), -eager, atol=0, rtol=0)

    padded = [torch.cat((t, torch.full_like(t[:, :5], 999)), dim=1) for t in (q, k, v)]
    packed_metadata = AttentionMetadata(
        extra={"protected_kv_prefix": 65},
        packed_padding=PackedPaddingMetadata(
            q.shape[1],
            k.shape[1],
            torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32),
            torch.tensor([0, k.shape[1]], device=q.device, dtype=torch.int32),
        ),
    )
    output = compiled(*padded, packed_metadata)
    torch.testing.assert_close(output[:, :128], eager, atol=0, rtol=0)
    assert torch.count_nonzero(output[:, 128:]) == 0


@pytest.mark.cuda
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@torch.inference_mode()
def test_dynamic_fullgraph_patterns_and_owners(hopper_bsa, kv_heads):
    torch.compiler.reset()
    try:
        # The installed SM90 BSA kernel requires query lengths divisible by 64.
        check_dynamic_selected_execution(CuDNNSparseAdapter, "auto", kv_heads, query_lengths=(192, 320, 448))
        check_dynamic_sparse_owners(CuDNNSparseAdapter, "auto", kv_heads, query_lengths=(192, 320))
    finally:
        torch.compiler.reset()


@pytest.mark.cuda
@torch.inference_mode()
def test_native_dynamic_rejection_preserves_provider_error(hopper_bsa):
    adapter = CuDNNSparseAdapter()
    adapter.prepare("auto", 128, 4, 4, torch.device("cuda"), (64, 64))
    q, k, v = make_attention_inputs(kv_heads=4, q_len=129, kv_len=321)
    indices = torch.zeros(2, 4, 3, 1, dtype=torch.int32, device=q.device)
    selection = BlockSelection(indices, torch.ones_like(indices[..., 0]).contiguous())

    def execute(q, k, v):
        return adapter.execute(q, k, v, selection, 0.125, (64, 64))

    torch.compiler.reset()
    try:
        for run in (execute, torch.compile(execute, fullgraph=True, dynamic=True)):
            with pytest.raises(NotImplementedError, match="seqlen_q to be a multiple of 64"):
                run(q, k, v)
    finally:
        torch.compiler.reset()
