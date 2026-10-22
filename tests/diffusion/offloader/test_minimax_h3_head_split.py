# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for head packing and the H3 adapter's DLO lifecycle."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.submodule.common.head_buckets import HeadBucketPlan
from vllm_omni.diffusion.offloader.submodule.common.head_workspace_cache import HeadWorkspaceCache

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("buckets", [1, 2, 3, 4])
@pytest.mark.parametrize("query_heads", [32, 64])
@pytest.mark.parametrize("shape", [(192,), (192, 7)])
def test_qkv_packing_roundtrip(buckets, shape, query_heads):
    shape = ((query_heads + 64) * 2, *shape[1:])
    layout = HeadBucketPlan(query_heads, 32, 2, 4, buckets)
    value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    torch.testing.assert_close(layout.pack_qkv(layout.pack_qkv(value), restore=True), value, rtol=0, atol=0)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = self.num_kv_heads = 16
        self.head_dim = 2
        self.qkv_proj = nn.Linear(7, 96)
        self.qkv_proj.total_num_heads = 16
        self.out_proj = nn.Linear(32, 7)
        method = type("UnquantizedLinearMethod", (), {})
        self.qkv_proj.quant_method = self.out_proj.quant_method = method()
        self.to_gate_compress = None

    def forward(self, x, **kwargs):
        return self.qkv_proj(x)


@pytest.fixture
def adapter_class(monkeypatch, mocker):
    from tests.diffusion.offloader.test_head_bucket_attention import _cpu_stream_shims

    _cpu_stream_shims(monkeypatch)
    from vllm_omni.diffusion.offloader.submodule.models.minimax_h3.h3_bucket_adapter import H3BucketAdapter

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    monkeypatch.setattr(torch.npu, "Stream", mocker.Mock)
    monkeypatch.setattr(torch.npu, "Event", mocker.Mock)
    return H3BucketAdapter


def test_adapter_disable_restores_forward_and_weights_for_reenable(adapter_class):
    module = TinyAttention()
    expected = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    x = torch.randn(3, 7)
    expected_output = module(x)
    for _ in range(2):
        adapter = adapter_class(module, object(), buckets=4)
        assert not torch.equal(module.qkv_proj.weight, expected["qkv_proj.weight"])
        adapter.close()
        adapter.close()  # A later cleanup retry may revisit an already closed adapter.
        assert "forward" not in module.__dict__
        assert not hasattr(module, "_head_bucket_adapter")
        for name, tensor in module.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        torch.testing.assert_close(module(x), expected_output, rtol=0, atol=0)


def test_adapter_allocation_failure_does_not_reorder_weights(adapter_class, monkeypatch, mocker):
    module = TinyAttention()
    expected = module.qkv_proj.weight.clone()
    monkeypatch.setattr(torch.npu, "Event", mocker.Mock(side_effect=RuntimeError("event allocation failed")))
    with pytest.raises(RuntimeError, match="event allocation failed"):
        adapter_class(module, object(), buckets=4)
    torch.testing.assert_close(module.qkv_proj.weight, expected, rtol=0, atol=0)
    assert "forward" not in module.__dict__


def test_new_dense_call_contract_accepts_vsa_metadata(adapter_class):
    module = TinyAttention()
    adapter = adapter_class(module, object(), buckets=4)
    with pytest.raises(ValueError, match="one packed request"):
        module(
            torch.zeros(2, 7),
            rope_table=None,
            cu_seqlens=torch.tensor([0, 8]),
            max_seqlen=8,
            packed_total=8,
            num_requests=2,
            vsa_prefix_segments=(1,),
            vdn_window=None,
        )
    adapter.close()


def test_vsa_gate_is_rejected_before_mutation(adapter_class):
    module = TinyAttention()
    module.to_gate_compress = nn.Linear(7, 32)
    expected = module.qkv_proj.weight.clone()
    with pytest.raises(ValueError, match="FastH3 VSA"):
        adapter_class(module, object(), buckets=4)
    torch.testing.assert_close(module.qkv_proj.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bounds,valid,total", [([1, 8], 8, 8), ([0, 7], 8, 8), ([0, 4, 8], 5, 8), ([0, 10, 100], 50, 100)]
)
def test_invalid_tnd_boundaries_do_not_launch_collectives(adapter_class, mocker, bounds, valid, total):
    adapter = adapter_class(TinyAttention(), object(), buckets=4)
    run = mocker.patch.object(adapter, "run")
    with pytest.raises(ValueError, match="boundaries"):
        adapter.forward(
            torch.zeros(total // 4, 7),
            rope_table=None,
            cu_seqlens=torch.tensor(bounds),
            max_seqlen=valid,
            packed_total=total,
        )
    run.assert_not_called()


def test_ring_attention_is_rejected_before_collectives(adapter_class, mocker):
    module = TinyAttention()
    module.attention = mocker.Mock(use_ring=True)
    adapter = adapter_class(module, object(), buckets=4)
    run = mocker.patch.object(adapter, "run")
    with pytest.raises(ValueError, match="Ring"):
        adapter.forward(
            torch.zeros(2, 7), rope_table=None, cu_seqlens=torch.tensor([0, 8]), max_seqlen=8, packed_total=8
        )
    run.assert_not_called()


def test_boundary_validation_is_shared_and_invalidated_by_mutation(adapter_class, mocker):
    shared: dict[HeadBucketPlan, HeadWorkspaceCache] = {}
    adapters = [adapter_class(TinyAttention(), object(), buckets=4, workspaces=shared) for _ in range(2)]
    for adapter in adapters:
        mocker.patch.object(adapter, "run")
    bounds = torch.tensor([0, 5, 8])
    read = mocker.spy(bounds, "tolist")
    kwargs = dict(rope_table=None, cu_seqlens=bounds, max_seqlen=5, packed_total=8, vdn_window=None)
    for adapter in adapters:
        adapter.forward(torch.zeros(2, 7), **kwargs)
    assert read.call_count == 1
    bounds[1] = 4
    with pytest.raises(ValueError, match="boundaries"):
        adapters[0].forward(torch.zeros(2, 7), **kwargs)


def test_vdn_attention_is_rejected_before_collectives(adapter_class, mocker):
    adapter = adapter_class(TinyAttention(), object(), buckets=4)
    run = mocker.patch.object(adapter, "run")
    with pytest.raises(ValueError, match="VDN"):
        adapter.forward(
            torch.zeros(2, 7),
            rope_table=None,
            cu_seqlens=torch.tensor([0, 8]),
            max_seqlen=8,
            packed_total=8,
            vdn_window=object(),
        )
    run.assert_not_called()


def test_request_boundary_synchronizes_before_replacing_all_workspaces(monkeypatch, mocker):
    from vllm_omni.diffusion.offloader import distributed_layerwise_backend as backend_module

    caches = [HeadWorkspaceCache(), HeadWorkspaceCache()]
    stream = mocker.Mock(device="npu:0", npu_stream=1)
    allocate = mocker.Mock(side_effect=lambda: mocker.Mock())
    for cache in caches:
        with cache.lease(1, allocate, stream=stream):
            pass
    with pytest.raises(ValueError):
        with caches[0].lease(1, allocate, stream=stream):
            raise ValueError("failed forward")

    def synchronize():
        assert allocate.call_count == 2
        assert all(cache._entries for cache in caches)

    sync = mocker.Mock(side_effect=synchronize)
    monkeypatch.setattr(backend_module.current_omni_platform, "synchronize", sync, raising=False)
    backend = object.__new__(backend_module.DistributedLayerwiseOffloadBackend)
    backend.device = torch.device("privateuseone")
    backend._head_adapter_factory = mocker.Mock(workspaces=dict(enumerate(caches)))
    backend.prepare_for_request()
    sync.assert_called_once()
    for cache in caches:
        with cache.lease(1, allocate, stream=stream):
            pass
    assert allocate.call_count == 4


def test_workspace_reuse_eviction_and_stream_guard(mocker):
    from vllm_omni.diffusion.offloader.submodule.common.head_workspace_cache import HeadWorkspaceCache

    cache = HeadWorkspaceCache(capacity=1)
    stream = mocker.Mock(device="npu:0", npu_stream=1)
    allocate = mocker.Mock(side_effect=lambda: mocker.Mock())
    for key in (1, 1, 2):
        with cache.lease(key, allocate, stream=stream):
            with pytest.raises(RuntimeError, match="already leased"):
                with cache.lease(key, allocate, stream=stream):
                    pass
    assert allocate.call_count == 2
    with pytest.raises(RuntimeError, match="one compute stream"):
        with cache.lease(2, allocate, stream=mocker.Mock(device="npu:0", npu_stream=2)):
            pass
    with pytest.raises(ValueError, match="collective failed"):
        with cache.lease(2, allocate, stream=stream):
            raise ValueError("collective failed")
    with pytest.raises(RuntimeError, match="poisoned"):
        with cache.lease(2, allocate, stream=stream):
            pass
    cache.close_after_synchronize()
    with pytest.raises(RuntimeError, match="closed"):
        with cache.lease(2, allocate, stream=stream):
            pass
