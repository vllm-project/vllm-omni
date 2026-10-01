# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for head packing and the H3 adapter's DLO lifecycle."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.submodule.common.head_buckets import HeadBucketPlan

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
    pytest.importorskip("torch_npu")
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
        )
    adapter.close()


def test_vsa_gate_is_rejected_before_mutation(adapter_class):
    module = TinyAttention()
    module.to_gate_compress = nn.Linear(7, 32)
    expected = module.qkv_proj.weight.clone()
    with pytest.raises(ValueError, match="FastH3 VSA"):
        adapter_class(module, object(), buckets=4)
    torch.testing.assert_close(module.qkv_proj.weight, expected, rtol=0, atol=0)


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
