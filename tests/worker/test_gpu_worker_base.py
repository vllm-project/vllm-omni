# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.outputs import StagePostWarmupMemoryStats
from vllm_omni.worker.base import OmniGPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_stage_post_warmup_memory_stats_returns_native_rpc_payload():
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    model = SimpleNamespace(
        get_stage_post_warmup_memory_stats=lambda: StagePostWarmupMemoryStats(
            allocated_bytes=11,
            reserved_bytes=13,
        )
    )
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    assert worker.get_stage_post_warmup_memory_stats() == {
        "allocated_bytes": 11,
        "reserved_bytes": 13,
    }


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(),
        SimpleNamespace(get_stage_post_warmup_memory_stats=lambda: None),
    ],
)
def test_get_stage_post_warmup_memory_stats_omits_unsupported_model(model):
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    assert worker.get_stage_post_warmup_memory_stats() is None


def test_get_stage_post_warmup_memory_stats_omits_nonzero_rank():
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 1
    worker.model_runner = SimpleNamespace(get_model=lambda: pytest.fail("non-reporting rank must not read the model"))

    assert worker.get_stage_post_warmup_memory_stats() is None


class _QuantMethod:
    def __init__(self, scale):
        self.scale = scale


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))
        self.register_buffer("inv_freq", torch.arange(4.0))
        self.register_buffer("kv_scale", torch.full((2,), 3.0))
        self.cos_sin_cache = torch.full((3,), 5.0)
        self.weight_view = self.weight.data[0]
        self.quant_method = _QuantMethod(torch.full((2,), 7.0))


def _allocation(tensor, tag):
    storage = tensor.untyped_storage()
    return storage.data_ptr(), SimpleNamespace(tag=tag, handle=(0, storage.nbytes(), storage.data_ptr(), 0))


def _sleeping_worker(monkeypatch, layer, allocator):
    import vllm.device_allocator

    import vllm_omni.worker.base as base

    monkeypatch.setattr(vllm.device_allocator, "get_mem_allocator_instance", lambda: allocator)
    monkeypatch.setattr(
        base,
        "current_omni_platform",
        SimpleNamespace(synchronize=lambda: None, get_current_memory_usage=lambda device: 0),
    )
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    worker.device = "cpu"
    worker.model_runner = SimpleNamespace(model=layer, get_draft_model=lambda: None)
    return worker


def _discard(layer):
    for tensor in (layer.weight.data, layer.inv_freq, layer.kv_scale, layer.cos_sin_cache, layer.quant_method.scale):
        tensor.zero_()


def test_level2_sleep_restores_non_parameter_tensors_of_the_weights_pool_on_weights_wake(monkeypatch):
    layer = _Layer()
    weights_pool = [layer.weight, layer.inv_freq, layer.cos_sin_cache, layer.quant_method.scale]
    allocator = SimpleNamespace(
        pointer_to_data=dict(
            [_allocation(t, "weights") for t in weights_pool] + [_allocation(layer.kv_scale, "kv_cache")]
        ),
        sleep=lambda offload_tags: None,
        wake_up=lambda tags: None,
    )
    worker = _sleeping_worker(monkeypatch, layer, allocator)

    worker.sleep(level=2)
    _discard(layer)
    worker.wake_up(tags=["kv_cache"])
    assert torch.equal(layer.inv_freq, torch.zeros(4)), "a kv_cache-only wake must not restore weights-pool tensors"

    worker.wake_up(tags=["weights"])
    assert torch.equal(layer.inv_freq, torch.arange(4.0))
    assert torch.equal(layer.cos_sin_cache, torch.full((3,), 5.0))
    assert torch.equal(layer.quant_method.scale, torch.full((2,), 7.0))
    # Outside the weights pool, or a parameter (also through a view): the caller's weight sync owns them.
    assert torch.equal(layer.kv_scale, torch.zeros(2))
    assert torch.equal(layer.weight.data, torch.zeros(4, 4))
    assert worker._sleep_saved_tensors == []


def test_level2_sleep_without_pool_map_saves_only_buffers(monkeypatch):
    layer = _Layer()
    allocator = SimpleNamespace(sleep=lambda offload_tags: None, wake_up=lambda tags: None)
    worker = _sleeping_worker(monkeypatch, layer, allocator)

    worker.sleep(level=2)
    _discard(layer)
    worker.wake_up(tags=None)

    assert torch.equal(layer.inv_freq, torch.arange(4.0))
    assert torch.equal(layer.kv_scale, torch.full((2,), 3.0))
    assert torch.equal(layer.cos_sin_cache, torch.zeros(3))


def test_level1_sleep_saves_nothing(monkeypatch):
    layer = _Layer()
    allocator = SimpleNamespace(
        pointer_to_data=dict([_allocation(layer.inv_freq, "weights")]),
        sleep=lambda offload_tags: None,
        wake_up=lambda tags: None,
    )
    worker = _sleeping_worker(monkeypatch, layer, allocator)

    worker.sleep(level=1)

    assert getattr(worker, "_sleep_saved_tensors", None) is None
