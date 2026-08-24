# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config.load import LoadConfig

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.model_loader.pinned_staging import (
    pinned_staging_weights_iterator,
    release_pinned_staging_cache,
)
from vllm_omni.diffusion.models.interface import consumes_borrowed_weight_tensors

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"})
def test_real_pinned_slab_h2d_parity_and_reuse():
    values = [
        torch.arange(4 << 20, dtype=torch.float32),
        torch.arange(4 << 20, dtype=torch.float32) + 7,
    ]
    staged = pinned_staging_weights_iterator(
        iter([("first", values[0]), ("second", values[1])]),
        capacity_bytes=32 << 20,
        min_bytes=1,
    )

    first_name, first = next(staged)
    first_ptr = first.data_ptr()
    assert first_name == "first"
    assert first.is_pinned()
    first_device = first.to("cuda", non_blocking=True)
    torch.accelerator.synchronize()
    assert torch.equal(first_device.cpu(), values[0])

    second_name, second = next(staged)
    assert second_name == "second"
    assert second.is_pinned()
    assert second.data_ptr() == first_ptr
    second_device = second.to("cuda", non_blocking=True)
    torch.accelerator.synchronize()
    assert torch.equal(second_device.cpu(), values[1])

    with pytest.raises(StopIteration):
        next(staged)


@hardware_test(res={"cuda": "L4"})
def test_loader_enabled_path_releases_host_cache(monkeypatch):
    class _CudaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4 << 20, device="cuda"))

        @consumes_borrowed_weight_tensors
        def load_weights(self, weights):
            loaded = set()
            for name, tensor in weights:
                self.weight.data.copy_(tensor)
                loaded.add(name)
            return loaded

    od_config = SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=1),
        quantization_config=None,
        enable_multithread_weight_load=True,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    source = torch.arange(4 << 20, dtype=torch.float32)
    loader.get_all_weights = lambda _model: iter([("weight", source)])  # type: ignore[method-assign]
    model = _CudaModel()
    monkeypatch.setenv("VLLM_OMNI_ENABLE_PINNED_WEIGHT_STAGING", "1")

    release_pinned_staging_cache()
    before = torch.cuda.host_memory_stats()["allocated_bytes.current"]
    loader.load_weights(model)
    after = torch.cuda.host_memory_stats()["allocated_bytes.current"]

    assert torch.equal(model.weight.cpu(), source)
    assert after <= before
