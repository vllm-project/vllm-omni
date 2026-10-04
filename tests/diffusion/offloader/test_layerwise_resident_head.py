# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend, current_omni_platform
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


class ResidentPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _encoder_modules: list[str] = []
    _vae_modules: list[str] = []
    _resident_modules: list[str] = []
    _offload_plan = OffloadPlan(block_attrs={"transformer": ("blocks",)}, layerwise_resident_layers={"transformer": 2})

    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.transformer.head_alias = self.transformer.blocks[0].weight

    def forward(self, value):
        for block in self.transformer.blocks:
            value = block(value)
        return value


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}))],
)
def test_resident_head_survives_repeated_requests_and_reenable(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for resident/streamed placement")
    if device == "cpu":
        patch_offload_runtime(monkeypatch, current_omni_platform)
    pipeline = ResidentPipeline().to(device)
    reference = copy.deepcopy(pipeline)
    backend = LayerWiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=device == "cuda"), torch.device(device)
    )
    resident_pointers = [block.weight.data_ptr() for block in pipeline.transformer.blocks[:2]]
    inputs = [torch.randn(2, 4, device=device) for _ in range(4)]
    with torch.inference_mode():
        expected = [reference(value) for value in inputs]
        for _ in range(2):
            backend.enable(pipeline)
            assert len(backend._dit_hooks) == 4
            assert pipeline.transformer.head_alias is pipeline.transformer.blocks[0].weight
            assert not hasattr(pipeline.transformer.blocks[0], "_hook_registry")
            for value, result in zip(inputs, expected, strict=True):
                torch.testing.assert_close(pipeline(value), result, rtol=0, atol=0)
                assert [block.weight.data_ptr() for block in pipeline.transformer.blocks[:2]] == resident_pointers
            backend.disable()
            for actual, original in zip(pipeline.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
            assert all(
                block._hook_registry.get_hook("layerwise_offload") is None for block in pipeline.transformer.blocks[2:]
            )
