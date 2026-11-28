# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LoRA buffers must live on the manager's compute device wherever the base weights are at wrap time.

``lora_a_stacked`` / ``lora_b_stacked`` are plain tensors, so CPU offload never moves them. With a
startup ``lora_path`` the layers are wrapped before the first forward, while offload still keeps the
base weights off the compute device, and vLLM would place the buffers next to those weights.
"""

from __future__ import annotations

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 16
# ``meta`` stands in for an accelerator: a device the real CPU base weights are not on.
COMPUTE = torch.device("meta")


@pytest.fixture(autouse=True)
def _cpu_linear_env(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


class _Pipeline(torch.nn.Module):
    """Pipeline whose base weights sit on CPU, as under CPU offload before the first forward."""

    def __init__(self):
        super().__init__()
        vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
        self.transformer = torch.nn.Module()
        with set_current_vllm_config(vllm_config):
            self.transformer.proj = ReplicatedLinear(
                HIDDEN, 24, bias=False, params_dtype=torch.float32, disable_tp=True, prefix="proj"
            )
            self.transformer.qkv = QKVParallelLinear(
                HIDDEN, 4, 2, bias=False, params_dtype=torch.float32, disable_tp=True, prefix="qkv"
            )


def _peft(rank: int) -> PEFTHelper:
    return PEFTHelper(r=rank, lora_alpha=rank, target_modules=["proj", "qkv"])


def _lora_tensors(wrapper) -> list[torch.Tensor]:
    return [*wrapper.lora_a_stacked, *wrapper.lora_b_stacked]


def test_buffers_are_placed_on_the_compute_device_when_base_weights_are_off_device():
    pipeline = _Pipeline()
    manager = DiffusionLoRAManager(pipeline=pipeline, device=COMPUTE, dtype=torch.float32)

    manager._replace_layers_with_lora(_peft(4))

    assert set(manager._lora_modules) == {"transformer.proj", "transformer.qkv"}
    for wrapper in manager._lora_modules.values():
        assert all(t.device == COMPUTE for t in _lora_tensors(wrapper))
        assert wrapper.base_layer.weight.device.type == "cpu"


def test_rank_growth_reallocates_buffers_on_the_compute_device():
    pipeline = _Pipeline()
    manager = DiffusionLoRAManager(pipeline=pipeline, device=COMPUTE, dtype=torch.float32)
    manager._replace_layers_with_lora(_peft(4))

    manager._ensure_max_lora_rank(32)

    for wrapper in manager._lora_modules.values():
        assert all(t.shape[2] == 32 or t.shape[3] == 32 for t in _lora_tensors(wrapper))
        assert all(t.device == COMPUTE for t in _lora_tensors(wrapper))
