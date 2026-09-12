# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    BaseVocoderCUDAGraphRoutine,
    VocoderCUDAGraphComponent,
    VocoderCUDAGraphDescriptor,
    VocoderRuntimeKey,
    VocoderRuntimeResolution,
)
from vllm_omni.worker.vocoder_cudagraph_manager import VocoderCUDAGraphManager

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@dataclass
class _StaticBuffers:
    input: torch.Tensor
    output: torch.Tensor
    descriptor_state: torch.Tensor


class _StaticAllocationRoutine(BaseVocoderCUDAGraphRoutine):
    component_id = "static-allocation"

    @property
    def runnable(self):
        return self.eager_call

    def eager_call(self, value: torch.Tensor) -> torch.Tensor:
        return value.new_full((1,), -1.0)

    def validate_runtime_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("expected one tensor")

    def resolve_runtime(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        available: Set[VocoderCUDAGraphDescriptor],
    ) -> VocoderRuntimeResolution:
        del kwargs
        size = int(args[0].numel())
        descriptor = next((item for item in available if item.variant == size), None)
        return VocoderRuntimeResolution(VocoderRuntimeKey(size), descriptor)

    def allocate_buffers(self, descriptor: VocoderCUDAGraphDescriptor, device: torch.device) -> _StaticBuffers:
        size = int(descriptor.variant)
        return _StaticBuffers(
            input=torch.zeros(1, device=device),
            output=torch.zeros(1, device=device),
            descriptor_state=torch.zeros(size, device=device),
        )

    def prepare_for_capture(self, buffers: object) -> None:
        assert isinstance(buffers, _StaticBuffers)
        buffers.descriptor_state.fill_(float(buffers.descriptor_state.numel()))

    def forward_for_capture(self, buffers: object) -> torch.Tensor:
        assert isinstance(buffers, _StaticBuffers)
        buffers.output.copy_(buffers.descriptor_state[:1] + buffers.input)
        return buffers.output

    def copy_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
    ) -> None:
        del kwargs
        assert isinstance(buffers, _StaticBuffers)
        buffers.input.copy_(args[0].reshape(-1)[:1])

    def output_after_replay(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
        captured_output: object,
    ) -> torch.Tensor:
        del args, kwargs, buffers
        assert isinstance(captured_output, torch.Tensor)
        return captured_output


class _StaticModel:
    supports_vocoder_cudagraph = True
    vocoder_cudagraph_shared_config_keys = frozenset()

    def __init__(self, component: VocoderCUDAGraphComponent) -> None:
        self.component = component

    def get_vocoder_cudagraph_components(self) -> tuple[VocoderCUDAGraphComponent, ...]:
        return (self.component,)


def _vllm_config() -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            vocoder_cudagraph_config={
                "components": {
                    "static-allocation": {
                        "enable_lazy_capture": True,
                        "max_extra_graphs": 1,
                    }
                }
            }
        ),
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=0),
    )


def test_runtime_lazy_capture_preserves_existing_graph_after_static_allocation_growth() -> None:
    device = torch.device("cuda")
    routine = _StaticAllocationRoutine()
    descriptor_a = VocoderCUDAGraphDescriptor(1)
    descriptor_b = VocoderCUDAGraphDescriptor(4096)
    component = VocoderCUDAGraphComponent("static-allocation", routine, [descriptor_a])
    manager = VocoderCUDAGraphManager(vllm_config=_vllm_config(), device=device)

    try:
        manager.prepare(_StaticModel(component))
        manager.capture_and_bind()

        output_a = component(torch.zeros(1, device=device))
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(output_a, torch.ones(1, device=device))

        managed = manager.managed_components["static-allocation"]
        entry_b = manager._runtime_capture_and_register(managed, descriptor_b)
        assert entry_b is not None

        output_b = component(torch.zeros(4096, device=device))
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(output_b, torch.full((1,), 4096.0, device=device))

        output_a_again = component(torch.zeros(1, device=device))
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(output_a_again, torch.ones(1, device=device))
    finally:
        manager.clear()
