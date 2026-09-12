# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Ascend VoxCPM2 LocDiT NPUGraph adapter."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.models import voxcpm2_talker as npu_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeEstimator(torch.nn.Module):
    def forward(self, x, mu, t, cond, dt):
        return x + mu + t + cond + dt


class VoxCPM2TalkerForConditionalGeneration:
    def __init__(self) -> None:
        self.estimator = FakeEstimator()
        self.tts = SimpleNamespace(feat_decoder=SimpleNamespace(estimator=self.estimator))


class FakeGraphRunner:
    instances = []
    supported = True

    def __init__(self, *, max_graphs, component_name, disable_config_hint) -> None:
        self.max_graphs = max_graphs
        self.component_name = component_name
        self.disable_config_hint = disable_config_hint
        self.calls = []
        self.__class__.instances.append(self)

    @classmethod
    def is_supported(cls) -> bool:
        return cls.supported

    def run(self, operation, inputs, constants, compute):
        self.calls.append((operation, inputs, constants))
        return compute(*inputs)


def test_loc_dit_npugraph_is_enabled_by_default_and_wraps_once(monkeypatch) -> None:
    FakeGraphRunner.instances.clear()
    FakeGraphRunner.supported = True
    monkeypatch.setattr(npu_adapter, "NPUExactGraphRunner", FakeGraphRunner)
    model = VoxCPM2TalkerForConditionalGeneration()

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(model)
    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(model)

    inputs = tuple(torch.full((2, 3), value) for value in range(5))
    output = model.estimator(*inputs)
    runner = FakeGraphRunner.instances[0]
    assert torch.equal(output, sum(inputs[1:], start=inputs[0]))
    assert runner.max_graphs == 8
    assert runner.component_name == "VoxCPM2 LocDiT"
    assert runner.calls == [("forward", inputs, ())]
    assert len(FakeGraphRunner.instances) == 1


def test_loc_dit_npugraph_falls_back_when_apis_are_unavailable(monkeypatch) -> None:
    FakeGraphRunner.instances.clear()
    FakeGraphRunner.supported = False
    monkeypatch.setattr(npu_adapter, "NPUExactGraphRunner", FakeGraphRunner)
    model = VoxCPM2TalkerForConditionalGeneration()

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(model)

    assert not hasattr(model.estimator, "_voxcpm2_npu_graph_runner")
    FakeGraphRunner.supported = True
