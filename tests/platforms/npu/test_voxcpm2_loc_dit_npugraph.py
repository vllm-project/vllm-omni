# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Ascend VoxCPM2 LocDiT NPUGraph adapter."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.models import voxcpm2_talker as npu_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeEstimator(torch.nn.Module):
    def forward(self, x, mu, t, cond, dt):
        return x + mu + t + cond + dt


class FakeTalker:
    def __init__(self, *, vllm_config=None, prefix="") -> None:
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.estimator = FakeEstimator()
        self.tts = SimpleNamespace(feat_decoder=SimpleNamespace(estimator=self.estimator))


class FakeTalkerSubclass(FakeTalker):
    pass


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


def test_model_patch_wraps_init_and_is_idempotent(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(npu_adapter, "setup_voxcpm2_loc_dit_npu_graph", calls.append)
    monkeypatch.setattr(npu_adapter, "_PATCHED", False)
    monkeypatch.setattr(npu_adapter, "_original_init", None)

    original_init = FakeTalker.__init__
    fake_module = SimpleNamespace(VoxCPM2TalkerForConditionalGeneration=FakeTalker)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker",
        fake_module,
    )

    try:
        npu_adapter.apply_voxcpm2_talker_patch()
        npu_adapter.apply_voxcpm2_talker_patch()
        model = FakeTalker(vllm_config="config", prefix="talker")
    finally:
        FakeTalker.__init__ = original_init

    assert calls == [model]
    assert model.vllm_config == "config"
    assert model.prefix == "talker"


def test_npu_platform_registers_patch_and_keeps_worker_cls(monkeypatch) -> None:
    pytest.importorskip(
        "vllm_ascend",
        reason="NPU platform registration requires the optional vllm-ascend plugin",
    )
    from vllm_ascend import utils as ascend_utils

    from vllm_omni.platforms.npu import _310p
    from vllm_omni.platforms.npu.models import (
        minicpmo_4_5_code2wav,
        qwen3_tts,
        qwen3_tts_tokenizer_v2,
    )
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    calls = []
    monkeypatch.setattr(ascend_utils, "adapt_patch", lambda **kwargs: None)
    monkeypatch.setattr(_310p, "apply_patches", lambda: None)
    monkeypatch.setattr(
        minicpmo_4_5_code2wav,
        "apply_minicpmo_4_5_code2wav_patch",
        lambda: None,
    )
    monkeypatch.setattr(qwen3_tts, "apply_qwen3_tts_patches", lambda: None)
    monkeypatch.setattr(
        qwen3_tts_tokenizer_v2,
        "apply_qwen3_tts_tokenizer_v2_patch",
        lambda: None,
    )
    monkeypatch.setattr(
        npu_adapter,
        "apply_voxcpm2_talker_patch",
        lambda: calls.append("voxcpm2"),
    )

    NPUOmniPlatform()
    worker_cls = NPUOmniPlatform.get_omni_ar_worker_cls()

    assert worker_cls == "vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker"
    assert calls == ["voxcpm2"]


def test_loc_dit_npugraph_supports_wrapped_subclass_and_wraps_once(monkeypatch) -> None:
    FakeGraphRunner.instances.clear()
    FakeGraphRunner.supported = True
    monkeypatch.setattr(
        npu_adapter,
        "_get_npu_exact_graph_runner_cls",
        lambda: FakeGraphRunner,
    )
    model = FakeTalkerSubclass()
    wrapped_model = SimpleNamespace(module=model)

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(wrapped_model)
    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(wrapped_model)

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
    monkeypatch.setattr(
        npu_adapter,
        "_get_npu_exact_graph_runner_cls",
        lambda: FakeGraphRunner,
    )
    model = FakeTalker()

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(model)

    assert not hasattr(model.estimator, "_voxcpm2_npu_graph_runner")
    FakeGraphRunner.supported = True
