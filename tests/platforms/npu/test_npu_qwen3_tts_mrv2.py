# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("vllm_ascend")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen3_tts_state_factory_selects_npu_and_generic_states(monkeypatch):
    import vllm_omni.platforms.npu.models.qwen3_tts_model_state as npu_state
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    calls = []

    def make_npu(*args):
        calls.append(("npu", args))
        return "npu-state"

    def make_omni(*args):
        calls.append(("omni", args))
        return "omni-state"

    monkeypatch.setattr(npu_state, "NPUQwen3TTSTalkerModelState", make_npu)
    # The factory imports OmniModelState lazily, so replace the class at its source.
    import vllm_omni.worker_v2.model_states.omni_model_state as omni_state

    monkeypatch.setattr(omni_state, "OmniModelState", make_omni)

    factory = Qwen3TTSTalkerForConditionalGeneration.create_mrv2_model_state
    npu_result = factory(object(), None, None, torch.device("npu"))
    cuda_result = factory(object(), None, None, torch.device("cuda"))
    assert npu_result == "npu-state"
    assert cuda_result == "omni-state"
    assert [call[0] for call in calls] == ["npu", "omni"]


def test_npu_talker_state_combines_omni_lifecycle_and_ascend_attention():
    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    from vllm_omni.platforms.npu.models.qwen3_tts_model_state import NPUQwen3TTSTalkerModelState
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    assert issubclass(NPUQwen3TTSTalkerModelState, OmniModelState)
    assert issubclass(NPUQwen3TTSTalkerModelState, AscendModelState)
    assert NPUQwen3TTSTalkerModelState.uses_ascend_attn_metadata is True


@pytest.mark.parametrize("use_v2", [False, True])
def test_generation_worker_selects_stage_runner(monkeypatch, use_v2):
    import vllm_omni.platforms.npu.worker.npu_generation_model_runner_v2 as npu_v2_runner
    import vllm_omni.platforms.npu.worker.npu_generation_worker as generation_worker

    class FakeV1Runner:
        def __init__(self, config, device):
            self.config = config
            self.device = device

    class FakeV2Runner(FakeV1Runner):
        pass

    monkeypatch.setattr(generation_worker.NPUGenerationWorker, "model_runner_cls", FakeV1Runner)
    monkeypatch.setattr(npu_v2_runner, "NPUGenerationModelRunnerV2", FakeV2Runner)
    monkeypatch.setattr(generation_worker, "init_workspace_manager", lambda *args: None)
    monkeypatch.setattr(generation_worker.NPUGenerationWorker, "_init_device", lambda self: "npu:0")

    worker = object.__new__(generation_worker.NPUGenerationWorker)
    worker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_v2_model_runner=use_v2))
    worker.use_v2_model_runner = use_v2
    worker.init_device()

    expected_runner = FakeV2Runner if use_v2 else FakeV1Runner
    assert isinstance(worker.model_runner, expected_runner)
    assert worker.model_runner.device == "npu:0"
    assert worker.model_runner.config is worker.vllm_config


def test_npu_generation_runner_mro_preserves_ascend_backend_methods():
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniGenerationModelRunner
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    mro = NPUOmniGenerationModelRunner.__mro__
    assert mro.index(NPUModelRunner) > mro.index(OmniGPUModelRunner)
    assert NPUOmniGenerationModelRunner.load_model is OmniGPUModelRunner.load_model
    assert NPUOmniGenerationModelRunner.capture_model is OmniGPUModelRunner.capture_model
    assert NPUOmniGenerationModelRunner._dummy_run is NPUModelRunner._dummy_run
    assert NPUOmniGenerationModelRunner.initialize_kv_cache.__qualname__.startswith("NPUOmniGenerationModelRunner.")


def test_npu_generation_runner_uses_ascend_input_and_forward_context(monkeypatch):
    from vllm_ascend.worker.v2.input_batch import AscendInputBatch

    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniGenerationModelRunner

    runner = object.__new__(NPUOmniGenerationModelRunner)
    assert callable(runner._close_omni_aux_output)
    runner.input_buffers = object()
    made: dict[str, Any] = {}
    monkeypatch.setattr(
        AscendInputBatch,
        "make_dummy",
        staticmethod(lambda *args, **kwargs: made.update(args=args, kwargs=kwargs) or "ascend-batch"),
    )
    assert (
        NPUOmniGenerationModelRunner._make_dummy_input_batch(runner, 2, 7, max_query_len=4, is_padding=True)
        == "ascend-batch"
    )
    assert made["args"][:3] == (2, 7, runner.input_buffers)
    assert made["kwargs"] == {"max_query_len": 4}

    from vllm_ascend import ascend_forward_context as forward_context_module

    context = object()
    calls: dict[str, Any] = {}

    def fake_context(metadata, config, **kwargs):
        calls.update(metadata=metadata, config=config, kwargs=kwargs)
        return context

    monkeypatch.setattr(forward_context_module, "set_ascend_forward_context", fake_context)
    runner.model = "model-instance"
    result = NPUOmniGenerationModelRunner._forward_context(
        runner,
        None,
        "config",
        num_tokens=7,
        cudagraph_runtime_mode="PIECEWISE",
        slot_mapping=None,
        is_padding=True,
    )
    assert result is context
    assert calls["kwargs"] == {
        "num_tokens": 7,
        "aclgraph_runtime_mode": "PIECEWISE",
        "model_instance": "model-instance",
    }
    with pytest.raises(NotImplementedError, match="slot_mapping"):
        NPUOmniGenerationModelRunner._forward_context(
            runner,
            None,
            "config",
            slot_mapping=object(),
        )
