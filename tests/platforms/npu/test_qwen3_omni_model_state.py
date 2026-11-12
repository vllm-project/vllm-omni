# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3 V2 must preserve Omni lifecycle and build Ascend metadata."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.models.qwen3_omni_model_state import NPUQwen3OmniModelState
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("stage", ["thinker", "talker", "code2wav"])
@pytest.mark.parametrize("npu", [False, True])
def test_factory_preserves_platform_state(monkeypatch, stage, npu):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: npu)
    init = Mock(return_value=None)
    monkeypatch.setattr(OmniModelState, "__init__", init)
    model = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = stage
    config, cache = object(), object()
    device = torch.device("cpu")
    state = init_omni_model_state(config, model, cache, device)
    assert type(state) is (NPUQwen3OmniModelState if npu and stage != "code2wav" else OmniModelState)
    init.assert_called_once_with(config, model, cache, device)


def test_attention_delegates_to_ascend_without_losing_omni_hooks(monkeypatch):
    state = NPUQwen3OmniModelState.__new__(NPUQwen3OmniModelState)
    prepare = Mock(return_value={"layer": object()})
    monkeypatch.setattr(AscendModelState, "prepare_attn", prepare)
    batch = AscendInputBatch.__new__(AscendInputBatch)
    args: tuple = (batch, CUDAGraphMode.PIECEWISE, (), None, [], None)
    result = state.prepare_attn(*args)
    prepare.assert_called_once_with(*args, for_capture=False, ubatch_idx=0, model_specific_attn_metadata=None)
    assert result is prepare.return_value
    assert state.uses_ascend_attn_metadata
    assert NPUQwen3OmniModelState.prepare_inputs is OmniModelState.prepare_inputs
    mro = NPUQwen3OmniModelState.__mro__
    assert mro.index(OmniModelState) < mro.index(AscendModelState)


@pytest.mark.parametrize("profiling", [False, True])
def test_npu_v2_initializes_stage_output_cache_after_native_kv(monkeypatch, profiling):
    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner, NPUOmniV2RunnerMixin

    runner = object.__new__(NPUOmniARModelRunner)
    calls = []
    config = Mock()

    def native_init(self, kv_cache_config, *, is_profiling):
        assert kv_cache_config is config
        calls.append("native")

    def omni_init(*, is_profiling):
        assert calls == ["native"]
        assert is_profiling is profiling
        calls.append("omni")

    monkeypatch.setattr(NPUOmniV2RunnerMixin, "initialize_kv_cache", native_init)
    monkeypatch.setattr(runner, "_init_omni_prefix_cache", omni_init)
    runner.initialize_kv_cache(config, is_profiling=profiling)
    assert calls == ["native", "omni"]


@pytest.mark.parametrize("device_type", ["npu", "cpu"])
def test_minicpmo_dispatches_via_lifecycle_flags(monkeypatch, device_type):
    """MiniCPM-o rides the shared lifecycle-flag dispatch from #10062.

    The model declares ``have_multimodal_outputs`` and ships no
    ``create_omni_model_state`` factory, so ``init_omni_model_state`` picks
    the state class from the device: the Ascend-backed NPU state on NPU,
    the generic Omni state elsewhere. This is the production path (verified
    end-to-end on a 910B1: three-stage serving starts and answers requests)
    and the test keys off the signal the dispatch actually reads
    (``device.type``).
    """
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )
    from vllm_omni.platforms.npu.models.omni_model_state import NPUOmniModelState

    # Single dispatch path by design: the factory hook was dropped in favour
    # of the shared lifecycle-flag branch, and it must not silently return.
    assert not hasattr(MiniCPMO45OmniForConditionalGeneration, "create_omni_model_state")

    init = Mock(return_value=None)
    monkeypatch.setattr(OmniModelState, "__init__", init)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.have_multimodal_outputs = True
    config, cache = object(), object()
    # The dispatch reads only ``device.type``; a fake device keeps the test
    # runnable where torch has no npu backend registered.
    device = SimpleNamespace(type=device_type)
    state = init_omni_model_state(config, model, cache, device)
    expected = NPUOmniModelState if device_type == "npu" else OmniModelState
    assert type(state) is expected
    init.assert_called_once_with(config, model, cache, device)
