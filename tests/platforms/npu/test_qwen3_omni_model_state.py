# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3 V2 must preserve Omni lifecycle and build Ascend metadata."""

from functools import partial
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


class _KvInitRecorder:
    """Record the native → stage-cache ordering without module-level state."""

    def __init__(self, config: object, profiling: bool, spec: object = None) -> None:
        self.config = config
        self.profiling = profiling
        self.spec = spec
        self.calls: list[str] = []
        self.widths_seen_during_native: list[int] = []

    def native(self, kv_cache_config, *, is_profiling) -> None:
        assert kv_cache_config is self.config
        if self.spec is not None:
            widths = self.spec.num_speculative_tokens_per_batch_size or []
            self.widths_seen_during_native.append(len(widths))
        self.calls.append("native")

    def omni(self, *, is_profiling) -> None:
        assert self.calls == ["native"]
        assert is_profiling is self.profiling
        self.calls.append("omni")


def _recording_native_init(runner, kv_cache_config, *args, **kwargs) -> None:
    """Stand-in for ``NPUOmniV2RunnerMixin.initialize_kv_cache``.

    Patched onto the mixin, so it receives ``self`` and must stay a plain
    function: a bound method installed on a class is not re-bound per instance.
    The recorder travels on the instance, so tests never share mutable state.
    """
    runner._kv_recorder.native(kv_cache_config, is_profiling=kwargs.get("is_profiling", False))


def _recording_omni_init(recorder: _KvInitRecorder, *, is_profiling: bool) -> None:
    """Stand-in for ``_init_omni_prefix_cache``, patched onto the instance."""
    recorder.omni(is_profiling=is_profiling)


def _install_recorder(monkeypatch, runner, recorder: _KvInitRecorder) -> None:
    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniV2RunnerMixin

    runner._kv_recorder = recorder
    monkeypatch.setattr(NPUOmniV2RunnerMixin, "initialize_kv_cache", _recording_native_init)
    monkeypatch.setattr(runner, "_init_omni_prefix_cache", partial(_recording_omni_init, recorder))


class _SpecConfig:
    """Minimal speculative_config for the scheduler-Ngram narrowing branch."""

    def __init__(self, dynamic: bool) -> None:
        self._dynamic = dynamic
        self.num_speculative_tokens_per_batch_size = None

    def uses_dynamic_speculative_decoding(self) -> bool:
        return self._dynamic


@pytest.mark.parametrize("profiling", [False, True])
def test_npu_v2_initializes_stage_output_cache_after_native_kv(monkeypatch, profiling):
    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner

    runner = object.__new__(NPUOmniARModelRunner)
    recorder = _KvInitRecorder(Mock(), profiling)
    _install_recorder(monkeypatch, runner, recorder)
    runner.initialize_kv_cache(recorder.config, is_profiling=profiling)
    assert recorder.calls == ["native", "omni"]


def test_npu_v2_reads_speculative_config_only_when_scheduler_ngram_armed(monkeypatch):
    """``speculative_config`` must be read behind ``_scheduler_ngram``.

    ``_scheduler_ngram`` is a class attribute defaulting to False and is only set
    once ``__init__`` has run. Reading ``speculative_config`` first raises
    AttributeError for any runner built without ``__init__`` — which is exactly
    how the test above builds one.
    """
    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner

    runner = object.__new__(NPUOmniARModelRunner)
    assert NPUOmniARModelRunner._scheduler_ngram is False
    assert not hasattr(runner, "speculative_config")
    recorder = _KvInitRecorder(Mock(), True)
    _install_recorder(monkeypatch, runner, recorder)
    runner.initialize_kv_cache(recorder.config, is_profiling=True)
    assert recorder.calls == ["native", "omni"]


@pytest.mark.parametrize("dynamic", [False, True])
def test_npu_v2_narrows_speculator_only_for_static_ngram(monkeypatch, dynamic):
    """An armed scheduler-Ngram run narrows widths during native KV init only."""
    from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner

    spec = _SpecConfig(dynamic=dynamic)
    runner = object.__new__(NPUOmniARModelRunner)
    runner._scheduler_ngram = True
    runner.speculative_config = spec
    runner.num_speculative_steps = 4
    recorder = _KvInitRecorder(Mock(), True, spec=spec)
    _install_recorder(monkeypatch, runner, recorder)
    runner.initialize_kv_cache(recorder.config, is_profiling=True)
    assert recorder.calls == ["native", "omni"]
    # Dynamic speculation is left alone; a static scheduler-Ngram run is narrowed
    # while the native cache initializes and restored afterwards.
    assert recorder.widths_seen_during_native == ([0] if dynamic else [2])
    assert spec.num_speculative_tokens_per_batch_size is None
