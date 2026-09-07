# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MiniCPM-o 4.5 audio-streaming output-kind coercion.

Chat ``stream=true`` coerces every AR stage to DELTA. For MiniCPM-o 4.5 that
hands ``llm2tts`` one thinker token per step while the forwarded hidden states
still span prompt + generated, so the talker is conditioned on a broadcast
mismatch (issue #5298). The thinker must stay FINAL_ONLY; only the talker
streams.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MINICPMO45_ARCH = "MiniCPMO45OmniForConditionalGeneration"


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    return object.__new__(OmniOpenAIServingChat)


def _stage(model_arch: str, model_stage: str) -> SimpleNamespace:
    # SimpleNamespace, not MagicMock: _stage_get prefers .get() when present,
    # and a MagicMock satisfies hasattr(obj, "get") and returns a mock.
    return SimpleNamespace(engine_args=SimpleNamespace(model_arch=model_arch, model_stage=model_stage))


def _minicpmo45_stages():
    return [_stage(MINICPMO45_ARCH, "llm"), _stage(MINICPMO45_ARCH, "tts")]


def _delta_params(n: int = 2):
    return [SamplingParams(output_kind=RequestOutputKind.DELTA) for _ in range(n)]


def test_thinker_forced_final_only_talker_stays_delta(serving_chat):
    serving_chat.engine_client = SimpleNamespace(stage_configs=_minicpmo45_stages())

    out = serving_chat._fix_minicpmo45_audio_stream_output_kinds(_delta_params(), ["text", "audio"])

    assert out[0].output_kind is RequestOutputKind.FINAL_ONLY
    assert out[1].output_kind is RequestOutputKind.DELTA


@pytest.mark.parametrize("modalities", [None, [], ["text"]])
def test_non_audio_request_is_untouched(serving_chat, modalities):
    serving_chat.engine_client = SimpleNamespace(stage_configs=_minicpmo45_stages())

    out = serving_chat._fix_minicpmo45_audio_stream_output_kinds(_delta_params(), modalities)

    assert [p.output_kind for p in out] == [RequestOutputKind.DELTA] * 2


def test_other_architectures_are_untouched(serving_chat):
    stages = [_stage("SomeOtherOmniForConditionalGeneration", "llm")]
    serving_chat.engine_client = SimpleNamespace(stage_configs=stages)

    out = serving_chat._fix_minicpmo45_audio_stream_output_kinds(_delta_params(1), ["text", "audio"])

    assert out[0].output_kind is RequestOutputKind.DELTA


def test_shorter_sampling_params_list_does_not_raise(serving_chat):
    serving_chat.engine_client = SimpleNamespace(stage_configs=_minicpmo45_stages())

    out = serving_chat._fix_minicpmo45_audio_stream_output_kinds(_delta_params(1), ["text", "audio"])

    assert len(out) == 1
    assert out[0].output_kind is RequestOutputKind.FINAL_ONLY
