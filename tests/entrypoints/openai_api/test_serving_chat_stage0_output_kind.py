# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which output kind each MiniCPM-o 4.5 stage gets on an audio chat stream.

A client that asked for ``stream=true`` should get a stream. Stage 0's kind also
decides what TTFT measures: under ``FINAL_ONLY`` the stage emits exactly one
output, at the end, so ``serving_time_to_first_output_ms`` -- the age of a
stage's first non-empty output -- reports the time to the thinker's *last*
token.

See ``OmniOpenAIServingChat._fix_minicpmo45_audio_stream_output_kinds``.
"""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ARCH = "MiniCPMO45OmniForConditionalGeneration"


def _stage(model_stage: str, arch: str = _ARCH):
    return SimpleNamespace(engine_args=SimpleNamespace(model_arch=arch, model_stage=model_stage))


def _service(stages):
    service = OmniOpenAIServingChat.__new__(OmniOpenAIServingChat)
    service.engine_client = SimpleNamespace(stage_configs=stages)
    return service


def _params(n: int = 3):
    return [SamplingParams(max_tokens=8) for _ in range(n)]


def _fix(service, params, modalities=("text", "audio"), request=None):
    return service._fix_minicpmo45_audio_stream_output_kinds(params, modalities, request)


@pytest.fixture
def service():
    return _service([_stage("llm"), _stage("tts"), _stage("token2wav")])


def test_thinker_streams_on_a_simplex_audio_request(service) -> None:
    params = _params()

    _fix(service, params)

    assert params[0].output_kind == RequestOutputKind.DELTA
    # The talker streams either way -- that half was never in question.
    assert params[1].output_kind == RequestOutputKind.DELTA


def test_duplex_session_keeps_the_thinker_final_only(service) -> None:
    # A duplex response is a streaming session: the orchestrator forwards stage 0
    # on segment boundaries and derives `is_final_update` from this output kind.
    params = _params()

    _fix(service, params, request=SimpleNamespace(omni_duplex_session=True))

    assert params[0].output_kind == RequestOutputKind.FINAL_ONLY
    assert params[1].output_kind == RequestOutputKind.DELTA


def test_a_request_without_the_duplex_marker_still_streams(service) -> None:
    params = _params()

    _fix(service, params, request=SimpleNamespace())

    assert params[0].output_kind == RequestOutputKind.DELTA


def test_text_only_request_is_untouched(service) -> None:
    params = _params()
    before = [p.output_kind for p in params]

    _fix(service, params, modalities=("text",))

    assert [p.output_kind for p in params] == before


def test_other_architectures_are_untouched() -> None:
    service = _service([_stage("llm", arch="Qwen3OmniMoeForConditionalGeneration")])
    params = _params(1)
    before = [p.output_kind for p in params]

    _fix(service, params)

    assert [p.output_kind for p in params] == before
