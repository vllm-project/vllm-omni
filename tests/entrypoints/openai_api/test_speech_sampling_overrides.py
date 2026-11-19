# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech sampling override validation, precedence and first-stage propagation."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_top_level_sampling_overrides_win_without_mutating_extra_params():
    extras: dict[str, object] = {"temperature": 0.9, "repetition_penalty": 1.05, "subtalker_sampling_params": {}}
    request = OpenAICreateSpeechRequest(input="hello", temperature=0, repetition_penalty=1, extra_params=extras)
    assert request.extra_params == {"temperature": 0.0, "repetition_penalty": 1.0, "subtalker_sampling_params": {}}
    assert extras == {"temperature": 0.9, "repetition_penalty": 1.05, "subtalker_sampling_params": {}}


@pytest.mark.parametrize("overrides", [{"temperature": float("nan")}, {"repetition_penalty": 0}])
def test_extra_params_reject_invalid_sampling_numbers(overrides):
    with pytest.raises(ValidationError, match="extra_params"):
        OpenAICreateSpeechRequest(input="hello", extra_params=overrides)


def test_null_overrides_preserve_nested_values_and_empty_subtalker_policy():
    extras: dict[str, object] = {"temperature": 0.3, "repetition_penalty": None, "subtalker_sampling_params": {}}
    request = OpenAICreateSpeechRequest(input="hello", temperature=None, repetition_penalty=None, extra_params=extras)
    assert request.extra_params == extras
    assert OpenAICreateSpeechRequest(input="hello").extra_params is None


@pytest.mark.parametrize("repetition_penalty", [1.0, 1.1])
@pytest.mark.parametrize("placement", ["top-level", "extra_params"])
def test_speech_sampling_overrides_reach_first_stage(repetition_penalty, placement):
    from vllm import SamplingParams

    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
    from vllm_omni.entrypoints.openai.tts_adapters.base import PreparedRequest

    requested = {"temperature": 0.7, "top_p": 0.8, "top_k": 17, "repetition_penalty": repetition_penalty}
    payload = {"input": "Hello", "extra_params": requested}
    if placement == "top-level":
        payload = {
            "input": "Hello",
            "temperature": requested["temperature"],
            "repetition_penalty": repetition_penalty,
            "extra_params": {"top_p": requested["top_p"], "top_k": requested["top_k"]},
        }
    request = OpenAICreateSpeechRequest.model_validate(payload)

    async def build(request, sampling_params_list, has_inline_ref_audio):
        return PreparedRequest(prompt={"prompt": request.input}, model_type="higgs_audio_v3")

    adapter = SimpleNamespace(
        normalize=lambda request: None,
        validate=lambda request: None,
        build=build,
        apply_sampling_overrides=lambda params, request, prompt, request_id: params,
    )
    defaults = SamplingParams(temperature=1.0, top_p=0.95, top_k=50, repetition_penalty=1.05)
    engine = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[defaults],
        generate=lambda **kwargs: kwargs["sampling_params_list"],
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.engine_client = engine
    serving.model_config = SimpleNamespace(async_chunk=True)
    serving._tts_model_type = "higgs_audio_v3"
    serving._get_tts_adapter = lambda: adapter
    serving._refresh_uploaded_speakers = lambda: None
    serving._validate_speech_sample_rate = lambda request: None
    serving._log_request = lambda *args: None
    serving._speech_output_policies = {}

    _, stage_sampling_params, _ = asyncio.run(serving._prepare_speech_generation(request, request_id="speech-test"))
    stage0_params = stage_sampling_params[0]
    assert stage0_params is not defaults
    for name, value in requested.items():
        assert getattr(stage0_params, name) == value
    assert stage0_params.extra_args == requested
    assert (defaults.temperature, defaults.top_p, defaults.top_k, defaults.repetition_penalty) == (1.0, 0.95, 50, 1.05)
    assert defaults.extra_args is None
