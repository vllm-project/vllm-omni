# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech sampling override validation, precedence and first-stage propagation."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest, StreamingSpeechSessionConfig

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


@pytest.mark.parametrize(
    ("name", "value", "shadowed_value"),
    [("temperature", 0.7, -1), ("repetition_penalty", 1.1, 0)],
)
def test_top_level_overrides_shadow_invalid_nested_values(name, value, shadowed_value):
    extras = {name: shadowed_value}
    request = OpenAICreateSpeechRequest.model_validate({"input": "hello", name: value, "extra_params": extras})
    assert request.extra_params == {name: value}
    assert extras == {name: shadowed_value}


@pytest.fixture
def speech_sampling_server(mocker):
    from vllm import SamplingParams
    from vllm.sampling_params import SamplingType

    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
    from vllm_omni.entrypoints.openai.tts_adapters.base import PreparedRequest

    async def build(request, sampling_params_list, has_inline_ref_audio):
        return PreparedRequest(prompt={"prompt": request.input}, model_type="higgs_audio_v3")

    adapter = SimpleNamespace(
        normalize=lambda request: None,
        validate=lambda request: None,
        build=mocker.AsyncMock(side_effect=build),
        apply_sampling_overrides=lambda params, request, prompt, request_id: params,
    )
    defaults = SamplingParams(temperature=1.0, top_p=0.95, top_k=50, repetition_penalty=1.05)
    # Defaults may already have a cached sampling mode before a request arrives.
    assert defaults.sampling_type == SamplingType.RANDOM
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
    return serving


@pytest.mark.parametrize("repetition_penalty", [1.0, 1.1])
@pytest.mark.parametrize("placement", ["top-level", "extra_params"])
def test_speech_sampling_overrides_reach_first_stage(speech_sampling_server, repetition_penalty, placement):
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
    serving = speech_sampling_server
    defaults = serving.engine_client.default_sampling_params_list[0]

    _, stage_sampling_params, _ = asyncio.run(serving._prepare_speech_generation(request, request_id="speech-test"))
    stage0_params = stage_sampling_params[0]
    assert stage0_params is not defaults
    for name, value in requested.items():
        assert getattr(stage0_params, name) == value
    assert stage0_params.extra_args == requested
    assert (defaults.temperature, defaults.top_p, defaults.top_k, defaults.repetition_penalty) == (1.0, 0.95, 50, 1.05)
    assert defaults.extra_args is None


@pytest.mark.parametrize("placement", ["top-level", "extra_params"])
@pytest.mark.parametrize(
    ("temperature", "effective_temperature", "do_sample"),
    [(0.0, 0.0, False), (1e-6, 0.01, True), (2.0, 2.0, True)],
)
def test_temperature_overrides_preserve_vllm_sampling_mode(
    speech_sampling_server, placement, temperature, effective_temperature, do_sample
):
    from vllm.sampling_params import SamplingType

    from vllm_omni.model_executor.models.qwen3_tts.sampling_policy import get_mtp_sampling_overrides

    overrides = {"temperature": temperature}
    payload: dict[str, object] = {**overrides} if placement == "top-level" else {"extra_params": overrides}
    request = OpenAICreateSpeechRequest.model_validate({"input": "hello", **payload})
    original = request.model_dump()
    _, params, _ = asyncio.run(speech_sampling_server._prepare_speech_generation(request, request_id="speech-test"))
    assert params[0].temperature == effective_temperature
    assert params[0].sampling_type == (SamplingType.RANDOM if do_sample else SamplingType.GREEDY)
    prepared_request, prepared_params, _ = speech_sampling_server._get_tts_adapter().build.call_args.args
    assert prepared_request.extra_params["temperature"] == effective_temperature
    assert prepared_params[0].temperature == effective_temperature
    if placement == "top-level":
        assert prepared_request.temperature == effective_temperature
    assert params[0].extra_args["temperature"] == effective_temperature
    assert request.model_dump() == original
    assert get_mtp_sampling_overrides(params[0]) == {"do_sample": do_sample, "temperature": effective_temperature}
    defaults = speech_sampling_server.engine_client.default_sampling_params_list[0]
    assert defaults.temperature == 1.0
    assert defaults.sampling_type == SamplingType.RANDOM
    assert defaults.extra_args is None


@pytest.mark.parametrize("placement", ["top-level", "extra_params"])
def test_temperature_overrides_reject_values_above_vllm_limit(speech_sampling_server, placement):
    overrides = {"temperature": 2.1}
    payload: dict[str, object] = {**overrides} if placement == "top-level" else {"extra_params": overrides}
    request = OpenAICreateSpeechRequest.model_validate({"input": "hello", **payload})
    with pytest.raises(ValueError, match="temperature"):
        asyncio.run(speech_sampling_server._prepare_speech_generation(request, request_id="speech-test"))
    speech_sampling_server._get_tts_adapter().build.assert_not_awaited()


@pytest.mark.parametrize("seed", [None, 42])
def test_sampling_overrides_preserve_server_owned_seed_marker(speech_sampling_server, seed):
    extras = {"temperature": 1e-6, "omni_user_seed": True}
    request = OpenAICreateSpeechRequest(input="hello", seed=seed, extra_params=extras)
    _, params, _ = asyncio.run(speech_sampling_server._prepare_speech_generation(request, request_id="speech-test"))

    assert params[0].temperature == 0.01
    assert params[0].extra_args["temperature"] == 0.01
    if seed is None:
        assert "omni_user_seed" not in params[0].extra_args
    else:
        assert params[0].extra_args["omni_user_seed"] is True
        assert params[0].extra_args["tts_local_seed"] == seed
        assert params[0].seed == seed
    assert extras == {"temperature": 1e-6, "omni_user_seed": True}
    assert speech_sampling_server.engine_client.default_sampling_params_list[0].extra_args is None


@pytest.mark.parametrize(
    "options",
    [
        {
            "temperature": 1e-6,
            "repetition_penalty": 1.1,
            "extra_params": {"temperature": 0.9, "repetition_penalty": 0},
        },
        {"extra_params": {"temperature": 1e-6, "repetition_penalty": 1.1}},
    ],
)
def test_websocket_sampling_overrides_reach_first_stage(speech_sampling_server, options):
    from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
    from vllm_omni.model_executor.models.qwen3_tts.sampling_policy import get_mtp_sampling_overrides

    config = StreamingSpeechSessionConfig(**options)
    original = config.model_dump()
    for text in ("Hello.", "Again."):
        request = OmniStreamingSpeechHandler._speech_request(config, text)
        assert request.input == text
        _, params, _ = asyncio.run(speech_sampling_server._prepare_speech_generation(request, request_id="speech-test"))
        assert params[0].repetition_penalty == 1.1
        assert params[0].temperature == 0.01
        assert get_mtp_sampling_overrides(params[0]) == {"do_sample": True, "temperature": 0.01}
    assert config.model_dump() == original


@pytest.mark.parametrize("options", [{"temperature": -1}, {"repetition_penalty": 0}])
def test_websocket_rejects_invalid_sampling_fields(options):
    with pytest.raises(ValidationError):
        StreamingSpeechSessionConfig(**options)
