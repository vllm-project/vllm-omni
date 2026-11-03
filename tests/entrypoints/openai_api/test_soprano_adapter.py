# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Soprano API contract: fixed voice, bounds, shared encoding and request errors."""

import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.outputs import CompletionOutput

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    SpeechServingContext,
    TTSGenerationError,
    validate_tts_request,
)
from vllm_omni.entrypoints.openai.tts_adapters.soprano import SopranoAdapter
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_MODEL = "test-local-soprano"


def adapter():
    engine = SimpleNamespace(model=_MODEL, stage_configs=[])
    return SopranoAdapter(SpeechServingContext(server=SimpleNamespace(), engine_client=engine))


def test_explicit_stage_detection_and_capabilities():
    a = adapter()
    assert resolve_adapter("soprano") is SopranoAdapter
    assert detect_tts_model_type("soprano_ar", "SopranoForCausalLM") == "soprano"
    assert detect_tts_model_type(None, "Qwen3ForCausalLM") is None
    assert detect_tts_model_type("soprano_decoder", "SopranoDecoderForGeneration") is None
    assert a.load_capabilities().supported_speakers == {"default"}
    assert a.capabilities.supported_languages == {"English"}


@pytest.mark.parametrize("voice", [None, "default", "DEFAULT"])
def test_default_voice_is_only_the_checkpoint_voice(voice):
    assert adapter().validate(OpenAICreateSpeechRequest(input="Hello.", voice=voice)) is None


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"voice": "alloy"}, "voice"),
        ({"voice": "vivian"}, "voice"),
        ({"voice": ""}, "voice"),
        ({"language": "Auto"}, "English"),
        ({"language": "Chinese"}, "English"),
        ({"ref_audio": "file:///missing.wav"}, "ref_audio"),
        ({"ref_text": "words"}, "ref_text"),
        ({"ref_audio_2": "file:///missing.wav"}, "ref_audio_2"),
        ({"instructions": ""}, "instructions"),
        ({"speaker_embedding": [0.1]}, "speaker_embedding"),
        ({"x_vector_only_mode": False}, "x_vector_only_mode"),
        ({"task_type": "Base"}, "task_type"),
        ({"non_streaming_mode": False}, "non_streaming_mode"),
        ({"ambient_sound": "rain"}, "ambient_sound"),
        ({"duration_seconds": 1}, "duration_seconds"),
        ({"initial_codec_chunk_frames": 1}, "initial_codec_chunk_frames"),
        ({"ref_audio_codes": [[1]], "ref_text": "words"}, "ref_audio_codes"),
        ({"stream": True}, "complete"),
        ({"response_format": "mp3"}, "wav"),
        ({"speed": 1.1}, "speed"),
        ({"word_timestamps": True}, "word_timestamps"),
        ({"max_new_tokens": 513}, "max_new_tokens"),
        ({"seed": -1}, "seed"),
        ({"extra_params": {"ignore_eos": True}}, "extra_params"),
        ({"extra_params": {"temperature": float("nan")}}, "finite"),
        ({"extra_params": {"top_p": 1.1}}, "sampling"),
        ({"extra_params": {"top_k": 1.5}}, "integer"),
        ({"extra_params": {"repetition_penalty": 0}}, "sampling"),
    ],
)
def test_reject_matrix(kwargs, fragment):
    error = validate_tts_request(adapter(), OpenAICreateSpeechRequest(input="Hello.", **kwargs))
    assert error is not None and fragment in error


def test_empty_text_and_valid_sampling_overrides():
    a = adapter()
    assert "empty" in a.validate(OpenAICreateSpeechRequest(input=" \n"))
    r = OpenAICreateSpeechRequest(
        input="Hello.",
        max_new_tokens=8,
        seed=44,
        extra_params={"temperature": 0.3, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.2},
    )
    assert a.validate(r) is None
    original = [SamplingParams(max_tokens=512, repetition_penalty=1.0)]
    changed = a.apply_sampling_overrides(original, r)
    assert changed[0].max_tokens == 8 and changed[0].repetition_penalty == 1.2
    assert original[0].max_tokens == 512 and original[0].repetition_penalty == 1.0


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        assert text.startswith("[STOP][TEXT]") and text.endswith("[START]")
        return [3, 1, 14, 2]


def test_prompt_matches_offline_preparation_and_rejects_multiple_units():
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    from vllm_omni.model_executor.models.soprano.offline import prepare_inputs

    a = adapter()
    a._tokenizer = Tokenizer()
    text = "I have 12 apples."
    prepared = asyncio.run(a.build(OpenAICreateSpeechRequest(input=text), [], False))
    assert prepared.prompt == prepare_inputs(text, a._tokenizer)[0]
    with pytest.raises(ValueError, match="one synthesis unit"):
        a._build_prompt(
            "This sentence is long enough to stand alone. This next sentence is also long enough to stand alone."
        )


def mock_output(failed=False):
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        finish_reason="stop",
        stop_reason=None,
        logprobs=None,
    )
    wave = torch.empty(0) if failed else 0.1 * torch.sin(torch.arange(4096).float() / 20)
    output = OmniRequestOutput(
        request_id="speech-test",
        outputs=[completion],
        finished=True,
        _multimodal_output={
            "audio": wave,
            "sr": torch.tensor(32000),
            "audio_synthesis_error": torch.tensor([failed]),
            "audio_truncated": torch.tensor([False]),
        },
        prompt_token_ids=None,
        encoder_prompt_token_ids=None,
        num_cached_tokens=None,
        prompt_logprobs=None,
        kv_transfer_params=None,
    )
    result = OmniRequestOutput.from_stage_output(output, stage_id=1, final_output_type="audio")
    result.metrics = {"stage_metrics": {"0": {"num_tokens_in": 4, "num_tokens_out": 8}}}
    return result


@pytest.fixture
def server(mocker):
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    engine = mocker.MagicMock()
    engine.errored = False
    engine.model = _MODEL
    engine.model_config = SimpleNamespace(model=_MODEL, async_chunk=False, revision=None, skip_tokenizer_init=True)
    engine.default_sampling_params_list = [
        SamplingParams(max_tokens=512, temperature=0.3, top_p=0.95, top_k=50, repetition_penalty=1.2)
    ]
    engine.stage_configs = [
        SimpleNamespace(
            engine_args=SimpleNamespace(model_stage="soprano_ar", model_arch="SopranoForCausalLM", model=_MODEL),
            tts_args={},
        )
    ]
    engine.tts_batch_max_items = 1

    async def generate(**kwargs):
        yield mock_output(failed=kwargs["sampling_params_list"][0].max_tokens == 1)

    engine.generate = mocker.MagicMock(side_effect=generate)
    models = mocker.MagicMock()
    models.is_base_model.return_value = True
    result = OmniOpenAIServingSpeech(engine_client=engine, models=models, request_logger=None)
    result._adapter._tokenizer = Tokenizer()
    yield result
    result.shutdown()


def test_shared_request_path_preserves_seed_sampling_and_prompt(server):
    request = OpenAICreateSpeechRequest(
        input="Hello.",
        seed=44,
        max_new_tokens=8,
        extra_params={"temperature": 0.4, "top_p": 0.9, "top_k": 20, "repetition_penalty": 1.1},
    )
    asyncio.run(server._prepare_speech_generation(request))
    kwargs = server.engine_client.generate.call_args.kwargs
    assert kwargs["prompt"]["prompt_token_ids"] == [3, 1, 14, 2]
    params = kwargs["sampling_params_list"][0]
    assert (
        params.seed,
        params.max_tokens,
        params.temperature,
        params.top_p,
        params.top_k,
        params.repetition_penalty,
    ) == (44, 8, 0.4, 0.9, 20, 1.1)
    defaults = server.engine_client.default_sampling_params_list[0]
    assert (
        defaults.seed,
        defaults.max_tokens,
        defaults.temperature,
        defaults.top_p,
        defaults.top_k,
        defaults.repetition_penalty,
    ) == (None, 512, 0.3, 0.95, 50, 1.2)
    asyncio.run(server._prepare_speech_generation(OpenAICreateSpeechRequest(input="Hello.")))
    next_params = server.engine_client.generate.call_args.kwargs["sampling_params_list"][0]
    assert (
        next_params.seed,
        next_params.max_tokens,
        next_params.temperature,
        next_params.top_p,
        next_params.top_k,
        next_params.repetition_penalty,
    ) == (None, 512, 0.3, 0.95, 50, 1.2)


def test_shared_encoding_and_error_mapping_leave_next_request_usable(server):
    async def exercise():
        bad = await server.create_speech(OpenAICreateSpeechRequest(input="Hello.", seed=44, max_new_tokens=1))
        assert bad.status_code == 500
        assert b"error" in bad.body and not bad.body.startswith(b"RIFF")
        good = await server.create_speech(OpenAICreateSpeechRequest(input="Hello.", seed=44))
        assert good.status_code == 200 and good.media_type == "audio/wav"
        wave, sr = sf.read(BytesIO(good.body))
        assert sr == 32000 and wave.ndim == 1 and wave.size == 4096

    asyncio.run(exercise())


@pytest.mark.parametrize("stream_format", ["audio", "sse"])
@pytest.mark.parametrize("response_format", ["pcm", "wav"])
def test_streaming_protocols_require_native_async_deployment(mocker, stream_format, response_format):
    a = adapter()
    request = OpenAICreateSpeechRequest(
        input="Hello.", stream=True, stream_format=stream_format, response_format=response_format
    )
    assert "soprano_streaming.yaml" in a.validate(request)
    a.ctx.engine_client.model_config = mocker.Mock(async_chunk=True)
    assert validate_tts_request(a, request) is None


def test_stream_failures_cannot_be_encoded_as_successful_pcm():
    a = adapter()
    with pytest.raises(TTSGenerationError, match="streaming synthesis failed"):
        a.validate_stream_output({"audio_synthesis_error": [torch.tensor([True])]})
    a.validate_stream_output({"audio_synthesis_error": [torch.tensor([False])]})
    with pytest.raises(TTSGenerationError, match="no playable audio"):
        a.validate_stream_audio(has_audio=False)
