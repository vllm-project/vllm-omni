# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, Request, UploadFile
from fastapi.testclient import TestClient
from vllm.config import SpeechToTextConfig
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.openai import serving_transcription as transcription_api
from vllm_omni.entrypoints.openai.serving_transcription import OmniOpenAIServingTranscription, TextOnlyEngineClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(**kwargs):
    return TranscriptionRequest(file=UploadFile(io.BytesIO(b"audio"), filename="test.wav"), **kwargs)


def _text_result(text="language English<asr_text>Hello world", *, finish_reason="stop"):
    return SimpleNamespace(
        error=None,
        final_output_type="text",
        prompt_token_ids=[11, 12],
        outputs=[SimpleNamespace(text=text, token_ids=[21, 22, 23], finish_reason=finish_reason, stop_reason=None)],
    )


def _alignment_result(pairs):
    return SimpleNamespace(
        error=None,
        final_output_type="timestamps",
        outputs=SimpleNamespace(data=np.asarray(pairs, dtype=np.int32)),
    )


async def _results(*items):
    for item in items:
        if isinstance(item, BaseException):
            raise item
        yield item


class _StreamingPostProcessor:
    def process_delta(self, text, finished):
        return text


@pytest.fixture
def service(mocker):
    """Exercise the real request path without loading a tokenizer or model."""
    from vllm_omni.model_executor.models.qwen3_asr.qwen3_asr import Qwen3ASRForConditionalGeneration

    serving = object.__new__(OmniOpenAIServingTranscription)
    serving.has_asr_aligner = True
    serving.asr_config = SpeechToTextConfig(sample_rate=1000, overlap_chunk_second=0)
    serving.model_config = SimpleNamespace(
        max_model_len=4096, is_encoder_decoder=False, hf_config=SimpleNamespace(model_type="qwen3_asr")
    )
    serving.model_cls = SimpleNamespace(
        validate_language=mocker.Mock(side_effect=lambda language: language),
        get_generation_prompt=mocker.Mock(return_value={"prompt_token_ids": [101, 102]}),
        get_num_audio_tokens=mocker.Mock(return_value=None),
        post_process_output=mocker.Mock(side_effect=Qwen3ASRForConditionalGeneration.post_process_output),
        get_transcription_language=mocker.Mock(side_effect=Qwen3ASRForConditionalGeneration.get_transcription_language),
        no_space_languages={"zh", "ja"},
        supported_languages={"en": "English", "zh": "Chinese", "ja": "Japanese"},
    )
    serving.streaming_post_processor_cls = _StreamingPostProcessor
    serving.enable_force_include_usage = False
    serving.task_type = "transcribe"
    serving.default_sampling_params = {"max_tokens": 128}
    serving.models = SimpleNamespace(model_name=mocker.Mock(return_value="Qwen/Qwen3-ASR-1.7B"))
    serving.engine_client = SimpleNamespace(
        generate=mocker.Mock(side_effect=lambda *args, **kwargs: _results(_text_result())),
        abort=mocker.AsyncMock(),
    )
    serving.renderer = SimpleNamespace(
        render_cmpl_async=mocker.AsyncMock(return_value=[{"type": "token", "prompt_token_ids": [101, 102]}])
    )
    serving._check_model = mocker.AsyncMock(return_value=None)
    serving._preflight = mocker.Mock()
    serving._base_request_id = mocker.Mock(return_value="unit-test")
    serving._maybe_get_adapters = mocker.Mock(return_value=None)
    serving._get_trace_headers = mocker.AsyncMock(return_value={"traceparent": "trace"})
    serving._log_inputs = mocker.Mock()
    serving._decode_and_chunk_speech_async = mocker.AsyncMock(return_value=([np.zeros(1500, dtype=np.float32)], 1.5))
    serving._detect_language = mocker.AsyncMock(return_value="en")
    return serving


def test_text_only_client_delegates_state_and_preserves_explicit_modalities(mocker):
    engine = SimpleNamespace(
        model_config=object(), generate=mocker.Mock(return_value=object()), abort=mocker.AsyncMock()
    )
    client = TextOnlyEngineClient(engine)

    assert client.model_config is engine.model_config
    assert client.abort is engine.abort
    assert client.generate("prompt", "params", "id") is engine.generate.return_value
    assert engine.generate.call_args.kwargs == {"output_modalities": ["text"]}

    client.generate("prompt", "params", "id", output_modalities=["text", "timestamps"], trace_headers={"a": "b"})
    assert engine.generate.call_args.kwargs == {
        "output_modalities": ["text", "timestamps"],
        "trace_headers": {"a": "b"},
    }


def test_constructor_wraps_native_requests_without_loading_extra_models(mocker):
    engine = SimpleNamespace()
    models = object()

    def initialize(instance, engine_client, serving_models, **kwargs):
        instance.engine_client = engine_client
        assert serving_models is models

    parent = mocker.patch.object(OpenAIServingTranscription, "__init__", autospec=True, side_effect=initialize)

    serving = OmniOpenAIServingTranscription(engine, models, has_asr_aligner=True, request_logger=None)

    assert isinstance(serving.engine_client, TextOnlyEngineClient)
    assert serving.engine_client._engine_client is engine
    assert serving.has_asr_aligner
    assert parent.call_args.kwargs == {"request_logger": None}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("aligner_enabled", "request_kwargs"),
    [
        (False, {"response_format": "verbose_json", "timestamp_granularities[]": ["word"]}),
        (True, {"response_format": "json"}),
        (True, {"response_format": "verbose_json", "timestamp_granularities[]": ["segment"]}),
        (True, {"response_format": "diarized_json"}),
        (True, {"use_beam_search": True}),
    ],
)
async def test_non_alignment_requests_preserve_native_model_behavior(service, mocker, aligner_enabled, request_kwargs):
    service.model_config.hf_config.model_type = "another_asr"
    service.has_asr_aligner = aligner_enabled
    expected = object()
    parent = mocker.patch.object(
        OpenAIServingTranscription, "create_transcription", autospec=True, return_value=expected
    )
    request = _request(**request_kwargs)

    result = await service.create_transcription(b"native audio", request)

    assert result is expected
    parent.assert_awaited_once_with(service, b"native audio", request, None)
    service._decode_and_chunk_speech_async.assert_not_awaited()
    service.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["json", "text"])
async def test_generic_word_timestamps_require_verbose_json(service, response_format):
    service.model_config.hf_config.model_type = "another_asr"

    result = await service.create_transcription(
        b"audio", _request(response_format=response_format, **{"timestamp_granularities[]": ["word"]})
    )

    assert isinstance(result, ErrorResponse)
    assert "requires response_format=verbose_json" in result.error.message
    service._decode_and_chunk_speech_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"timestamp_granularities[]": ["word", "segment"]}, "word timestamps only"),
        ({"stream": True}, "do not support streaming"),
        ({"use_beam_search": True}, "Beam search"),
    ],
)
async def test_generic_forced_alignment_rejects_unsupported_options(service, request_kwargs, message):
    service.model_config.hf_config.model_type = "another_asr"
    request = _request(**{"response_format": "verbose_json", "timestamp_granularities[]": ["word"], **request_kwargs})

    result = await service.create_transcription(b"audio", request)

    assert isinstance(result, ErrorResponse)
    assert message in result.error.message
    service._decode_and_chunk_speech_async.assert_not_awaited()
    service.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["json", "text"])
async def test_plain_transcription_stops_at_text_stage(service, response_format):
    request = _request(
        response_format=response_format, language="en", prompt="meeting context", max_completion_tokens=23
    )
    raw_request = Request({"type": "http", "headers": [(b"traceparent", b"trace")]})

    result = await service.create_transcription(b"encoded audio", request, raw_request)

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Hello world"
    assert result.usage.seconds == 2
    assert request.model == "Qwen/Qwen3-ASR-1.7B"
    service._decode_and_chunk_speech_async.assert_awaited_once_with(b"encoded audio")
    service.engine_client.generate.assert_called_once()
    prompt, params, request_id = service.engine_client.generate.call_args.args
    assert request_id == "transcribe-unit-test"
    assert prompt["prompt_token_ids"] == [101, 102]
    assert prompt["type"] == "token"
    service.renderer.render_cmpl_async.assert_awaited_once_with([{"prompt_token_ids": [101, 102]}])
    waveform, sample_rate = prompt["multi_modal_data"]["audio"]
    assert waveform is service._decode_and_chunk_speech_async.return_value[0][0]
    assert sample_rate == 1000
    assert prompt["additional_information"]["language"] == ["en"]
    assert params.max_tokens == 23
    assert params.output_kind == RequestOutputKind.FINAL_ONLY
    assert service.engine_client.generate.call_args.kwargs == {
        "output_modalities": ["text"],
        "lora_request": None,
        "trace_headers": {"traceparent": "trace"},
    }
    stt_params = service.model_cls.get_generation_prompt.call_args.args[0]
    assert stt_params.request_prompt == "meeting context"
    assert stt_params.audio is waveform
    assert raw_request.state.request_metadata.final_usage_info.total_tokens == 5
    service.engine_client.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_verbose_transcription_decodes_words_and_detected_language(service):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result("language English<asr_text>Hello world"),
        _alignment_result([[120, 500], [600, 1400]]),
    )
    request = _request(response_format="verbose_json", **{"timestamp_granularities[]": ["word"]})

    result = await service.create_transcription(b"audio", request)

    assert isinstance(result, TranscriptionResponseVerbose)
    assert result.text == "Hello world"
    assert result.language == "English"
    assert result.duration == 1.5
    assert [word.model_dump() for word in result.words] == [
        {"word": "Hello", "start": 0.12, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.4},
    ]
    assert service.engine_client.generate.call_args.kwargs["output_modalities"] == ["text", "timestamps"]
    assert "additional_information" not in service.engine_client.generate.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_prompt",
    [
        "native text prompt",
        [31, 32],
        {"prompt": "native text prompt", "multi_modal_data": {"audio": "native audio format"}},
        {"prompt_token_ids": [31, 32], "multi_modal_data": {"audio": "native audio format"}},
    ],
)
async def test_generic_alignment_renders_native_prompt_and_budgets_expanded_input(service, native_prompt):
    service.model_config.hf_config.model_type = "another_asr"
    service.model_config.max_model_len = 7
    service.model_cls.get_generation_prompt.return_value = native_prompt
    service.model_cls.post_process_output.side_effect = lambda text: text.removeprefix("RAW:").strip()
    features = object()
    engine_input = {"type": "multimodal", "prompt_token_ids": [1, 2, 3, 4, 5], "mm_kwargs": features}
    service.renderer.render_cmpl_async.return_value = [engine_input]
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result("RAW: Hello world"), _alignment_result([[0, 400], [500, 1000]])
    )

    result = await service.create_transcription(
        b"audio", _request(response_format="verbose_json", language="en", **{"timestamp_granularities[]": ["word"]})
    )

    assert result.text == "Hello world"
    assert [word.word for word in result.words] == ["Hello", "world"]
    assert service.engine_client.generate.call_args.kwargs["output_modalities"] == ["text", "timestamps"]
    service.model_cls.post_process_output.assert_called_once_with("RAW: Hello world")
    normalized = service.renderer.render_cmpl_async.call_args.args[0][0]
    if isinstance(native_prompt, str):
        assert normalized == {"prompt": native_prompt}
    elif isinstance(native_prompt, list):
        assert normalized == {"prompt_token_ids": native_prompt}
    else:
        assert normalized is native_prompt
    prompt, params, _ = service.engine_client.generate.call_args.args
    assert prompt["type"] == "multimodal"
    assert prompt["mm_kwargs"] is features
    assert params.max_tokens == 2
    assert prompt["multi_modal_data"]["audio"][0] is service._decode_and_chunk_speech_async.return_value[0][0]
    assert "multi_modal_data" not in engine_input


@pytest.mark.asyncio
async def test_encoder_decoder_alignment_keeps_native_layout_and_counts_only_decoder_prefix(service):
    service.model_config.hf_config.model_type = "whisper"
    service.model_config.is_encoder_decoder = True
    service.model_config.max_model_len = 8
    service.model_cls.post_process_output.side_effect = lambda text: text.strip()
    native_prompt = {
        "encoder_prompt": {"prompt": "", "multi_modal_data": {"audio": "native encoder audio"}},
        "decoder_prompt": "native decoder prefix",
    }
    service.model_cls.get_generation_prompt.return_value = native_prompt
    encoder_input = {"type": "token", "prompt_token_ids": list(range(2000))}
    decoder_input = {"type": "multimodal", "prompt_token_ids": [1, 2, 3], "mm_kwargs": object()}
    engine_input = {"type": "enc_dec", "encoder_prompt": encoder_input, "decoder_prompt": decoder_input}
    service.renderer.render_cmpl_async.return_value = [engine_input]
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result("Hello world"), _alignment_result([[0, 400], [500, 1000]])
    )

    result = await service.create_transcription(
        b"audio", _request(response_format="verbose_json", language="en", **{"timestamp_granularities[]": ["word"]})
    )

    assert result.text == "Hello world"
    normalized = service.renderer.render_cmpl_async.call_args.args[0][0]
    assert normalized["encoder_prompt"] is native_prompt["encoder_prompt"]
    assert normalized["decoder_prompt"] == {"prompt": "native decoder prefix"}
    prompt, params, _ = service.engine_client.generate.call_args.args
    assert prompt["encoder_prompt"] is encoder_input
    assert prompt["decoder_prompt"] is decoder_input
    assert params.max_tokens == 5
    assert prompt["additional_information"]["language"] == ["en"]
    assert "multi_modal_data" not in engine_input


@pytest.mark.asyncio
async def test_native_explicit_language_detection_runs_before_alignment_prompt_construction(service):
    service.model_config.hf_config.model_type = "another_asr"
    service.model_cls.supports_explicit_language_detection = True
    service.model_cls.post_process_output.side_effect = lambda text: text
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result("Hello world"), _alignment_result([[0, 400], [500, 1000]])
    )

    result = await service.create_transcription(
        b"audio", _request(response_format="verbose_json", **{"timestamp_granularities[]": ["word"]})
    )

    assert result.language == "en"
    service._detect_language.assert_awaited_once_with(
        service._decode_and_chunk_speech_async.return_value[0][0], "transcribe-unit-test-lang_detect"
    )
    assert service.model_cls.get_generation_prompt.call_args.args[0].language == "en"
    assert service.engine_client.generate.call_args.args[0]["additional_information"]["language"] == ["en"]


@pytest.mark.asyncio
async def test_out_of_order_chunks_keep_transcript_order_and_absolute_word_times(service):
    chunks = [np.zeros(2000, dtype=np.float32), np.zeros(1000, dtype=np.float32)]
    service._decode_and_chunk_speech_async.return_value = (chunks, 3.0)
    second_complete = asyncio.Event()
    completion_order = []

    async def first_chunk():
        await second_complete.wait()
        completion_order.append(0)
        yield _text_result("language English<asr_text>first")
        yield _alignment_result([[250, 1000]])

    async def second_chunk():
        completion_order.append(1)
        # Pooling output may arrive before this chunk's final text output.
        yield _alignment_result([[100, 1400]])
        yield _text_result("language English<asr_text>second")
        second_complete.set()

    service.engine_client.generate.side_effect = [first_chunk(), second_chunk()]

    result = await service.create_transcription(b"audio", _request(response_format="verbose_json"))

    assert completion_order == [1, 0]
    assert result.text == "first second"
    assert [word.model_dump() for word in result.words] == [
        {"word": "first", "start": 0.25, "end": 1.0},
        {"word": "second", "start": 2.1, "end": 3.0},
    ]
    assert [call.args[2] for call in service.engine_client.generate.call_args_list] == [
        "transcribe-unit-test-0",
        "transcribe-unit-test-1",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_family", ["qwen", "ming"])
async def test_auto_detected_chinese_does_not_insert_spaces_between_chunks(service, model_family):
    prefix = "language Chinese<asr_text>"
    if model_family == "ming":
        _use_native_ming_processing(service)
        prefix = "Chinese\t"
    service._decode_and_chunk_speech_async.return_value = (
        [np.zeros(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)],
        2.0,
    )
    service.engine_client.generate.side_effect = [
        _results(_text_result(prefix + text), _alignment_result([[0, 300], [400, 800]])) for text in ["你好", "世界"]
    ]

    result = await service.create_transcription(
        b"audio", _request(response_format="verbose_json", **{"timestamp_granularities[]": ["word"]})
    )

    assert result.text == "你好世界"
    assert result.language == "Chinese"
    assert [word.word for word in result.words] == ["你", "好", "世", "界"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "transcripts", "language", "expected"),
    [
        ("verbose_json", ["", "Hello", "", "world", ""], "English", "Hello world"),
        ("text", ["", "你", "", "好", ""], "Chinese", "你好"),
        ("json", ["", "", ""], "None", ""),
    ],
)
async def test_non_streaming_empty_chunks_preserve_text_and_word_offsets(
    service, response_format, transcripts, language, expected
):
    service._decode_and_chunk_speech_async.return_value = (
        [np.zeros(1000, dtype=np.float32) for _ in transcripts],
        float(len(transcripts)),
    )
    sources = []
    for transcript in transcripts:
        outputs = [_text_result(f"language {language}<asr_text>{transcript}")]
        if response_format == "verbose_json":
            outputs.append(_alignment_result([[100, 500]] if transcript else []))
        sources.append(_results(*outputs))
    service.engine_client.generate.side_effect = sources

    result = await service.create_transcription(b"audio", _request(response_format=response_format))

    assert result.text == expected
    if response_format == "verbose_json":
        assert [(word.start, word.end) for word in result.words] == [
            (index + 0.1, index + 0.5) for index, transcript in enumerate(transcripts) if transcript
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", ["", "..."])
async def test_transcript_without_words_returns_empty_word_list(service, transcript):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result(f"language None<asr_text>{transcript}"), _alignment_result([])
    )

    result = await service.create_transcription(b"audio", _request(response_format="verbose_json"))

    assert result.text == transcript
    assert result.language == ""
    assert result.words == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "aligner_enabled", "message"),
    [
        ({"response_format": "srt"}, True, "supports json, text and verbose_json"),
        ({"use_beam_search": True}, True, "Beam search"),
        ({"timestamp_granularities[]": ["word"]}, True, "requires response_format=verbose_json"),
        (
            {"response_format": "verbose_json", "timestamp_granularities[]": ["segment"]},
            True,
            "word timestamps only",
        ),
        ({"response_format": "verbose_json", "stream": True}, True, "do not support streaming"),
        ({"response_format": "verbose_json"}, False, "ASR stage connected to a forced aligner"),
    ],
)
async def test_unsupported_requests_are_rejected_before_audio_decode(service, request_kwargs, aligner_enabled, message):
    service.has_asr_aligner = aligner_enabled

    result = await service.create_transcription(b"audio", _request(**request_kwargs))

    assert isinstance(result, ErrorResponse)
    assert result.error.code == 400
    assert message in result.error.message
    service._decode_and_chunk_speech_async.assert_not_awaited()
    service.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_model_is_rejected_without_preflight_or_audio_decode(service):
    expected = service.create_error_response("unknown model")
    service._check_model.return_value = expected

    result = await service.create_transcription(b"audio", _request(model="other"))

    assert result is expected
    service._preflight.assert_not_called()
    service._decode_and_chunk_speech_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [np.zeros(0, dtype=np.float32)]])
async def test_empty_audio_is_rejected(service, chunks):
    service._decode_and_chunk_speech_async.return_value = (chunks, 0.0)

    result = await service.create_transcription(b"empty", _request())

    assert isinstance(result, ErrorResponse)
    assert "audio samples" in result.error.message
    service.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_plain_transcription_returns_sse_without_running_aligner(service):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result("Hello ", finish_reason=None), _text_result("world")
    )

    stream = await service.create_transcription(b"audio", _request(stream=True))
    events = [event async for event in stream]

    assert events[-1] == "data: [DONE]\n\n"
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks) == "Hello world"
    assert service.engine_client.generate.call_args.kwargs["output_modalities"] == ["text"]
    assert service.engine_client.generate.call_args.args[1].output_kind == RequestOutputKind.DELTA


def _use_native_qwen_streaming_processor(service):
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRStreamingPostProcessor

    service.streaming_post_processor_cls = Qwen3ASRStreamingPostProcessor
    return Qwen3ASRStreamingPostProcessor


def _use_native_ming_processing(service):
    from vllm_omni.model_executor.models.ming3_audio.ming3_audio import (
        BailingMoeV3AudioForConditionalGeneration,
        Ming3AudioStreamingPostProcessor,
    )

    service.model_config.hf_config.model_type = "bailing_moe_v3_audio"
    service.model_cls.post_process_output.side_effect = BailingMoeV3AudioForConditionalGeneration.post_process_output
    service.model_cls.get_transcription_language.side_effect = (
        BailingMoeV3AudioForConditionalGeneration.get_transcription_language
    )
    service.model_cls.supported_languages = BailingMoeV3AudioForConditionalGeneration.supported_languages
    service.model_cls.no_space_languages = BailingMoeV3AudioForConditionalGeneration.no_space_languages
    service.streaming_post_processor_cls = Ming3AudioStreamingPostProcessor
    return Ming3AudioStreamingPostProcessor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repeat_prompt_ids", "audio_tokens", "include_usage", "continuous_usage", "force_usage"),
    [
        (False, None, False, False, False),
        (True, None, True, False, False),
        (True, 17, True, True, False),
        (False, None, False, True, True),
    ],
)
async def test_streaming_usage_accumulates_prompt_tokens_once_per_chunk(
    service, repeat_prompt_ids, audio_tokens, include_usage, continuous_usage, force_usage
):
    _use_native_qwen_streaming_processor(service)
    service.enable_force_include_usage = force_usage
    service.model_cls.get_num_audio_tokens.return_value = audio_tokens
    chunks = [
        ["language English<asr_", "text>Hello"],
        ["language None<asr_text>"],
        ["language English<asr_text>wo", "rld"],
    ]
    sources = []
    for prompt_count, deltas in zip([2, 3, 4], chunks):
        outputs = []
        for index, delta in enumerate(deltas):
            result = _text_result(delta, finish_reason="stop" if index == len(deltas) - 1 else None)
            result.prompt_token_ids = list(range(prompt_count)) if index == 0 or repeat_prompt_ids else None
            outputs.append(result)
        sources.append(_results(*outputs))
    metadata = SimpleNamespace(final_usage_info=None)
    request = _request(
        model=service.models.model_name(),
        stream=True,
        stream_include_usage=include_usage,
        stream_continuous_usage_stats=continuous_usage,
    )

    events = [
        event
        async for event in service.transcription_stream_generator(request, sources, "usage-stream", metadata, 3.0, " ")
    ]

    assert events[-1] == "data: [DONE]\n\n"
    responses = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    expected = {
        "prompt_tokens": 9 + (audio_tokens or 0),
        "completion_tokens": 15,
        "total_tokens": 24 + (audio_tokens or 0),
    }
    assert {field: getattr(metadata.final_usage_info, field) for field in expected} == expected
    if include_usage or force_usage:
        final = responses.pop()
        assert final["choices"] == []
        assert final["usage"] == expected
    assert "".join(response["choices"][0]["delta"]["content"] for response in responses) == "Hello world"
    if continuous_usage:
        assert [response["usage"]["prompt_tokens"] for response in responses] == [
            count + (audio_tokens or 0) for count in [2, 5, 9, 9]
        ]
        assert [response["usage"]["completion_tokens"] for response in responses] == [6, 9, 12, 15]
        assert all(
            response["usage"]["total_tokens"]
            == response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
            for response in responses
        )
    else:
        assert all(response.get("usage") is None for response in responses)
    service.engine_client.abort.assert_not_awaited()


async def _collect_chunked_transcription_stream(service, chunks, *, language=None, factory_snapshots=None):
    async def chunk_source(deltas):
        deltas = deltas or [""]
        for index, delta in enumerate(deltas):
            # Both concurrent requests must run before either stream completes.
            await asyncio.sleep(0)
            if factory_snapshots is not None:
                factory_snapshots.append(service.streaming_post_processor_cls)
            yield _text_result(delta, finish_reason="stop" if index == len(deltas) - 1 else None)

    request = _request(stream=True, language=language, model=service.models.model_name())
    separator = "" if language in service.model_cls.no_space_languages else " "
    stream = service.transcription_stream_generator(
        request,
        [chunk_source(deltas) for deltas in chunks],
        "chunked-stream",
        SimpleNamespace(final_usage_info=None),
        float(len(chunks)),
        separator,
    )
    events = [event async for event in stream]
    assert events[-1] == "data: [DONE]\n\n"
    responses = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert all(response["object"] == "transcription.chunk" for response in responses)
    return "".join(choice["delta"].get("content") or "" for response in responses for choice in response["choices"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (
            [["la", "nguage C", "hinese<asr_", "text>你", "好"], ["language Chinese<asr_text>世", "界"]],
            "你好世界",
        ),
        (
            [["la", "nguage Japanese<asr_te", "xt>こん", "にちは"], ["language Japanese<asr_text>世", "界"]],
            "こんにちは世界",
        ),
        (
            [["la", "nguage En", "glish<asr_text>He", "llo"], ["language English<asr_text>wo", "rld"]],
            "Hello world",
        ),
    ],
    ids=["auto-chinese", "auto-japanese", "auto-english"],
)
async def test_qwen_streaming_joins_chunks_after_native_split_prefix_decoding(service, chunks, expected):
    original_factory = _use_native_qwen_streaming_processor(service)

    text = await _collect_chunked_transcription_stream(service, chunks)

    assert text == expected
    assert service.streaming_post_processor_cls is original_factory
    service.engine_client.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_qwen_streaming_empty_first_middle_and_last_chunks_add_no_spaces(service):
    _use_native_qwen_streaming_processor(service)
    chunks = [
        ["language None<asr_text>"],
        ["language English<asr_text>He", "llo"],
        [],
        ["language English<asr_text>wo", "rld"],
        ["language English<asr_text>"],
    ]

    text = await _collect_chunked_transcription_stream(service, chunks)

    assert text == "Hello world"


@pytest.mark.asyncio
async def test_qwen_streaming_explicit_language_controls_chunk_separator(service):
    _use_native_qwen_streaming_processor(service)

    text = await _collect_chunked_transcription_stream(
        service, [["language Chinese<asr_text>你", "好"], ["language Chinese<asr_text>世", "界"]], language="en"
    )

    assert text == "你好 世界"
    service.model_cls.post_process_output.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_qwen_streams_keep_joining_state_and_original_factory_separate(service):
    original_factory = _use_native_qwen_streaming_processor(service)
    snapshots: list[type] = []

    chinese, english = await asyncio.gather(
        _collect_chunked_transcription_stream(
            service,
            [["la", "nguage Chinese<asr_text>你", "好"], ["language Chinese<asr_text>世界"]],
            factory_snapshots=snapshots,
        ),
        _collect_chunked_transcription_stream(
            service,
            [["language English<asr_", "text>He", "llo"], ["language English<asr_text>wo", "rld"]],
            factory_snapshots=snapshots,
        ),
    )

    assert chinese == "你好世界"
    assert english == "Hello world"
    assert snapshots and all(factory is original_factory for factory in snapshots)
    assert service.streaming_post_processor_cls is original_factory
    service.engine_client.abort.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("language", "expected"), [(None, "你好世界"), ("en", "你好 世界")])
async def test_ming_streaming_uses_language_hook_after_split_prefix_decoding(service, language, expected):
    original_factory = _use_native_ming_processing(service)

    text = await _collect_chunked_transcription_stream(
        service, [["Chi", "nese\t你", "好"], ["Chinese\t世", "界"]], language=language
    )

    assert text == expected
    assert service.model_cls.supported_languages == {}
    if language is None:
        service.model_cls.get_transcription_language.assert_called_once_with("Chinese\t你")
    else:
        service.model_cls.get_transcription_language.assert_not_called()
    service.model_cls.post_process_output.assert_not_called()
    assert service.streaming_post_processor_cls is original_factory


@pytest.mark.asyncio
async def test_generic_streaming_keeps_model_postprocessor_before_chunk_joining(service):
    instances = []

    class ModelPostProcessor:
        def __init__(self):
            self.calls = []
            instances.append(self)

        def process_delta(self, text, finished):
            self.calls.append((text, finished))
            return text.removeprefix("<native>").upper()

    service.model_config.hf_config.model_type = "another_asr"
    del service.model_cls.get_transcription_language
    service.streaming_post_processor_cls = ModelPostProcessor

    text = await _collect_chunked_transcription_stream(service, [["<native>a", "b"], ["<native>c", "d"]])

    assert text == "AB CD"
    assert [processor.calls for processor in instances] == [
        [("<native>a", False), ("b", True)],
        [("<native>c", False), ("d", True)],
    ]
    service.model_cls.post_process_output.assert_not_called()
    assert service.streaming_post_processor_cls is ModelPostProcessor


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "close"])
async def test_stream_termination_after_return_aborts_all_chunks_and_closes_sources(service, termination):
    service._decode_and_chunk_speech_async.return_value = (
        [np.zeros(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)],
        2.0,
    )
    waiting = asyncio.Event()
    closed = asyncio.Event()
    never_complete = asyncio.Event()

    async def pending_chunk():
        try:
            yield _text_result("Hello ", finish_reason=None)
            waiting.set()
            await never_complete.wait()
        finally:
            closed.set()

    sources = [pending_chunk(), _results(_text_result("world"))]
    service.engine_client.generate.side_effect = sources
    stream = await service.create_transcription(b"audio", _request(stream=True))
    assert "Hello" in await anext(stream)

    if termination == "close":
        await stream.aclose()
    else:
        next_event = asyncio.create_task(anext(stream))
        await waiting.wait()
        next_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_event

    assert closed.is_set()
    assert all(source.ag_frame is None for source in sources)
    service.engine_client.abort.assert_awaited_once_with(["transcribe-unit-test-0", "transcribe-unit-test-1"])


@pytest.mark.asyncio
async def test_streaming_engine_error_sends_error_event_and_aborts(service):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        SimpleNamespace(error="stream stage failed")
    )

    stream = await service.create_transcription(b"audio", _request(stream=True))
    events = [event async for event in stream]

    assert events[-1] == "data: [DONE]\n\n"
    error = json.loads(events[0].removeprefix("data: "))
    assert "stream stage failed" in error["error"]["message"]
    service.engine_client.abort.assert_awaited_once_with(["transcribe-unit-test"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([SimpleNamespace(error="stage failed")], "stage failed"),
        ([_text_result(finish_reason="error")], "generation failed"),
        ([], "did not return a transcription"),
    ],
)
async def test_generation_errors_propagate_and_abort_request(service, items, message):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(*items)

    with pytest.raises(RuntimeError, match=message):
        await service.create_transcription(b"audio", _request())

    service.engine_client.abort.assert_awaited_once_with(["transcribe-unit-test"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("alignment", "message"),
    [(None, "did not return word timestamps"), (_alignment_result([[0, 100]]), "did not match")],
)
async def test_missing_or_mismatched_alignment_is_an_error(service, alignment, message):
    items = [_text_result()]
    if alignment is not None:
        items.append(alignment)
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(*items)

    with pytest.raises(RuntimeError, match=message):
        await service.create_transcription(b"audio", _request(response_format="verbose_json"))

    service.engine_client.abort.assert_awaited_once_with(["transcribe-unit-test"])


@pytest.mark.asyncio
async def test_engine_validation_error_returns_400_and_aborts_request(service):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(ValueError("input too long"))

    result = await service.create_transcription(b"audio", _request())

    assert isinstance(result, ErrorResponse)
    assert result.error.code == 400
    assert "input too long" in result.error.message
    service.engine_client.abort.assert_awaited_once_with(["transcribe-unit-test"])


@pytest.mark.asyncio
async def test_cancellation_aborts_all_registered_chunk_requests(service):
    service._decode_and_chunk_speech_async.return_value = (
        [np.zeros(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)],
        2.0,
    )
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await service.create_transcription(b"audio", _request())

    service.engine_client.abort.assert_awaited_once_with(["transcribe-unit-test-0", "transcribe-unit-test-1"])


@pytest.fixture
def client(service):
    app = FastAPI()
    app.state.openai_serving_transcription = service
    app.include_router(transcription_api.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("response_format", ["json", "text"])
def test_multipart_transcription_route_returns_requested_format(client, service, response_format):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"encoded wav data", "audio/wav")},
        data={"response_format": response_format},
    )

    assert response.status_code == 200
    service._decode_and_chunk_speech_async.assert_awaited_once_with(b"encoded wav data")
    if response_format == "text":
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "Hello world"
    else:
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["text"] == "Hello world"


def test_multipart_verbose_route_parses_word_granularity(client, service):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(
        _text_result(), _alignment_result([[100, 500], [600, 1000]])
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"audio", "audio/wav")},
        data={"response_format": "verbose_json", "timestamp_granularities[]": "word"},
    )

    assert response.status_code == 200
    assert response.json()["words"] == [
        {"word": "Hello", "start": 0.1, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]
    assert response.json()["language"] == "English"


def test_multipart_route_returns_validation_error(client, service):
    service.has_asr_aligner = False

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"audio", "audio/wav")},
        data={"response_format": "verbose_json", "timestamp_granularities[]": "word"},
    )

    assert response.status_code == 400
    assert "ASR stage connected to a forced aligner" in response.json()["error"]["message"]
    service.engine_client.generate.assert_not_called()


def test_multipart_route_returns_sse_stream(client, service):
    service.engine_client.generate.side_effect = lambda *args, **kwargs: _results(_text_result("Hello"))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"audio", "audio/wav")},
        data={"stream": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content":"Hello"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_multipart_route_enforces_upload_size_before_decoding(client, service, mocker):
    mocker.patch("vllm.entrypoints.speech_to_text.base.utils.envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 0)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"oversized", "audio/wav")},
        )

    service._decode_and_chunk_speech_async.assert_not_awaited()
    service.engine_client.generate.assert_not_called()
