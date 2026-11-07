# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import json
import logging
import wave
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import UploadFile
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.exceptions import VLLMError
from vllm.sampling_params import RequestOutputKind, SamplingParams

from tests.helpers.serving_chat import build_serving_chat, make_text_omni_output
from vllm_omni.entrypoints.openai.serving_transcription import (
    OmniServingTranscription,
    _ChunkedTranscriptionPrompt,
    _TranscriptionEngineClient,
)
from vllm_omni.entrypoints.transcription import (
    SameModelTranscriber,
    TranscriptionSettings,
    TranscriptionStage,
    resolve_transcription_stage,
)
from vllm_omni.utils.audio_resample import StreamingAudioResampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def transcription_logs(caplog, monkeypatch):
    logger = logging.getLogger(f"{__name__}.transcription")
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_chat.logger", logger)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_transcription.logger", logger)
    caplog.set_level(logging.INFO, logger=logger.name)
    return caplog


def _output(**kwargs):
    output = make_text_omni_output(**kwargs)
    output.stage_id = 0
    return output


@pytest.fixture
def transcriber(mocker):
    model_cls = mocker.Mock()
    model_cls.supported_languages = {"de": "German", "en": "English"}
    model_cls.no_space_languages = {"ja", "zh"}
    model_cls.validate_language.side_effect = lambda language: language
    model_cls.supports_explicit_language_detection = False
    model_cls.get_speech_to_text_config.return_value = SimpleNamespace(sample_rate=16000, max_audio_clip_s=30)
    model_cls.post_process_output.side_effect = lambda text: text
    renderer = SimpleNamespace(render_cmpl_async=mocker.AsyncMock(return_value=[{"prompt_token_ids": [1, 2]}]))
    engine = SimpleNamespace(
        renderer=renderer,
        default_sampling_params_list=[SamplingParams(temperature=0.8, stop=["conversation-stop"])],
    )
    stage = TranscriptionStage(0, SimpleNamespace(max_model_len=100), model_cls)
    return SameModelTranscriber(engine, stage, {"zenith", "other-served-alias"})


@pytest.fixture
def chunked_transcriber(transcriber, monkeypatch):
    # Small real waveforms exercise the splitter without model weights.
    transcriber.config.sample_rate = 100
    transcriber.config.max_audio_clip_s = 2
    transcriber.config.min_energy_split_window_size = None
    transcriber.stage.model_cls.get_generation_prompt.return_value = {"prompt": "recognize"}
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.parse_model_prompt", lambda config, prompt: prompt)
    return transcriber


@pytest.mark.asyncio
async def test_long_pcm_transcription_is_sequential_complete_and_request_local(chunked_transcriber):
    transcriber = chunked_transcriber
    submitted = []
    closed = []

    async def generate(**kwargs):
        assert len(closed) == len(submitted)
        submitted.append(kwargs)
        try:
            yield _output(text=f"part{len(submitted)}", finish_reason="stop")
        finally:
            closed.append(kwargs["request_id"])

    transcriber.engine.generate = generate
    recording = np.arange(650, dtype="<i2").tobytes()
    settings = TranscriptionSettings(model="zenith", language="de", prompt="Qwen")
    text = await transcriber.transcribe_pcm(recording, 100, settings, "asr-parent")
    assert len(submitted) >= 4
    assert text == " ".join(f"part{i + 1}" for i in range(len(submitted)))
    assert len({request["request_id"] for request in submitted}) == len(submitted)
    assert all(request["request_id"].startswith("asr-parent-") for request in submitted)
    assert all(request["output_modalities"] == ["text"] for request in submitted)
    assert closed == [request["request_id"] for request in submitted]
    params = [call.args[0] for call in transcriber.stage.model_cls.get_generation_prompt.call_args_list]
    assert all(0 < param.audio.size <= 200 for param in params)
    assert all(param.language == "de" and param.request_prompt == "Qwen" for param in params)
    np.testing.assert_array_equal(
        np.concatenate([param.audio for param in params]), np.arange(650, dtype=np.float32) / 32768.0
    )
    assert transcriber.config.max_audio_clip_s == 2
    assert transcriber.config.min_energy_split_window_size is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "abort", None])
async def test_later_incomplete_chunk_fails_without_returning_prefix(chunked_transcriber, finish_reason):
    transcriber = chunked_transcriber
    submitted = []

    async def generate(**kwargs):
        submitted.append(kwargs["request_id"])
        reason = "stop" if len(submitted) == 1 else finish_reason
        yield _output(text="recognized prefix", finish_reason=reason)

    transcriber.engine.generate = generate
    with pytest.raises(ValueError, match="Transcription"):
        await transcriber.transcribe_pcm(b"\x01\x00" * 650, 100, TranscriptionSettings(model="zenith"), "asr-parent")
    assert len(submitted) == 2


@pytest.mark.asyncio
async def test_cancelling_later_chunk_closes_it_and_never_starts_remaining_chunks(chunked_transcriber):
    transcriber = chunked_transcriber
    started = asyncio.Event()
    closed = []
    submitted = []

    async def generate(**kwargs):
        submitted.append(kwargs["request_id"])
        try:
            if len(submitted) == 2:
                started.set()
                await asyncio.Event().wait()
            yield _output(text="prefix", finish_reason="stop")
        finally:
            closed.append(kwargs["request_id"])

    transcriber.engine.generate = generate
    task = asyncio.create_task(
        transcriber.transcribe_pcm(b"\x01\x00" * 650, 100, TranscriptionSettings(model="zenith"), "asr-parent")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(submitted) == 2
    assert closed == submitted


@pytest.mark.parametrize("silence_at_boundary", [False, True])
def test_split_recording_preserves_samples_and_trailing_clip(chunked_transcriber, silence_at_boundary):
    waveform = np.ones(451, dtype=np.float32)
    if silence_at_boundary:
        waveform[150:160] = 0
    chunks = chunked_transcriber._split_recording(waveform, 100)
    np.testing.assert_array_equal(np.concatenate(chunks), waveform)
    assert all(0 < chunk.size <= 200 for chunk in chunks)
    assert chunks[-1].size < 200
    assert all(np.shares_memory(chunk, waveform) for chunk in chunks)
    if silence_at_boundary:
        assert chunks[0].size == 150


def test_long_recording_resamples_once_before_boundary_search(transcriber, mocker):
    transcriber.config.sample_rate = 16000.0
    transcriber.config.max_audio_clip_s = 1
    waveform = np.sin(np.arange(24000 * 3 + 3, dtype=np.float32) / 10)
    expected = StreamingAudioResampler(24000, 16000).process(waveform, final=True)
    resampler = mocker.spy(StreamingAudioResampler, "process")
    chunks = transcriber._split_recording(waveform, 24000)
    resampler.assert_called_once()
    assert all(0 < chunk.size <= 16000 for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), expected)


@pytest.mark.asyncio
async def test_short_recording_does_not_invoke_splitter_or_change_request_id(chunked_transcriber, mocker):
    transcriber = chunked_transcriber
    split = mocker.patch("vllm_omni.entrypoints.transcription.split_audio")
    transcribe = mocker.patch.object(transcriber, "transcribe_prompt", return_value="short")
    assert (
        await transcriber.transcribe_pcm(b"\x01\x00" * 200, 100, TranscriptionSettings(model="zenith"), "parent")
        == "short"
    )
    split.assert_not_called()
    transcribe.assert_awaited_once_with(transcriber.stage.model_cls.get_generation_prompt.return_value, "parent")


@pytest.mark.asyncio
@pytest.mark.parametrize("silent", [False, True])
async def test_empty_chunk_is_only_omitted_for_digital_silence(chunked_transcriber, silent):
    transcriber = chunked_transcriber

    async def generate(**kwargs):
        yield _output(text="", finish_reason="stop")

    transcriber.engine.generate = generate
    recording = (b"\x00\x00" if silent else b"\x01\x00") * 450
    operation = transcriber.transcribe_pcm(recording, 100, TranscriptionSettings(model="zenith"), "parent")
    if silent:
        assert await operation == ""
    else:
        with pytest.raises(ValueError, match="non-silent chunk"):
            await operation


def test_encoded_decode_uses_global_limits_not_per_clip_limit(transcriber, mocker, monkeypatch):
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.envs.VLLM_MAX_AUDIO_DECODE_DURATION_S", 120)
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.envs.VLLM_MAX_AUDIO_DECODE_BYTES", 123456)
    decoded = np.ones(16000 * 40, dtype=np.float32)
    load = mocker.patch("vllm_omni.entrypoints.transcription.load_audio", return_value=(decoded, 16000))
    waveform, sample_rate = transcriber._prepare_encoded(b"encoded")
    assert waveform is decoded
    assert sample_rate == 16000
    assert load.call_args.kwargs == {"sr": 16000, "mono": True, "max_duration_s": 120, "max_decode_bytes": 123456}
    transcriber.max_file_bytes = 1
    with pytest.raises(ValueError, match="upload size"):
        transcriber._prepare_encoded(b"too large")
    assert load.call_count == 1


@pytest.mark.asyncio
async def test_encoded_recording_returns_one_complete_transcript(chunked_transcriber):
    transcriber = chunked_transcriber
    recording = BytesIO()
    with wave.open(recording, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(100)
        output.writeframes(b"\x01\x00" * 450)

    async def generate(**kwargs):
        yield _output(text="words", finish_reason="stop")

    transcriber.engine.generate = generate
    text = await transcriber.transcribe_encoded(recording.getvalue(), TranscriptionSettings(model="zenith"), "upload")
    count = transcriber.stage.model_cls.get_generation_prompt.call_count
    assert count >= 3
    assert text == " ".join(["words"] * count)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ValueError("duration limit"), ValueError("decode bytes limit"), ValueError("malformed")]
)
async def test_encoded_decode_failure_never_starts_inference(transcriber, mocker, failure):
    mocker.patch("vllm_omni.entrypoints.transcription.load_audio", side_effect=failure)
    recognize = mocker.patch.object(transcriber, "transcribe_prompt")
    chunks = mocker.patch.object(transcriber, "transcribe_chunks")
    with pytest.raises(ValueError) as caught:
        await transcriber.transcribe_encoded(b"recording", TranscriptionSettings(model="zenith"), "upload")
    assert caught.value is failure
    recognize.assert_not_called()
    chunks.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_long", [False, True])
async def test_http_preprocessing_never_renders_an_oversized_recording(chunked_transcriber, mocker, is_long):
    transcriber = chunked_transcriber
    handler = object.__new__(OmniServingTranscription)
    handler.transcriber = transcriber
    handler.model_cls = transcriber.stage.model_cls
    handler.model_config = transcriber.stage.model_config
    handler.asr_config = transcriber.config
    handler.renderer = transcriber.renderer
    handler.task_type = "transcribe"
    waveform = np.ones(450 if is_long else 100, dtype=np.float32)
    handler._decode_and_chunk_speech_async = mocker.AsyncMock(return_value=([waveform], waveform.size / 100))
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_transcription.parse_model_prompt", side_effect=lambda config, p: p
    )
    request = TranscriptionRequest(
        file=UploadFile(BytesIO(b"recording"), filename="audio.wav"), model="zenith", language="de", prompt="Qwen"
    )
    prompts, duration, offsets = await handler._preprocess_speech_to_text(request, b"recording", "upload")
    assert len(prompts) == 1
    assert duration == waveform.size / 100
    assert offsets == [0.0]
    if is_long:
        assert isinstance(prompts[0], _ChunkedTranscriptionPrompt)
        assert prompts[0].settings.language == "de"
        assert prompts[0].settings.prompt == "Qwen"
        np.testing.assert_array_equal(np.concatenate(prompts[0].chunks), waveform)
        assert all(chunk.size <= 200 for chunk in prompts[0].chunks)
        assert dict(prompts[0]) == {"type": "token", "prompt_token_ids": []}
        handler.renderer.render_cmpl_async.assert_not_awaited()
        handler.model_cls.get_generation_prompt.assert_not_called()
    else:
        assert prompts == [{"prompt_token_ids": [1, 2]}]
        handler.renderer.render_cmpl_async.assert_awaited_once()
        assert handler.model_cls.get_generation_prompt.call_args.args[0].audio is waveform


@pytest.mark.asyncio
@pytest.mark.parametrize("output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.parametrize("fail_late", [False, True])
async def test_http_chunk_adapter_emits_one_complete_result_or_no_prefix(chunked_transcriber, output_kind, fail_late):
    transcriber = chunked_transcriber
    submitted = []
    lora = object()
    headers = {"traceparent": "test"}

    async def generate(**kwargs):
        submitted.append(kwargs)
        if fail_late and len(submitted) == 2:
            raise RuntimeError("later clip failed")
        yield _output(text="partial")
        yield _output(text=f"part{len(submitted)}", finish_reason="stop")

    transcriber.engine.generate = generate
    adapter = _TranscriptionEngineClient(transcriber)
    prompt = _ChunkedTranscriptionPrompt(
        transcriber._split_recording(np.ones(450, dtype=np.float32), 100),
        TranscriptionSettings(model="zenith"),
    )
    params = SamplingParams(max_tokens=50, temperature=0.2, output_kind=output_kind)
    results = []

    async def consume():
        async for result in adapter.generate(prompt, params, "upload", lora_request=lora, trace_headers=headers):
            results.append(result)

    if fail_late:
        with pytest.raises(RuntimeError, match="later clip failed"):
            await consume()
        assert results == []
        assert len(submitted) == 2
    else:
        await consume()
        assert len(results) == 1
        assert results[0].request_id == "upload"
        assert results[0].finished
        assert results[0].outputs[0].text == " ".join(f"part{i + 1}" for i in range(len(submitted)))
        assert results[0].outputs[0].finish_reason == "stop"
        assert len(results[0].prompt_token_ids) == 2 * len(submitted)
        assert len(results[0].outputs[0].token_ids) == 3 * len(submitted)
    assert all(request["lora_request"] is lora and request["trace_headers"] is headers for request in submitted)
    assert all(request["sampling_params_list"][0].temperature == 0.2 for request in submitted)
    assert all(request["sampling_params_list"][0].max_tokens == 50 for request in submitted)
    assert params.output_kind == output_kind


def test_transcription_settings_resolve_only_served_aliases(transcriber):
    assert transcriber.validate_settings({"model": "zenith", "language": "de"}).language == "de"
    assert transcriber.validate_settings({"model": "other-served-alias"}).language is None
    with pytest.raises(ValueError, match="served base model"):
        transcriber.validate_settings({"model": "whisper-1"})
    with pytest.raises(ValueError, match="language"):
        transcriber.validate_settings({"model": "zenith", "language": "unknown"})
    with pytest.raises(ValueError, match="settings"):
        transcriber.validate_settings({"model": "zenith", "timestamps": True})


@pytest.mark.parametrize("stage_id", [0, 1])
def test_capability_uses_comprehension_configuration_and_text_endpoint(monkeypatch, stage_id):
    config = SimpleNamespace(name="comprehension-config")
    stage = SimpleNamespace(is_comprehension=True, final_output=True, final_output_type="text")
    model_cls = object()
    resolve = Mock(return_value=model_cls)
    monkeypatch.setattr("vllm.model_executor.model_loader.get_model_cls", resolve)
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.supports_transcription", lambda cls: cls is model_cls)
    engine = SimpleNamespace(
        _get_comprehension_stage_index=lambda: stage_id,
        output_modalities=["text", "audio"],
        _compute_final_output_stage_ids=lambda _: [stage_id],
        get_stage_renderer=Mock(),
        engine=SimpleNamespace(
            stage_configs=[None] * stage_id + [stage],
            stage_vllm_configs=[None] * stage_id + [SimpleNamespace(model_config=config)],
        ),
    )
    assert resolve_transcription_stage(engine) == TranscriptionStage(stage_id, config, model_cls)
    resolve.assert_called_once_with(config)
    stage.final_output_type = "audio"
    assert resolve_transcription_stage(engine) is None
    stage.final_output_type = "text"
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.supports_transcription", lambda _: False)
    assert resolve_transcription_stage(engine) is None


def test_pcm_preparation_resamples_without_mutating_recording(transcriber):
    waveform = np.ones(24000, dtype=np.float32) * 0.25
    original = waveform.copy()
    settings = TranscriptionSettings(model="zenith", language="de", prompt="Qwen")
    transcriber._prepare_audio(waveform, 24000, settings)
    params = transcriber.stage.model_cls.get_generation_prompt.call_args.args[0]
    assert params.task_type == "transcribe"
    assert params.language == "de"
    assert params.request_prompt == "Qwen"
    assert params.audio.size == 16000
    np.testing.assert_array_equal(waveform, original)
    with pytest.raises(ValueError, match="duration limit"):
        transcriber._prepare_audio(np.zeros(24000 * 31), 24000, settings)


@pytest.mark.asyncio
async def test_transcription_cumulative_output_and_fresh_text_only_sampling(transcriber, monkeypatch):
    submitted = []

    async def generate(**kwargs):
        submitted.append(kwargs)
        yield _output(text="Wie")
        yield _output(text="Wie geht es dir?", finish_reason="stop")

    transcriber.engine.generate = generate
    monkeypatch.setattr("vllm_omni.entrypoints.transcription.parse_model_prompt", lambda config, prompt: prompt)
    assert await transcriber.transcribe_prompt({"prompt": "recognize"}, "asr-item") == "Wie geht es dir?"
    request = submitted[0]
    assert request["entry_stage_id"] == 0
    assert request["output_modalities"] == ["text"]
    assert request["request_id"] == "asr-item"
    params = request["sampling_params_list"][0]
    assert params.temperature == 0
    assert params.stop == []
    assert params.max_tokens == 98
    assert params.output_kind == RequestOutputKind.CUMULATIVE
    assert transcriber.engine.default_sampling_params_list[0].stop == ["conversation-stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "abort", None])
async def test_incomplete_recognition_is_not_success(transcriber, finish_reason):
    async def generate(**kwargs):
        yield _output(text="prefix", finish_reason=finish_reason)

    transcriber.engine.generate = generate
    with pytest.raises(ValueError, match="Transcription"):
        async for _ in transcriber.generate({"prompt_token_ids": [1]}, SamplingParams(), "asr-limit"):
            pass


@pytest.mark.asyncio
async def test_transcription_cancellation_closes_engine_generator(transcriber):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def generate(**kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
            yield _output()
        finally:
            closed.set()

    async def consume():
        async for _ in transcriber.generate({"prompt_token_ids": [1]}, SamplingParams(), "asr-cancel"):
            pass

    transcriber.engine.generate = generate
    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


def _inline_request(**overrides):
    return ChatCompletionRequest(
        model="zenith",
        stream=True,
        messages=[
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "d2F2", "format": "wav"}}]}
        ],
        **overrides,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_inline_transcript_is_opt_in_and_not_an_assistant_delta(transcriber, failure, transcription_logs):
    handler = build_serving_chat()
    handler.input_audio_transcriber = transcriber
    assert handler._inline_transcription_request(_inline_request()) is None
    request = _inline_request(input_audio_transcription={"language": "de"})
    inline = handler._inline_transcription_request(request)
    transcriber.transcribe_encoded = AsyncMock(
        side_effect=RuntimeError("private contents") if failure else None,
        return_value="Guten Tag",
    )
    answer = 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'

    async def assistant_stream():
        yield answer
        yield "data: [DONE]\n\n"

    chunks = [chunk async for chunk in handler._stream_with_input_transcription(assistant_stream(), inline, "chat-id")]
    suffix = "failed" if failure else "completed"
    assert chunks[0].startswith(f"event: omni.input_audio_transcription.{suffix}\n")
    event = json.loads(chunks[0].split("data: ")[1])
    assert event["message_index"] == 0
    assert event["content_index"] == 0
    assert "choices" not in event
    assert "private contents" not in chunks[0]
    if not failure:
        assert event["transcript"] == "Guten Tag"
    assert chunks[1:] == [answer, "data: [DONE]\n\n"]
    transcriber.transcribe_encoded.assert_awaited_once_with(base64.b64decode("d2F2"), inline.settings, "asr-chat-id")
    if failure:
        assert len(transcription_logs.records) == 1
        record = transcription_logs.records[0]
        assert record.levelno == logging.ERROR
        assert isinstance(record.exc_info[1], RuntimeError)
        assert record.exc_info[2] is not None
        assert "Inline ASR failed request_id=asr-chat-id message_index=0 content_index=0" in record.getMessage()
        assert "private contents" in transcription_logs.text
    else:
        assert not transcription_logs.records
    assert "Guten Tag" not in transcription_logs.text
    assert "d2F2" not in transcription_logs.text


def test_inline_rejects_history_only_audio(transcriber):
    handler = build_serving_chat()
    handler.input_audio_transcriber = transcriber
    request = _inline_request(input_audio_transcription={})
    request.messages.append({"role": "user", "content": "ordinary text"})
    with pytest.raises(ValueError, match="latest user message"):
        handler._inline_transcription_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["zenith", None, ""])
async def test_http_default_request_reaches_upstream_handler(transcriber, monkeypatch, model):
    handler = object.__new__(OmniServingTranscription)
    handler.transcriber = transcriber
    handler.models = SimpleNamespace(model_name=lambda: "zenith")
    upstream = AsyncMock(return_value="recognized")
    monkeypatch.setattr(OpenAIServingTranscription, "create_transcription", upstream)
    request = TranscriptionRequest(file=UploadFile(BytesIO(b"recording"), filename="audio.wav"), model=model)
    assert await handler.create_transcription(b"recording", request) == "recognized"
    assert request.model == "zenith"
    upstream.assert_awaited_once_with(b"recording", request, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"response_format": "verbose_json"},
        {"timestamp_granularities[]": ["word"]},
        {"to_language": "en"},
        {"hotwords": "Qwen"},
        {"model": "whisper-1"},
        {"use_beam_search": True},
    ],
)
async def test_http_rejects_unavailable_transcription_options(transcriber, monkeypatch, options, transcription_logs):
    handler = object.__new__(OmniServingTranscription)
    handler.transcriber = transcriber
    handler.create_error_response = Mock(return_value="unsupported")
    upstream = AsyncMock()
    monkeypatch.setattr(OpenAIServingTranscription, "create_transcription", upstream)
    request = TranscriptionRequest(
        file=UploadFile(BytesIO(b"recording"), filename="audio.wav"), **{"model": "zenith", **options}
    )
    assert await handler.create_transcription(b"recording", request) == "unsupported"
    upstream.assert_not_awaited()
    assert len(transcription_logs.records) == 1
    assert transcription_logs.records[0].levelno == logging.WARNING
    assert transcription_logs.records[0].exc_info is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ValueError, TypeError, AttributeError, RuntimeError])
async def test_http_internal_failure_logs_once_and_preserves_response(
    transcriber, monkeypatch, transcription_logs, error_type
):
    handler = object.__new__(OmniServingTranscription)
    handler.transcriber = transcriber
    handler.create_error_response = Mock(return_value="failed")
    failure = error_type("recognition failed inside renderer")

    async def upstream(self, audio_data, request, raw_request):
        raw_request.state.request_metadata = SimpleNamespace(request_id="transcribe-http-id")
        raise failure

    monkeypatch.setattr(OpenAIServingTranscription, "create_transcription", upstream)
    raw_request = SimpleNamespace(state=SimpleNamespace())
    request = TranscriptionRequest(file=UploadFile(BytesIO(b"recording"), filename="audio.wav"), model="zenith")
    if error_type is ValueError:
        assert await handler.create_transcription(b"recording", request, raw_request) == "failed"
        handler.create_error_response.assert_called_once_with(
            "Input transcription failed; check the recording and model audio/context/output limits"
        )
    else:
        with pytest.raises(error_type) as caught:
            await handler.create_transcription(b"recording", request, raw_request)
        assert caught.value is failure
        handler.create_error_response.assert_not_called()
    assert len(transcription_logs.records) == 1
    record = transcription_logs.records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info[1] is failure
    assert record.exc_info[2] is not None
    assert "HTTP ASR failed request_id=transcribe-http-id" in record.getMessage()
    assert "recognition failed inside renderer" in transcription_logs.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [VLLMError("upstream-owned"), asyncio.CancelledError()])
async def test_http_upstream_errors_and_cancellation_remain_unlogged(
    transcriber, monkeypatch, transcription_logs, failure
):
    handler = object.__new__(OmniServingTranscription)
    handler.transcriber = transcriber
    monkeypatch.setattr(OpenAIServingTranscription, "create_transcription", AsyncMock(side_effect=failure))
    request = TranscriptionRequest(file=UploadFile(BytesIO(b"recording"), filename="audio.wav"), model="zenith")
    with pytest.raises(type(failure)) as caught:
        await handler.create_transcription(b"recording", request)
    assert caught.value is failure
    assert not transcription_logs.records


@pytest.mark.asyncio
async def test_http_adapter_uses_selected_stage_and_sampling(transcriber):
    stage = TranscriptionStage(1, transcriber.stage.model_config, transcriber.stage.model_cls)
    transcriber.stage = stage
    transcriber.engine.default_sampling_params_list.append(SamplingParams())
    submitted = []

    async def generate(**kwargs):
        submitted.append(kwargs)
        output = _output(text="recognized", finish_reason="stop")
        output.stage_id = 1
        yield output

    transcriber.engine.generate = generate
    adapter = _TranscriptionEngineClient(transcriber)
    assert adapter.model_config is stage.model_config
    assert adapter.renderer is transcriber.renderer
    params = SamplingParams(max_tokens=50, temperature=0.1)
    outputs = [output async for output in adapter.generate({"prompt_token_ids": [1]}, params, "http-asr")]
    assert outputs[0].outputs[0].text == "recognized"
    assert submitted[0]["entry_stage_id"] == 1
    assert submitted[0]["output_modalities"] == ["text"]
    assert submitted[0]["sampling_params_list"][1].max_tokens == 50
    assert submitted[0]["sampling_params_list"][1].temperature == 0.1
