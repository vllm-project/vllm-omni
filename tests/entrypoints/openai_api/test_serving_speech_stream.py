# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI, WebSocket
from pytest_mock import MockerFixture
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.openai import serving_speech_stream as streaming_speech_module
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.model_executor.stage_input_processors.forced_aligner import ALIGNER_WORDS_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_aligner_res(pairs, words):
    """Build a stand-in for the forced-aligner stage's pooling output.

    Mirrors what rides the generator in production: ``res.outputs.data`` is a
    ``[n_words, 2]`` int32 tensor of ``[start_ms, end_ms]`` and the word strings
    travel in ``additional_information``. Decoded by ``extract_word_timestamps``.
    """
    return SimpleNamespace(
        outputs=SimpleNamespace(data=torch.tensor(pairs, dtype=torch.int32), additional_information=None),
        additional_information={ALIGNER_WORDS_KEY: list(words)},
    )


def _build_test_app(
    speech_service=None,
    *,
    idle_timeout=30.0,
    config_timeout=10.0,
    mocker: MockerFixture | None = None,
):
    if speech_service is None:
        assert mocker is not None
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"RIFF" + b"\x00" * 32, "audio/wav"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-1", object(), {}))
        speech_service.forced_aligner_enabled = False

        async def mock_generate_pcm_chunks(
            _generator, _request_id, *, include_sample_rate=False, tts_params=None, collect=None
        ):
            for chunk in (b"\x01\x02", b"\x03\x04\x05"):
                yield (chunk, 24000) if include_sample_rate else chunk

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()

    handler = OmniStreamingSpeechHandler(
        speech_service=speech_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
    )
    app = FastAPI()

    @app.websocket("/v1/audio/speech/stream")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, speech_service


class TestStreamingSpeechWebSocket:
    def test_non_streaming_single_frame(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["sentence_index"] == 0
                assert start["sentence_text"] == "Hello world."
                assert start["format"] == "wav"

                audio = ws.receive_bytes()
                assert audio.startswith(b"RIFF")

                done = ws.receive_json()
                assert done == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": len(audio),
                    "error": False,
                }

                session_done = ws.receive_json()
                assert session_done == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

        assert speech_service._generate_audio_bytes.await_count == 1

    def test_input_done_flushes_and_keeps_connection_open(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})

                # The same connection serves several utterances; the config sent
                # once at the top keeps applying and utterance_index rises.
                for expected_index, text in enumerate(("First utterance. ", "Second utterance. ")):
                    ws.send_json({"type": "input.text", "text": text})
                    ws.send_json({"type": "input.done"})

                    start = ws.receive_json()
                    assert start["type"] == "audio.start"
                    assert start["utterance_index"] == expected_index
                    assert start["sentence_text"] == text.strip()
                    assert ws.receive_bytes().startswith(b"RIFF")
                    assert ws.receive_json()["type"] == "audio.done"
                    assert ws.receive_json() == {
                        "type": "session.done",
                        "utterance_index": expected_index,
                        "total_sentences": 1,
                    }

        assert speech_service._generate_audio_bytes.await_count == 2
        assert [call.args[0].voice for call in speech_service._generate_audio_bytes.await_args_list] == [
            "Vivian",
            "Vivian",
        ]

    def test_sentence_index_stays_within_the_flushed_utterance(self, mocker: MockerFixture):
        # sentence_index counts within one flush and utterance_index counts the
        # flushes, so a late utterance never reports "sentence 2 of 1".
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})

                for expected_index in range(3):
                    ws.send_json({"type": "input.text", "text": "Hello world. "})
                    ws.send_json({"type": "input.done"})

                    start = ws.receive_json()
                    assert start["utterance_index"] == expected_index
                    assert start["sentence_index"] == 0
                    ws.receive_bytes()

                    done = ws.receive_json()
                    assert done["utterance_index"] == expected_index
                    assert done["sentence_index"] == 0

                    session_done = ws.receive_json()
                    assert session_done["utterance_index"] == expected_index
                    assert session_done["total_sentences"] == 1
                    assert start["sentence_index"] < session_done["total_sentences"]

    def test_session_config_between_utterances_replaces_config(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                for expected_index, voice in enumerate(("Vivian", "Serena")):
                    ws.send_json({"type": "session.config", "voice": voice})
                    ws.send_json({"type": "input.text", "text": "Hello world. "})
                    ws.send_json({"type": "input.done"})

                    assert ws.receive_json()["type"] == "audio.start"
                    ws.receive_bytes()
                    assert ws.receive_json()["type"] == "audio.done"
                    assert ws.receive_json() == {
                        "type": "session.done",
                        "utterance_index": expected_index,
                        "total_sentences": 1,
                    }

        assert [call.args[0].voice for call in speech_service._generate_audio_bytes.await_args_list] == [
            "Vivian",
            "Serena",
        ]

    def test_session_config_rejected_while_input_is_buffered(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "session.config", "voice": "Serena"})

                error = ws.receive_json()
                assert error["type"] == "error"
                assert "while input is buffered" in error["message"]

                # The buffered text survives the rejected reconfiguration.
                ws.send_json({"type": "input.done"})
                assert ws.receive_json()["sentence_text"] == "Hello world."
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

        assert speech_service._generate_audio_bytes.await_args_list[0].args[0].voice == "Vivian"

    def test_session_close_ends_connection(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

                ws.send_json({"type": "session.close"})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()

    def test_idle_timeout_closes_reused_connection(self, mocker: MockerFixture):
        app, _ = _build_test_app(idle_timeout=0.05, mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

                # Holding a connection open is not free: an idle client still
                # gets timed out after the flush.
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Idle timeout: no message received",
                }

    def test_streaming_multiple_binary_frames(self, mocker: MockerFixture):
        captured_requests = []
        captured_tts_params = []

        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_prepare_speech_generation(request):
            captured_requests.append(request)
            return "req-stream", object(), {"_qwen3_tts_effective_max_tokens": [192]}

        speech_service._prepare_speech_generation = mock_prepare_speech_generation

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False, tts_params=None):
            captured_tts_params.append(tts_params)
            for chunk in (b"\x01\x02", b"\x03\x04\x05", b"\x06"):
                yield (chunk, 24000) if include_sample_rate else chunk

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "initial_codec_chunk_frames": 12,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["format"] == "pcm"
                assert start["sample_rate"] == 24000

                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_bytes() == b"\x03\x04\x05"
                assert ws.receive_bytes() == b"\x06"

                done = ws.receive_json()
                assert done == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 6,
                    "error": False,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

        assert len(captured_requests) == 1
        assert captured_requests[0].stream is True
        assert captured_requests[0].response_format == "pcm"
        assert captured_requests[0].initial_codec_chunk_frames == 12
        assert captured_tts_params == [{"_qwen3_tts_effective_max_tokens": [192]}]
        assert speech_service._generate_audio_bytes.await_count == 0

    def test_word_timestamps_requires_configured_aligner(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "word_timestamps": True,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                error = ws.receive_json()
                assert error["type"] == "error"
                assert "without --forced-aligner" in error["message"]

    def test_word_timestamps_emit_pipeline_json_frame(self, mocker: MockerFixture):
        captured_requests = []
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = True

        async def mock_prepare_speech_generation(request):
            captured_requests.append(request)
            return "req-stream", object(), {}

        speech_service._prepare_speech_generation = mock_prepare_speech_generation

        first_chunk = b"\x01" * 1000
        second_chunk = b"\x02" * 1000

        # The forced-aligner stage rides the same generator: its pooling output
        # is surfaced via the ``collect`` channel once the audio has streamed.
        async def mock_generate_pcm_chunks(
            _generator, _request_id, *, include_sample_rate=False, tts_params=None, collect=None
        ):
            for chunk in (first_chunk, second_chunk):
                yield (chunk, 1000) if include_sample_rate else chunk
            if collect is not None:
                collect["aligner_res"] = _fake_aligner_res([[0, 200], [200, 900]], ["Hello", "world"])

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "word_timestamps": True,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["word_timestamps"] is True

                # Audio streams first; timestamps are null until the sentence
                # is fully aligned.
                chunk = ws.receive_json()
                assert chunk["type"] == "audio.chunk"
                assert chunk["utterance_index"] == 0
                assert chunk["sentence_index"] == 0
                assert chunk["chunk_id"] == 0
                assert chunk["chunk_start_ms"] == 0
                assert chunk["chunk_end_ms"] == 500
                assert chunk["sample_rate"] == 1000
                assert base64.b64decode(chunk["audio_b64"]) == first_chunk
                assert chunk["timestamps"] is None

                chunk = ws.receive_json()
                assert chunk["type"] == "audio.chunk"
                assert chunk["chunk_id"] == 1
                assert chunk["chunk_start_ms"] == 500
                assert chunk["chunk_end_ms"] == 1000
                assert chunk["sample_rate"] == 1000
                assert base64.b64decode(chunk["audio_b64"]) == second_chunk
                assert chunk["timestamps"] is None

                # Final frame: empty audio carrying the whole-sentence timestamps.
                chunk = ws.receive_json()
                assert chunk["type"] == "audio.chunk"
                assert chunk["chunk_id"] == 2
                assert chunk["chunk_start_ms"] == 0
                assert chunk["chunk_end_ms"] == 1000
                assert chunk["sample_rate"] == 1000
                assert base64.b64decode(chunk["audio_b64"]) == b""
                assert chunk["timestamps"] == [
                    {"word": "Hello", "start_ms": 0, "end_ms": 200},
                    {"word": "world", "start_ms": 200, "end_ms": 900},
                ]

                done = ws.receive_json()
                assert done == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 2000,
                    "error": False,
                }

        assert captured_requests[0].word_timestamps is True

    def test_word_timestamps_emit_word_dicts(self, mocker: MockerFixture):
        # The streaming layer forwards the aligner's (already monotonic,
        # non-overlapping) words as JSON dicts in the trailing frame.
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = True
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req", object(), {}))

        async def mock_generate_pcm_chunks(
            _generator, _request_id, *, include_sample_rate=False, tts_params=None, collect=None
        ):
            chunk = b"\x01" * 1000
            yield (chunk, 1000) if include_sample_rate else chunk
            if collect is not None:
                collect["aligner_res"] = _fake_aligner_res([[0, 1000], [1000, 1200]], ["Hello", "world"])

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "word_timestamps": True,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                # Real-time audio chunk (timestamps null), then the timestamp frame.
                assert ws.receive_json()["timestamps"] is None
                final = ws.receive_json()
                assert final["type"] == "audio.chunk"
                timestamps = final["timestamps"]
                assert timestamps == [
                    {"word": "Hello", "start_ms": 0, "end_ms": 1000},
                    {"word": "world", "start_ms": 1000, "end_ms": 1200},
                ]
                for ts in timestamps:
                    assert ts["end_ms"] >= ts["start_ms"]

    def test_flush_on_input_done(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world without punctuation"})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_invalid_streaming_config(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "wav",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "response_format='pcm'" in error["message"]

    def test_empty_input_text_emits_no_audio(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": ""})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_multiple_sentences_are_buffered_into_one_request(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "First sentence. "})
                ws.send_json({"type": "input.text", "text": "Second sentence. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["sentence_index"] == 0
                assert start["sentence_text"] == "First sentence. Second sentence."
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_unknown_message_type_keeps_session_open(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "unknown"})

                error = ws.receive_json()
                assert error == {"type": "error", "message": "Unknown message type: unknown"}

                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})
                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_config_timeout_closes_session(self, mocker: MockerFixture):
        app, _ = _build_test_app(config_timeout=0.01, mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                error = ws.receive_json()
                assert error == {"type": "error", "message": "Timeout waiting for session.config"}

    def test_generation_error_marks_audio_done(self, mocker: MockerFixture):
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(side_effect=RuntimeError("boom"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-err", object(), {}))
        speech_service._generate_pcm_chunks = mocker.AsyncMock()
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Generation failed for utterance 0, sentence 0: boom",
                }
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 0,
                    "error": True,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_streaming_generation_error_marks_audio_done(self, mocker: MockerFixture):
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-stream-err", object(), {}))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False, tts_params=None):
            yield b"\x01\x02"
            raise RuntimeError("stream boom")

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Generation failed for utterance 0, sentence 0: stream boom",
                    "partial_audio": True,
                    "action": "discard",
                }
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 2,
                    "error": True,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_invalid_input_text_type_returns_validation_error(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": 123})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "input.text requires a string value",
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_input_text_message_too_large(self, monkeypatch, mocker: MockerFixture):
        monkeypatch.setattr(streaming_speech_module, "_MAX_INPUT_TEXT_MESSAGE_SIZE", 32)
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "x" * 128})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "input.text message too large",
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_session_config_message_too_large(self, monkeypatch, mocker: MockerFixture):
        monkeypatch.setattr(streaming_speech_module, "_MAX_CONFIG_MESSAGE_SIZE", 64)
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian", "ref_audio": "x" * 512})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "session.config message too large",
                }

    def test_disconnect_aborts_streaming_request(self, mocker: MockerFixture):
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-abort", object(), {}))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False, tts_params=None):
            yield b"\x01\x02"

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        handler = OmniStreamingSpeechHandler(speech_service=speech_service)

        websocket = mocker.MagicMock()
        websocket.send_json = mocker.AsyncMock(side_effect=[None, WebSocketDisconnect()])
        websocket.send_bytes = mocker.AsyncMock(side_effect=WebSocketDisconnect())

        config = mocker.MagicMock()
        config.model = None
        config.voice = "Vivian"
        config.task_type = None
        config.language = None
        config.instructions = None
        config.response_format = "pcm"
        config.speed = 1.0
        config.max_new_tokens = None
        config.initial_codec_chunk_frames = None
        config.ref_audio = None
        config.ref_text = None
        config.x_vector_only_mode = None
        config.speaker_embedding = None
        config.stream_audio = True
        config.word_timestamps = False

        with pytest.raises(WebSocketDisconnect):
            asyncio.run(
                handler._generate_and_send(
                    websocket,
                    config,
                    "Hello world.",
                    utterance_index=0,
                    sentence_index=0,
                )
            )

        speech_service.engine_client.abort.assert_awaited_once_with("req-abort")
        assert websocket.send_json.await_count == 2


class TestGeneratePcmChunksContract:
    """Guard: _generate_pcm_chunks must exist on OmniOpenAIServingSpeech.

    The WebSocket handler calls speech_service._generate_pcm_chunks()
    at runtime. If the method is removed, all WS TTS streaming breaks
    with an AttributeError. This test catches that at CI time.
    """

    def test_generate_pcm_chunks_defined(self):
        assert hasattr(OmniOpenAIServingSpeech, "_generate_pcm_chunks")
        assert asyncio.iscoroutinefunction(OmniOpenAIServingSpeech._generate_pcm_chunks) or callable(
            OmniOpenAIServingSpeech._generate_pcm_chunks
        )
