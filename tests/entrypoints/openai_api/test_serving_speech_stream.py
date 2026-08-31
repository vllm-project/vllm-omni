# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import threading
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI, WebSocket
from pydantic import ValidationError
from pytest_mock import MockerFixture
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.openai import serving_speech_stream as streaming_speech_module
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.openai.tts_adapters.base import TTSModelAdapter
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
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
    commitment_supported=False,
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

    if commitment_supported:
        assert mocker is not None
        adapter = mocker.MagicMock(spec=TTSModelAdapter)
        adapter.supported_text_input_modes = frozenset({"buffered", "commitment"})
        speech_service._get_tts_adapter.return_value = adapter

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
    def test_text_input_mode_defaults_to_buffered(self):
        assert streaming_speech_module.StreamingSpeechSessionConfig().text_input_mode == "buffered"

    def test_text_input_mode_accepts_commitment(self):
        assert (
            streaming_speech_module.StreamingSpeechSessionConfig(text_input_mode="commitment").text_input_mode
            == "commitment"
        )

    def test_text_input_mode_rejects_unknown_mode(self):
        with pytest.raises(ValidationError, match="text_input_mode"):
            streaming_speech_module.StreamingSpeechSessionConfig(
                text_input_mode="incremental"  # type: ignore[arg-type]
            )

    def test_adapter_text_input_mode_capabilities(self):
        assert TTSModelAdapter.supported_text_input_modes == frozenset({"buffered"})
        assert Qwen3TTSAdapter.supported_text_input_modes == frozenset({"buffered", "commitment"})

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

    def test_commitment_mode_segments_before_eof_in_order_and_preserves_source(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(
            mocker=mocker,
            commitment_supported=True,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "language": "english",
                        "text_input_mode": "commitment",
                    }
                )
                # The unresolved numeric suffix must not leak at this packet
                # frontier. The first synthesis request is emitted only once
                # the following packet closes the sentence.
                ws.send_json({"type": "input.text", "text": "The total is 2026"})
                ws.send_json({"type": "input.text", "text": " dollars. "})

                first = ws.receive_json()
                assert first["type"] == "audio.start"
                assert first["sentence_index"] == 0
                assert first["sentence_text"] == "The total is 2026 dollars."
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"

                # This suffix has no strong boundary and is therefore flushed
                # only by EOF. Leading whitespace remains raw source text.
                ws.send_json({"type": "input.text", "text": "Thank you"})
                ws.send_json({"type": "input.done"})
                second = ws.receive_json()
                assert second["type"] == "audio.start"
                assert second["sentence_index"] == 1
                assert second["sentence_text"] == " Thank you"
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"

                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 2,
                }

        assert [call.args[0].input for call in speech_service._generate_audio_bytes.await_args_list] == [
            "The total is 2026 dollars.",
            " Thank you",
        ]
        assert all("request_id" in call.kwargs for call in speech_service._generate_audio_bytes.await_args_list)

    def test_commitment_keeps_decimals_and_terminator_runs_in_one_request(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Value:\n.5 seconds. Wait... What?! "})
                ws.send_json({"type": "input.done"})

                for expected_text in ("Value:\n", ".5 seconds.", " Wait...", " What?!"):
                    start = ws.receive_json()
                    assert start["type"] == "audio.start"
                    assert start["sentence_text"] == expected_text
                    ws.receive_bytes()
                    assert ws.receive_json()["type"] == "audio.done"

                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 4,
                }

        assert [call.args[0].input for call in speech_service._generate_audio_bytes.await_args_list] == [
            "Value:\n",
            ".5 seconds.",
            " Wait...",
            " What?!",
        ]

    def test_commitment_newline_is_not_consumed_as_unit_whitespace(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "There are 3\nMore things. "})
                ws.send_json({"type": "input.done"})

                for expected_text in ("There are 3\n", "More things."):
                    start = ws.receive_json()
                    assert start["type"] == "audio.start"
                    assert start["sentence_text"] == expected_text
                    ws.receive_bytes()
                    assert ws.receive_json()["type"] == "audio.done"

                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 2,
                }

        assert [call.args[0].input for call in speech_service._generate_audio_bytes.await_args_list] == [
            "There are 3\n",
            "More things.",
        ]

    @pytest.mark.asyncio
    async def test_commitment_queue_backpressures_segment_producer(self, mocker: MockerFixture):
        handler = OmniStreamingSpeechHandler(speech_service=mocker.MagicMock())
        state = streaming_speech_module._CommitmentUtterance(
            index=0,
            config=streaming_speech_module.StreamingSpeechSessionConfig(),
            queue=asyncio.Queue(maxsize=1),
        )
        state.queue.put_nowait("first")
        state.segment_parts.append("second!")
        handler._enqueue_commitment_segment(state)

        producer = asyncio.create_task(handler._commitment_segment_producer(state))
        await asyncio.sleep(0)
        assert not producer.done()

        assert state.queue.get_nowait() == "first"
        state.queue.task_done()
        assert await asyncio.wait_for(state.queue.get(), timeout=1) == "second!"
        state.queue.task_done()

        state.ingress.put_nowait(streaming_speech_module._UTTERANCE_EOF)
        await asyncio.wait_for(producer, timeout=1)
        assert state.queue.get_nowait() is streaming_speech_module._UTTERANCE_EOF
        state.queue.task_done()
        await asyncio.wait_for(state.ingress.join(), timeout=1)
        await asyncio.wait_for(state.queue.join(), timeout=1)

    def test_commitment_backpressure_does_not_block_session_close(self, mocker: MockerFixture):
        generation_started = threading.Event()
        release_generation = threading.Event()
        abort_observed = threading.Event()

        async def blocked_audio(*_args, **_kwargs):
            generation_started.set()
            await asyncio.to_thread(release_generation.wait)
            return b"RIFF", "audio/wav"

        async def observe_abort(*_args, **_kwargs):
            abort_observed.set()

        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = blocked_audio
        speech_service.engine_client.abort.side_effect = observe_abort

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "Chinese",
                        "text_input_mode": "commitment",
                    }
                )
                # One active request plus eight queued segments fills the
                # bounded queue; the tenth segment blocks only the producer.
                ws.send_json({"type": "input.text", "text": "甲！乙！丙！丁！戊！己！庚！辛！壬！癸！ "})
                assert ws.receive_json()["type"] == "audio.start"
                assert generation_started.wait(timeout=1)

                try:
                    ws.send_json({"type": "session.close"})
                    assert abort_observed.wait(timeout=1)
                finally:
                    # Release the helper thread even if the responsiveness
                    # assertion fails against a regressed implementation.
                    release_generation.set()

        speech_service.engine_client.abort.assert_awaited_once()
        assert speech_service._generate_audio_bytes.await_count == 1

    @pytest.mark.asyncio
    async def test_commitment_worker_exit_releases_backpressured_producer(self, mocker: MockerFixture):
        _, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        handler = OmniStreamingSpeechHandler(speech_service=speech_service)
        worker_exited = asyncio.Event()
        input_done_processed = asyncio.Event()
        observed_states = []

        original_finish = handler._finish_commitment_utterance

        def observe_finish(state):
            original_finish(state)
            input_done_processed.set()

        mocker.patch.object(handler, "_finish_commitment_utterance", side_effect=observe_finish)

        async def exit_when_queue_full(_websocket, state):
            observed_states.append(state)
            while not state.queue.full():
                await asyncio.sleep(0)
            await input_done_processed.wait()
            worker_exited.set()
            # A consumer that returns unexpectedly must be treated just like
            # one that raises; otherwise the full-queue producer never exits.
            return

        mocker.patch.object(handler, "_commitment_worker", side_effect=exit_when_queue_full)

        class ProbeWebSocket:
            def __init__(self):
                self.frames = [
                    '{"type":"session.config","language":"Chinese","text_input_mode":"commitment"}',
                    '{"type":"input.text","text":"甲！乙！丙！丁！戊！己！庚！辛！壬！癸！ "}',
                    '{"type":"input.done"}',
                ]
                self.never = asyncio.Event()
                self.sent = []

            async def accept(self):
                pass

            async def receive_text(self):
                if self.frames:
                    return self.frames.pop(0)
                await self.never.wait()
                raise AssertionError("unreachable")

            async def send_json(self, data):
                self.sent.append(data)

            async def send_bytes(self, _data):
                pass

            async def close(self):
                pass

        websocket = ProbeWebSocket()
        await asyncio.wait_for(handler.handle_session(websocket), timeout=1)  # type: ignore[arg-type]

        assert worker_exited.is_set()
        assert input_done_processed.is_set()
        assert any(
            message["type"] == "error" and "synthesis consumer exited before observing EOF" in message["message"]
            for message in websocket.sent
        )
        assert len(observed_states) == 1
        state = observed_states[0]
        assert all(task is not None and task.done() for task in (state.producer, state.consumer, state.worker))
        await asyncio.wait_for(state.ingress.join(), timeout=1)
        await asyncio.wait_for(state.queue.join(), timeout=1)

    @pytest.mark.asyncio
    async def test_receive_keeps_pending_frame_when_committed_work_settles(self, mocker: MockerFixture):
        handler = OmniStreamingSpeechHandler(speech_service=mocker.MagicMock())
        state = streaming_speech_module._CommitmentUtterance(
            index=0,
            config=streaming_speech_module.StreamingSpeechSessionConfig(),
        )
        state.queue.put_nowait("work")

        worker_stop = asyncio.Event()
        state.worker = asyncio.create_task(worker_stop.wait())
        receive_started = asyncio.Event()
        frame_ready = asyncio.Event()

        class ProbeWebSocket:
            receive_calls = 0

            async def receive_text(self):
                self.receive_calls += 1
                receive_started.set()
                await frame_ready.wait()
                return '{"type":"input.text","text":"next"}'

        websocket = ProbeWebSocket()
        receive = asyncio.create_task(
            handler._receive_text(
                websocket,  # type: ignore[arg-type]
                timeout=1,
                commitment=state,
            )
        )
        try:
            await asyncio.wait_for(receive_started.wait(), timeout=1)
            assert state.queue.get_nowait() == "work"
            state.queue.task_done()
            await state.queue.join()
            await asyncio.sleep(0)
            frame_ready.set()

            assert await asyncio.wait_for(receive, timeout=1) == (
                '{"type":"input.text","text":"next"}',
                False,
            )
            assert websocket.receive_calls == 1
        finally:
            worker_stop.set()
            await state.worker

    @pytest.mark.asyncio
    async def test_receive_preserves_next_frame_after_eof_worker_finishes(self, mocker: MockerFixture):
        handler = OmniStreamingSpeechHandler(speech_service=mocker.MagicMock())
        state = streaming_speech_module._CommitmentUtterance(
            index=0,
            config=streaming_speech_module.StreamingSpeechSessionConfig(),
            eof=True,
        )

        worker_stop = asyncio.Event()
        state.worker = asyncio.create_task(worker_stop.wait())
        receive_started = asyncio.Event()
        frame_ready = asyncio.Event()

        class ProbeWebSocket:
            receive_calls = 0

            async def receive_text(self):
                self.receive_calls += 1
                receive_started.set()
                await frame_ready.wait()
                return '{"type":"input.text","text":"next utterance"}'

        websocket = ProbeWebSocket()
        receive = asyncio.create_task(
            handler._receive_text(
                websocket,  # type: ignore[arg-type]
                timeout=1,
                commitment=state,
            )
        )
        await asyncio.wait_for(receive_started.wait(), timeout=1)
        worker_stop.set()
        await state.worker
        await asyncio.sleep(0)
        frame_ready.set()

        assert await asyncio.wait_for(receive, timeout=1) == (
            '{"type":"input.text","text":"next utterance"}',
            True,
        )
        assert websocket.receive_calls == 1

    @pytest.mark.asyncio
    async def test_websocket_writes_are_serialized(self):
        class ProbeWebSocket:
            def __init__(self):
                self.active_writes = 0
                self.max_active_writes = 0

            async def _write(self):
                self.active_writes += 1
                self.max_active_writes = max(self.max_active_writes, self.active_writes)
                await asyncio.sleep(0)
                self.active_writes -= 1

            async def send_json(self, _data):
                await self._write()

            async def send_bytes(self, _data):
                await self._write()

        probe = ProbeWebSocket()
        websocket = streaming_speech_module._SerializedWebSocket(probe)

        await asyncio.gather(websocket.send_json({"type": "error"}), websocket.send_bytes(b"audio"))

        assert probe.max_active_writes == 1

    @pytest.mark.parametrize("language", (None, "Auto", "French"))
    def test_commitment_mode_requires_chinese_or_english(self, language, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker, commitment_supported=True)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "language": language,
                        "text_input_mode": "commitment",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "language='Chinese' or language='English'" in error["message"]

    def test_commitment_mode_requires_model_opt_in(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)
        adapter = mocker.MagicMock(spec=TTSModelAdapter)
        adapter.supported_text_input_modes = frozenset({"buffered"})
        speech_service._get_tts_adapter.return_value = adapter

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "Chinese",
                        "text_input_mode": "commitment",
                    }
                )
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "text_input_mode='commitment' is not supported by the configured TTS model",
                }

    def test_commitment_failure_drops_queued_segments_and_finishes_at_eof(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = RuntimeError("boom")

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "First! Second! Third!"})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "boom" in error["message"]
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 0,
                    "error": True,
                }
                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 1,
                }

        assert speech_service._generate_audio_bytes.await_count == 1

    def test_commitment_utterance_limit_fails_until_eof(self, monkeypatch, mocker: MockerFixture):
        monkeypatch.setattr(streaming_speech_module, "_MAX_COMMITMENT_UTTERANCE_CHARS", 3)
        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "Chinese",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "1234"})
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Commitment utterance exceeds 3 characters",
                }
                ws.send_json({"type": "input.text", "text": "late"})
                assert "current utterance has failed" in ws.receive_json()["message"]
                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 0,
                }

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_commitment_rejects_overlap_and_reconfiguration_while_draining(self, mocker: MockerFixture):
        async def slow_audio(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            return b"RIFF", "audio/wav"

        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = slow_audio

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello!"})
                ws.send_json({"type": "input.done"})
                ws.send_json({"type": "input.text", "text": "overlap"})
                ws.send_json({"type": "session.config", "language": "Chinese"})

                messages = [ws.receive_json(), ws.receive_json(), ws.receive_json()]
                assert messages[0]["type"] == "audio.start"
                assert messages[1]["type"] == "error"
                assert "still active" in messages[1]["message"]
                assert messages[2]["type"] == "error"
                assert "utterance is active" in messages[2]["message"]
                assert ws.receive_bytes() == b"RIFF"
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json()["type"] == "session.done"

    def test_commitment_session_close_aborts_nonstreaming_request_once(self, mocker: MockerFixture):
        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(3600)

        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = never_finishes

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                # The following space confirms that the terminator run has
                # ended while input remains open.
                ws.send_json({"type": "input.text", "text": "Hello! "})
                assert ws.receive_json()["type"] == "audio.start"
                # Cancellation after EOF must still omit session.done.
                ws.send_json({"type": "input.done"})
                ws.send_json({"type": "session.close"})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()

        speech_service.engine_client.abort.assert_awaited_once()
        request_id = speech_service.engine_client.abort.await_args.args[0]
        assert request_id.startswith("speech-stream-")

    @pytest.mark.asyncio
    async def test_commitment_cancellation_suppresses_abort_induced_terminal_events(self, mocker: MockerFixture):
        generation_started = asyncio.Event()
        abort_released_generation = asyncio.Event()

        async def abort_induced_failure(*_args, **_kwargs):
            generation_started.set()
            await abort_released_generation.wait()
            raise RuntimeError("engine request aborted")

        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(side_effect=abort_induced_failure)
        speech_service.forced_aligner_enabled = False
        handler = OmniStreamingSpeechHandler(speech_service=speech_service)
        websocket = mocker.MagicMock()
        websocket.send_json = mocker.AsyncMock()
        websocket.send_bytes = mocker.AsyncMock()
        cancellation_event = asyncio.Event()

        generation = asyncio.create_task(
            handler._generate_and_send(
                websocket,
                streaming_speech_module.StreamingSpeechSessionConfig(language="English"),
                "Hello!",
                utterance_index=0,
                sentence_index=0,
                suppress_done_on_cancel=True,
                cancellation_event=cancellation_event,
            )
        )
        await generation_started.wait()

        # Model an engine abort that wakes generation with an ordinary
        # exception instead of delivering task-level CancelledError.
        cancellation_event.set()
        abort_released_generation.set()

        assert await generation is False
        websocket.send_json.assert_awaited_once()
        assert websocket.send_json.await_args.args[0]["type"] == "audio.start"
        websocket.send_bytes.assert_not_awaited()

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

    def test_commitment_active_generation_pauses_idle_timeout(self, mocker: MockerFixture):
        generation_started = threading.Event()
        release_generation = threading.Event()

        async def blocked_audio(*_args, **_kwargs):
            generation_started.set()
            await asyncio.to_thread(release_generation.wait)
            return b"RIFF", "audio/wav"

        app, speech_service = _build_test_app(
            idle_timeout=0.02,
            commitment_supported=True,
            mocker=mocker,
        )
        speech_service._generate_audio_bytes.side_effect = blocked_audio

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                # A non-terminator after ``!`` closes the punctuation run
                # without requiring EOF.
                ws.send_json({"type": "input.text", "text": "Hello! "})

                assert ws.receive_json() == {
                    "type": "audio.start",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "sentence_text": "Hello!",
                    "format": "wav",
                }
                assert generation_started.wait(timeout=1)

                try:
                    # Hold generation across ten idle-timeout windows. Active
                    # model work is not client idleness and must not be
                    # cancelled or aborted.
                    assert not release_generation.wait(timeout=0.2)
                    speech_service.engine_client.abort.assert_not_awaited()
                finally:
                    # Also release the worker when an assertion fails so the
                    # TestClient server thread cannot be left behind.
                    release_generation.set()

                assert ws.receive_bytes() == b"RIFF"
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 4,
                    "error": False,
                }

                # No EOF was sent. Once generation finishes, the empty queue
                # is genuinely idle and the receive timeout starts again.
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Idle timeout: no message received",
                }

        speech_service.engine_client.abort.assert_not_awaited()

    def test_commitment_unresolved_suffix_still_times_out(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(
            idle_timeout=0.02,
            commitment_supported=True,
            mocker=mocker,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                # With no committed boundary, the worker has no queued or
                # active generation to exclude from the idle budget.
                ws.send_json({"type": "input.text", "text": "unresolved"})
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Idle timeout: no message received",
                }

        speech_service._generate_audio_bytes.assert_not_awaited()
        speech_service.engine_client.abort.assert_not_awaited()

    def test_streaming_multiple_binary_frames(self, mocker: MockerFixture):
        captured_requests = []
        captured_tts_params = []

        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_prepare_speech_generation(request, request_id=None):
            captured_requests.append(request)
            return request_id or "req-stream", object(), {"_qwen3_tts_effective_max_tokens": [192]}

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
                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 0,
                }

    def test_word_timestamps_emit_pipeline_json_frame(self, mocker: MockerFixture):
        captured_requests = []
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = True

        async def mock_prepare_speech_generation(request, request_id=None):
            captured_requests.append(request)
            return request_id or "req-stream", object(), {}

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
