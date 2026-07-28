"""WebSocket handler for streaming text input TTS.

Accepts buffered or resumable text input. Resumable modes feed fragments
through the engine's native streaming-input path so one request retains state.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client (default, word_timestamps=false):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        ...
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}

    Server -> Client (when word_timestamps=true):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "pcm"}
        {"type": "audio.chunk", "sentence_index": 0, "chunk_id": 0, "audio_b64": "<base64 PCM>", "timestamps": null}
        ...
        {"type": "audio.chunk", "audio_b64": "", "timestamps": [{"word", "start_ms", "end_ms"}, ...]}
        {"type": "audio.done", "sentence_index": 0}
        # Audio is JSON base64 PCM (not binary). A trailing empty-audio chunk carries the
        # full sentence-relative alignment. timestamps: list = aligned, [] = silence, null = failed.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechInputCommit,
    StreamingSpeechInputCommitted,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.utils.forced_aligner import ForcedAlignerLoadError
from vllm_omni.utils.forced_aligner import align as forced_align

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_BYTES_PER_SAMPLE = 2  # 16-bit mono PCM
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_MAX_BUFFER_SIZE = 100_000


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    A connection may carry multiple sequential sessions. Each session starts
    with ``session.config`` and ends with ``session.done``.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Serve sequential speech sessions on one WebSocket connection."""
        await websocket.accept()
        first_session = True

        while True:
            try:
                config = await self._receive_config(
                    websocket,
                    timeout=self._config_timeout if first_session else None,
                )
                if config is None:
                    return
                first_session = False

                if config.model and hasattr(self._speech_service, "_check_model"):
                    error = await self._speech_service._check_model(
                        OpenAICreateSpeechRequest(input="ping", model=config.model)
                    )
                    if error is not None:
                        await self._send_error(websocket, str(error))
                        return

                if config.streaming_mode == "sentence":
                    await self._handle_buffered_session(websocket, config)
                else:
                    await self._handle_resumable_session(websocket, config)
            except WebSocketDisconnect:
                logger.info("Streaming speech: client disconnected")
                return
            except Exception as e:
                logger.exception("Streaming speech session error: %s", e)
                try:
                    await self._send_error(websocket, f"Internal error: {e}")
                except Exception:
                    logger.debug("Failed to send error to streaming speech client", exc_info=True)
                return

    async def _receive_config(
        self,
        websocket: WebSocket,
        *,
        timeout: float | None,
    ) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            if timeout is None:
                raw = await websocket.receive_text()
            else:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
            await self._send_error(websocket, "session.config message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "session.config must be a JSON object")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _receive_input_message(self, websocket: WebSocket) -> dict:
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._idle_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Idle timeout: no message received") from exc

        if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
            await self._send_error(websocket, "input.text message too large")
            return {}
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
            return {}
        if not isinstance(msg, dict):
            await self._send_error(websocket, "WebSocket messages must be JSON objects")
            return {}
        return msg

    def _build_resumable_request(
        self,
        config: StreamingSpeechSessionConfig,
        text: str,
    ) -> OpenAICreateSpeechRequest:
        return OpenAICreateSpeechRequest(
            input=text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format="pcm",
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            non_streaming_mode=False,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=True,
        )

    async def _iter_token_level_requests(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> AsyncGenerator[OpenAICreateSpeechRequest, None]:
        total_chars = 0
        while True:
            msg = await self._receive_input_message(websocket)
            if not msg:
                continue
            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                if not text:
                    continue
                total_chars += len(text)
                if total_chars > _MAX_BUFFER_SIZE:
                    raise ValueError("input.text buffer exceeded limit")
                yield self._build_resumable_request(config, text)
            elif msg_type == "input.done":
                return
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def _iter_sentence_commit_requests(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> AsyncGenerator[OpenAICreateSpeechRequest, None]:
        pending_parts: list[str] = []
        total_chars = 0
        sentence_index = 0
        while True:
            msg = await self._receive_input_message(websocket)
            if not msg:
                continue
            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                total_chars += len(text)
                if total_chars > _MAX_BUFFER_SIZE:
                    raise ValueError("input.text buffer exceeded limit")
                pending_parts.append(text)
                continue
            if msg_type not in ("input.commit", "input.done"):
                await self._send_error(websocket, f"Unknown message type: {msg_type}")
                continue

            text = "".join(pending_parts)
            if not text.strip():
                if msg_type == "input.done":
                    return
                await self._send_error(websocket, "input.commit requires buffered text")
                continue

            if msg_type == "input.commit":
                try:
                    commit = StreamingSpeechInputCommit.model_validate(msg)
                except ValidationError as exc:
                    await self._send_error(websocket, f"Invalid input.commit: {exc}")
                    continue
            else:
                commit = StreamingSpeechInputCommit(type="input.commit")

            pending_parts.clear()
            yield self._build_resumable_request(config, text)
            await websocket.send_json(
                StreamingSpeechInputCommitted(
                    commit_id=commit.commit_id,
                    sentence_index=sentence_index,
                    chars_committed=len(text),
                ).model_dump()
            )
            sentence_index += 1
            if msg_type == "input.done":
                return

    async def _handle_buffered_session(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        text_parts: list[str] = []
        while True:
            msg = await self._receive_input_message(websocket)
            if not msg:
                continue
            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                text_parts.append(text)
            elif msg_type == "input.done":
                full_text = "".join(text_parts).strip()
                total_sentences = 0
                if full_text:
                    await self._generate_and_send(websocket, config, full_text, 0)
                    total_sentences = 1
                await websocket.send_json(
                    {
                        "type": "session.done",
                        "total_sentences": total_sentences,
                    }
                )
                return
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def _handle_resumable_session(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        if config.streaming_mode == "sentence_commit":
            requests = self._iter_sentence_commit_requests(websocket, config)
        else:
            requests = self._iter_token_level_requests(websocket, config)

        try:
            initial_request = await anext(requests)
        except StopAsyncIteration:
            await websocket.send_json({"type": "session.done", "total_sentences": 0})
            return

        segment_count = 1

        async def remaining_requests() -> AsyncGenerator[OpenAICreateSpeechRequest, None]:
            nonlocal segment_count
            async for request in requests:
                segment_count += 1
                yield request

        try:
            request_id, generator, _ = await self._speech_service._prepare_resumable_speech_generation(
                initial_request,
                remaining_requests(),
            )
        except Exception:
            await requests.aclose()
            raise
        await websocket.send_json(
            {
                "type": "audio.start",
                "sentence_index": 0,
                "sentence_text": initial_request.input[:80] + ("..." if len(initial_request.input) > 80 else ""),
                "format": "pcm",
                "sample_rate": _PCM_SAMPLE_RATE,
            }
        )

        total_bytes = 0
        generation_failed = False
        try:
            async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                async for chunk in stream:
                    total_bytes += len(chunk)
                    await websocket.send_bytes(chunk)
        except WebSocketDisconnect:
            await self._speech_service.engine_client.abort(request_id)
            raise
        except Exception as exc:
            generation_failed = True
            logger.exception("Resumable speech generation failed: %s", exc)
            try:
                await self._speech_service.engine_client.abort(request_id)
            except Exception:
                pass
            await self._send_error(websocket, f"Generation failed: {exc}")
        await websocket.send_json(
            {
                "type": "audio.done",
                "sentence_index": 0,
                "total_bytes": total_bytes,
                "error": generation_failed,
            }
        )
        await websocket.send_json(
            {
                "type": "session.done",
                "total_sentences": segment_count,
            }
        )

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        # Reject unmet word-timestamps preconditions early with a clear reason.
        if config.word_timestamps:
            if self._speech_service.forced_aligner_config is None:
                await self._send_error(
                    websocket,
                    "word_timestamps=true but the server was launched without "
                    "--forced-aligner; either restart the server with that flag "
                    "or set word_timestamps=false in session.config.",
                )
                return
            if not (config.stream_audio and response_format == "pcm"):
                await self._send_error(
                    websocket,
                    "word_timestamps=true requires stream_audio=true and "
                    "response_format='pcm' (the aligner consumes raw PCM).",
                )
                return

        request = OpenAICreateSpeechRequest(
            input=sentence_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            non_streaming_mode=config.non_streaming_mode,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=config.stream_audio,
            word_timestamps=config.word_timestamps,
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            # Nominal stream rate; each audio.chunk carries the authoritative
            # per-chunk sample_rate.
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        if config.word_timestamps:
            start_payload["word_timestamps"] = True
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                if config.word_timestamps:
                    total_bytes = await self._stream_audio_with_alignments(
                        websocket=websocket,
                        request_id=request_id,
                        generator=generator,
                        sentence_text=sentence_text,
                        sentence_index=sentence_index,
                        language=config.language,
                    )
                else:
                    async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                        async for chunk in stream:
                            total_bytes += len(chunk)
                            await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = await self._speech_service._generate_audio_bytes(request)
                total_bytes = len(audio_bytes)
                await websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort streaming speech request %s", request_id, exc_info=True)
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(websocket, f"Generation failed for sentence {sentence_index}: {e}")
        finally:
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)

    async def _stream_audio_with_alignments(
        self,
        *,
        websocket: WebSocket,
        request_id: str,
        generator,
        sentence_text: str,
        sentence_index: int,
        language: str | None = None,
    ) -> int:
        """Stream PCM as JSON ``audio.chunk`` frames, aligned per sentence.

        Forward each PCM chunk live (``timestamps: null``) while buffering the
        sentence audio, then run the forced aligner once over the whole
        sentence and emit a final empty-audio ``audio.chunk`` with the word
        timestamps. On aligner failure timestamps is ``null``; for silence it
        is ``[]`` (audio always flows regardless).
        """
        aligner_config = self._speech_service.forced_aligner_config
        assert aligner_config is not None  # gated by the precondition check

        sentence_audio = bytearray()
        total_bytes = 0
        sample_rate = _PCM_SAMPLE_RATE
        chunk_id = 0

        async def send_chunk(
            chunk: bytes,
            chunk_sample_rate: int,
            timestamps_payload: list[dict] | None,
            chunk_start_ms: int,
            chunk_end_ms: int,
        ) -> None:
            nonlocal chunk_id
            await websocket.send_json(
                {
                    "type": "audio.chunk",
                    "sentence_index": sentence_index,
                    "chunk_id": chunk_id,
                    "chunk_start_ms": chunk_start_ms,
                    "chunk_end_ms": chunk_end_ms,
                    "sample_rate": chunk_sample_rate,
                    "audio_b64": base64.b64encode(chunk).decode("ascii"),
                    "timestamps": timestamps_payload,
                }
            )
            chunk_id += 1

        async with aclosing(
            self._speech_service._generate_pcm_chunks(
                generator,
                request_id,
                include_sample_rate=True,
            )
        ) as stream:
            async for chunk, chunk_sample_rate in stream:
                sample_rate = chunk_sample_rate
                chunk_start_ms = int(round((len(sentence_audio) / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                sentence_audio.extend(chunk)
                chunk_end_ms = int(round((len(sentence_audio) / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                total_bytes += len(chunk)
                # Audio first, timestamps after the whole sentence is aligned.
                await send_chunk(chunk, chunk_sample_rate, None, chunk_start_ms, chunk_end_ms)

        # Single alignment pass over the full sentence, then emit timestamps.
        # A load/config failure is permanent, so surface the reason once; audio
        # has already streamed, so the trailing frame still carries null.
        try:
            timestamps_payload = await self._align_sentence(
                audio=bytes(sentence_audio),
                text=sentence_text,
                sample_rate=sample_rate,
                config=aligner_config,
                language=language,
            )
        except ForcedAlignerLoadError as exc:
            await self._send_error(websocket, f"forced aligner unavailable: {exc}")
            timestamps_payload = None
        sentence_end_ms = int(round((len(sentence_audio) / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
        await send_chunk(b"", sample_rate, timestamps_payload, 0, sentence_end_ms)

        return total_bytes

    @staticmethod
    async def _align_sentence(
        *,
        audio: bytes,
        text: str,
        sample_rate: int,
        config,
        language: str | None = None,
    ) -> list[dict] | None:
        """Convert a sentence alignment into JSON word-timestamp dicts.

        Returns ``None`` on aligner failure, ``[]`` when it ran but produced no
        tokens. Monotonic, non-overlapping bounds are guaranteed by the decoder.
        ``language`` is forwarded to word segmentation.
        """
        aligned = await forced_align(
            audio=audio,
            text=text,
            sample_rate=sample_rate,
            config=config,
            language=language,
        )
        if aligned is None:
            return None
        return [{"word": ts.word, "start_ms": ts.start_ms, "end_ms": ts.end_ms} for ts in aligned]

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed; safe to ignore
