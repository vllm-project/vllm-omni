"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

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

    Server -> Client (when word_timestamps=true, issue #3631):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "pcm",
         "word_timestamps": true}
        {"type": "audio.chunk", "sentence_index": 0, "chunk_id": 0,
         "sample_rate": 24000,
         "audio_b64": "<base64 PCM>",
         "timestamps": [{"word": "...", "start_ms": ..., "end_ms": ...}, ...] | null}
        {"type": "audio.done", "sentence_index": 0}
        ...

    Notes on the timestamps path:
      - The server keeps an in-memory sentence audio prefix and reruns
        alignment as each PCM chunk arrives. Each frame carries newly
        stable timestamps only. Offsets are sentence-relative.
      - timestamps: []   -> aligner ran successfully but produced no
                            tokens (silence / pause chunk).
      - timestamps: null -> aligner failed for this chunk (decode error,
                            timeout, model unavailable). Audio is always
                            sent regardless.
"""

import asyncio
import base64
import json
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import (
    SPLIT_CLAUSE,
    SPLIT_SENTENCE,
    SentenceSplitter,
)
from vllm_omni.utils.forced_aligner import align as forced_align

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_ALIGNMENT_STABILITY_GUARD_MS = 240


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the existing OmniOpenAIServingSpeech pipeline.

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
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
            splitter = SentenceSplitter(boundary_re=boundary_re)
            sentence_index = 0

            # 2. Receive text chunks until input.done
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    return

                if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                    await self._send_error(websocket, "input.text message too large")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(websocket, "WebSocket messages must be JSON objects")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    sentences = splitter.add_text(text)
                    for sentence in sentences:
                        await self._generate_and_send(websocket, config, sentence, sentence_index)
                        sentence_index += 1

                elif msg_type == "input.done":
                    # Flush remaining buffer
                    remaining = splitter.flush()
                    if remaining:
                        await self._generate_and_send(websocket, config, remaining, sentence_index)
                        sentence_index += 1

                    # Send session.done
                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "total_sentences": sentence_index,
                        }
                    )
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                logger.debug("Failed to send error to streaming speech client", exc_info=True)

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
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

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        # Word-timestamps preconditions (issue #3631). Reject early so
        # the client gets a clear server-side reason rather than a
        # silent fallback or weird mode interaction.
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
    ) -> int:
        """Stream PCM as JSON ``audio.chunk`` frames carrying alignment.

        Flow: keep a sentence-level PCM prefix in memory, align that prefix
        as new chunks arrive, and send the previous PCM chunk with any newly
        stable word timestamps. This adds at most one-chunk delay while
        avoiding the larger latency of waiting for the whole sentence.

        Failure handling preserves the contract that audio always flows:
        an alignment failure surfaces as ``timestamps: null``;
        the empty-tokens case (silence chunk) surfaces as
        ``timestamps: []`` so clients can tell the two apart.
        """
        aligner_config = self._speech_service.forced_aligner_config
        assert aligner_config is not None  # gated by the precondition check

        prefix_audio = bytearray()
        pending_chunk: bytes | None = None
        pending_sample_rate = _PCM_SAMPLE_RATE
        total_bytes = 0
        sample_rate = _PCM_SAMPLE_RATE
        chunk_id = 0
        emitted_timestamp_count = 0

        async def align_prefix(*, final: bool) -> tuple[list[dict] | None, int]:
            aligned = await forced_align(
                audio=bytes(prefix_audio),
                text=sentence_text,
                sample_rate=sample_rate,
                config=aligner_config,
            )
            if aligned is None:
                return None, emitted_timestamp_count

            stable_until_ms = None
            if not final:
                prefix_duration_ms = (len(prefix_audio) / 2 / sample_rate) * 1000.0
                stable_until_ms = max(0.0, prefix_duration_ms - _ALIGNMENT_STABILITY_GUARD_MS)

            timestamps_payload: list[dict] = []
            next_emitted = emitted_timestamp_count
            for ts in aligned[emitted_timestamp_count:]:
                if stable_until_ms is not None and ts.end_ms > stable_until_ms:
                    break
                timestamps_payload.append(
                    {
                        "word": ts.word,
                        "start_ms": ts.start_ms,
                        "end_ms": ts.end_ms,
                        "confidence": ts.confidence,
                    }
                )
                next_emitted += 1
            return timestamps_payload, next_emitted

        async def send_chunk(chunk: bytes, chunk_sample_rate: int, timestamps_payload: list[dict] | None) -> None:
            nonlocal chunk_id
            await websocket.send_json(
                {
                    "type": "audio.chunk",
                    "sentence_index": sentence_index,
                    "chunk_id": chunk_id,
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
                prefix_audio.extend(chunk)
                total_bytes += len(chunk)

                if pending_chunk is not None:
                    timestamps_payload, emitted_timestamp_count = await align_prefix(final=False)
                    await send_chunk(pending_chunk, pending_sample_rate, timestamps_payload)

                pending_chunk = chunk
                pending_sample_rate = chunk_sample_rate

        if pending_chunk is not None:
            timestamps_payload, emitted_timestamp_count = await align_prefix(final=True)
            await send_chunk(pending_chunk, pending_sample_rate, timestamps_payload)

        return total_bytes

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
