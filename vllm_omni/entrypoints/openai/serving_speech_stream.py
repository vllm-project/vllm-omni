# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers it until input.done, and
generates audio once for the buffered input using the existing TTS pipeline.

input.done is a flush, not a close: it ends the current utterance and the
connection stays open, so the next utterance reuses the same connection
instead of paying another WebSocket handshake. A connection ends on
session.close, on the idle timeout, or when the client closes the socket.
The session config is sticky across flushes and can be replaced by sending
another session.config between utterances.

"Utterance" here names the flush unit rather than any linguistic unit: it is
whatever text the client had buffered when it sent input.done, from a word to
several paragraphs, synthesized as a single request.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (first message; repeatable)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of utterance, flush and keep connection open
        {"type": "session.close"}         # End of connection

    Server -> Client (default, word_timestamps=false):
        {"type": "audio.start", "utterance_index": 0, "sentence_index": 0,
         "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        ...
        {"type": "audio.done", "utterance_index": 0, "sentence_index": 0}
        {"type": "session.done", "utterance_index": 0, "total_sentences": N}
        {"type": "error", "message": "..."}
        # session.done ends the flushed utterance, not the connection. An
        # utterance is just the flush unit: whatever text was buffered when
        # input.done arrived, of any length. utterance_index counts those
        # flushes across the connection, while sentence_index counts within
        # one of them and so pairs with total_sentences.

    Server -> Client (when word_timestamps=true):
        {"type": "audio.start", "utterance_index": 0, "sentence_index": 0,
         "sentence_text": "...", "format": "pcm"}
        {"type": "audio.chunk", "utterance_index": 0, "sentence_index": 0, "chunk_id": 0,
         "audio_b64": "<base64 PCM>", "timestamps": null}
        ...
        {"type": "audio.chunk", "audio_b64": "", "timestamps": [{"word", "start_ms", "end_ms"}, ...]}
        {"type": "audio.done", "utterance_index": 0, "sentence_index": 0}
        # Audio is JSON base64 PCM (not binary). A trailing empty-audio chunk carries the
        # full sentence-relative alignment. timestamps: list = aligned, [] = silence, null = failed.
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
from vllm_omni.utils.forced_aligner import extract_word_timestamps

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_BYTES_PER_SAMPLE = 2  # 16-bit mono PCM
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    A connection carries one or more utterances. Text arrives incrementally,
    is buffered until input.done, and audio is generated once for the
    buffered input using the existing OmniOpenAIServingSpeech pipeline. The
    connection outlives each utterance so a client can keep synthesizing on
    it without reconnecting.

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
        """Main loop for a single WebSocket connection.

        Serves any number of utterances: text is buffered until input.done,
        which flushes it as one TTS request and then leaves the connection
        open for the next one. Rejecting a message is only fatal before the
        first valid session.config; afterwards the error is reported and the
        connection survives.
        """
        await websocket.accept()

        config: StreamingSpeechSessionConfig | None = None
        text_parts: list[str] = []
        utterance_index = 0

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._config_timeout if config is None else self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    if config is None:
                        await self._send_error(websocket, "Timeout waiting for session.config")
                    else:
                        await self._send_error(websocket, "Idle timeout: no message received")
                    return

                msg = await self._parse_message(websocket, raw)
                if msg is None:
                    if config is None:
                        return  # Malformed handshake, connection closing
                    continue

                msg_type = msg.get("type")

                if msg_type == "session.config":
                    if text_parts:
                        await self._send_error(
                            websocket,
                            "session.config cannot be applied while input is buffered; send input.done first",
                        )
                        continue
                    new_config = await self._build_config(websocket, msg)
                    if new_config is None:
                        if config is None:
                            return  # Error already sent, connection closing
                        continue  # Keep serving with the previous config
                    config = new_config

                elif config is None:
                    await self._send_error(
                        websocket,
                        f"Expected session.config, got: {msg_type}",
                    )
                    return

                elif msg_type == "input.text":
                    text = msg.get("text", "")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    text_parts.append(text)

                elif msg_type == "input.done":
                    full_text = "".join(text_parts).strip()
                    text_parts.clear()
                    total_sentences = 0
                    if full_text:
                        # However long the buffered text is, the pipeline takes
                        # it as one request, so every flush is sentence 0 of 1.
                        await self._generate_and_send(
                            websocket,
                            config,
                            full_text,
                            utterance_index=utterance_index,
                            sentence_index=0,
                        )
                        total_sentences = 1

                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "utterance_index": utterance_index,
                            "total_sentences": total_sentences,
                        }
                    )
                    utterance_index += 1

                elif msg_type == "session.close":
                    await websocket.close()
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

    async def _parse_message(self, websocket: WebSocket, raw: str) -> dict | None:
        """Decode one client message, or report why it was rejected.

        Size limits are per message type: session.config carries ref_audio
        payloads and gets the larger budget.
        """
        if len(raw) > max(_MAX_CONFIG_MESSAGE_SIZE, _MAX_INPUT_TEXT_MESSAGE_SIZE):
            await self._send_error(websocket, "WebSocket message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "WebSocket messages must be JSON objects")
            return None

        if msg.get("type") == "session.config":
            if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
                await self._send_error(websocket, "session.config message too large")
                return None
        elif len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
            await self._send_error(websocket, "input.text message too large")
            return None

        return msg

    async def _build_config(self, websocket: WebSocket, msg: dict) -> StreamingSpeechSessionConfig | None:
        """Validate a session.config message and its model."""
        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        if config.model and hasattr(self._speech_service, "_check_model"):
            error = await self._speech_service._check_model(OpenAICreateSpeechRequest(input="ping", model=config.model))
            if error is not None:
                await self._send_error(websocket, str(error))
                return None

        return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        *,
        utterance_index: int,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket.

        ``utterance_index`` identifies the flush this sentence belongs to and
        ``sentence_index`` its position inside that flush.
        """
        response_format = config.response_format or "wav"

        # Reject unmet word-timestamps preconditions early with a clear reason.
        if config.word_timestamps:
            if not self._speech_service.forced_aligner_enabled:
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
                    "response_format='pcm' (timestamps ride the per-sentence "
                    "PCM audio.chunk stream).",
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
            "utterance_index": utterance_index,
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
                request_id, generator, tts_params = await self._speech_service._prepare_speech_generation(request)
                if config.word_timestamps:
                    total_bytes = await self._stream_audio_with_alignments(
                        websocket=websocket,
                        request_id=request_id,
                        generator=generator,
                        sentence_text=sentence_text,
                        utterance_index=utterance_index,
                        sentence_index=sentence_index,
                        language=config.language,
                        tts_params=tts_params,
                    )
                else:
                    async with aclosing(
                        self._speech_service._generate_pcm_chunks(
                            generator,
                            request_id,
                            tts_params=tts_params,
                        )
                    ) as stream:
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
            logger.error(
                "Generation failed for utterance %d, sentence %d: %s",
                utterance_index,
                sentence_index,
                e,
            )
            await self._send_error(
                websocket,
                f"Generation failed for utterance {utterance_index}, sentence {sentence_index}: {e}",
                partial_audio=total_bytes > 0,
            )
        finally:
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "utterance_index": utterance_index,
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
        utterance_index: int,
        sentence_index: int,
        language: str | None = None,
        tts_params: dict | None = None,
    ) -> int:
        """Stream PCM as JSON ``audio.chunk`` frames, aligned per sentence.

        Forward each PCM chunk live (``timestamps: null``). The forced-aligner
        pipeline stage (appended when the server is launched with
        ``--forced-aligner``) consumes the synthesized audio internally and its
        pooling output rides the same generator, so once the audio finishes we
        pull the word timestamps straight off that aligner output and emit a
        final empty-audio ``audio.chunk`` carrying them. Timestamps is ``null``
        when the aligner produced none; audio always flows regardless.
        """
        audio_bytes_seen = 0
        total_bytes = 0
        sample_rate = _PCM_SAMPLE_RATE
        chunk_id = 0
        # Receives the aligner stage's pooling output from the generator.
        collect: dict = {}

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
                    "utterance_index": utterance_index,
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
                tts_params=tts_params,
                collect=collect,
            )
        ) as stream:
            async for chunk, chunk_sample_rate in stream:
                sample_rate = chunk_sample_rate
                chunk_start_ms = int(round((audio_bytes_seen / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                audio_bytes_seen += len(chunk)
                chunk_end_ms = int(round((audio_bytes_seen / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                total_bytes += len(chunk)
                # Audio first, timestamps after the whole sentence is aligned.
                await send_chunk(chunk, chunk_sample_rate, None, chunk_start_ms, chunk_end_ms)

        # Pull word timestamps off the aligner stage's pooling output (it rode
        # the same generator); extract_word_timestamps re-segments the sentence
        # text for the word strings when the aligner output doesn't carry them.
        aligner_res = collect.get("aligner_res")
        timestamps_payload = (
            extract_word_timestamps(aligner_res, sentence_text, language) if aligner_res is not None else None
        )
        sentence_end_ms = int(round((audio_bytes_seen / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
        await send_chunk(b"", sample_rate, timestamps_payload, 0, sentence_end_ms)

        return total_bytes

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str, *, partial_audio: bool = False) -> None:
        """Send an error message to the client."""
        try:
            payload: dict[str, object] = {
                "type": "error",
                "message": message,
            }
            if partial_audio:
                payload.update(partial_audio=True, action="discard")
            await websocket.send_json(payload)
        except Exception:
            pass  # Connection may already be closed; safe to ignore
