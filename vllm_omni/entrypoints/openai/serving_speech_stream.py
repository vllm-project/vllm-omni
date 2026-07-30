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
from vllm_omni.utils.forced_aligner import ForcedAlignerLoadError
from vllm_omni.utils.forced_aligner import align as forced_align

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
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                if config.word_timestamps:
                    total_bytes = await self._stream_audio_with_alignments(
                        websocket=websocket,
                        request_id=request_id,
                        generator=generator,
                        sentence_text=sentence_text,
                        utterance_index=utterance_index,
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
            logger.error(
                "Generation failed for utterance %d, sentence %d: %s",
                utterance_index,
                sentence_index,
                e,
            )
            await self._send_error(
                websocket,
                f"Generation failed for utterance {utterance_index}, sentence {sentence_index}: {e}",
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
