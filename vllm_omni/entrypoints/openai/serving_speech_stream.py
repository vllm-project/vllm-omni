"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
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

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_MAX_BUFFER_SIZE = 100_000  # max accumulated text chars for token-level mode


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

            # Route to token-level handler if requested.
            if config.streaming_mode == "token_level":
                server_enabled = getattr(
                    self._speech_service.engine_client.model_config,
                    "streaming_text_enabled",
                    False,
                )
                if not server_enabled:
                    await self._send_error(
                        websocket,
                        "token_level streaming is disabled on this server "
                        "(set streaming_text_enabled: true in stage config to enable)",
                    )
                    return
                await self._handle_token_level_session(websocket, config)
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

    async def _handle_token_level_session(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """True engine-level streaming: text arrives incrementally, audio
        generation starts immediately and runs concurrently.

        1. Collect minimum text buffer (MIN_INITIAL_CHARS)
        2. Submit initial TTS request to the engine (audio generation starts)
        3. Concurrently: read text from WebSocket -> extend_text messages to orchestrator
                         stream audio from engine -> send to WebSocket
        4. On input.done -> send text_finished signal
        """
        response_format = config.response_format or "pcm"
        all_text = ""
        input_done = False

        MIN_INITIAL_CHARS = 60
        while len(all_text) < MIN_INITIAL_CHARS:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self._idle_timeout,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout")
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                all_text += text
                if len(all_text) > _MAX_BUFFER_SIZE:
                    await self._send_error(websocket, "input.text buffer exceeded limit")
                    return
            elif msg_type == "input.done":
                input_done = True
                break

        if not all_text.strip():
            await websocket.send_json({"type": "session.done", "total_sentences": 0})
            return

        initial_request = OpenAICreateSpeechRequest(
            input=all_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=True,
            streaming_text_input=True,
            streaming_drain_max_steps=config.streaming_drain_max_steps,
        )
        request_id, generator, _ = await self._speech_service._prepare_speech_generation(
            initial_request,
        )

        start_payload: dict = {
            "type": "audio.start",
            "sentence_index": 0,
            "sentence_text": all_text[:80] + ("..." if len(all_text) > 80 else ""),
            "format": response_format,
        }
        if response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False

        _extend_count = 0
        _extend_chars_total = 0

        finished_sent = False
        input_error: str | None = None

        def _send_extend(new_text: str, finished: bool) -> None:
            nonlocal _extend_count, _extend_chars_total
            _extend_count += 1
            _extend_chars_total += len(new_text) if new_text else 0
            logger.info(
                "[WS][extend] req=%s chunk#%d text_len=%d finished=%s cumulative_chars=%d",
                request_id,
                _extend_count,
                len(new_text) if new_text else 0,
                finished,
                _extend_chars_total,
            )
            self._speech_service.engine_client.extend_streaming_text(
                request_id,
                new_text=new_text,
                finished=finished,
            )

        def _finish_text() -> None:
            nonlocal finished_sent
            if not finished_sent:
                finished_sent = True
                _send_extend("", finished=True)

        async def feed_text() -> None:
            nonlocal all_text, input_error
            if input_done:
                _finish_text()
                return
            try:
                text_chars_total = len(all_text)
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=self._idle_timeout,
                        )
                    except asyncio.TimeoutError:
                        break
                    if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                        input_error = "input.text message too large"
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("type")
                    if msg_type == "input.text":
                        new_text = msg.get("text", "")
                        if not isinstance(new_text, str):
                            input_error = "input.text requires a string value"
                            break
                        if not new_text:
                            continue
                        text_chars_total += len(new_text)
                        if text_chars_total > _MAX_BUFFER_SIZE:
                            input_error = "input.text buffer exceeded limit"
                            break
                        _send_extend(new_text, finished=False)
                    elif msg_type == "input.done":
                        break
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.debug("feed_text error", exc_info=True)
                input_error = "input.text streaming failed"
            if input_error is None:
                _finish_text()

        text_task = asyncio.create_task(feed_text())

        try:
            async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                async for chunk in stream:
                    total_bytes += len(chunk)
                    await websocket.send_bytes(chunk)
            if not text_task.done():
                text_task.cancel()
                try:
                    await text_task
                except asyncio.CancelledError:
                    pass
            if input_error is None:
                _finish_text()
        except WebSocketDisconnect:
            text_task.cancel()
            try:
                await self._speech_service.engine_client.abort(request_id)
            except Exception:
                pass
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Token-level generation failed: %s", e)
            await self._send_error(websocket, f"Generation failed: {e}")
        finally:
            if not text_task.done():
                text_task.cancel()
                try:
                    await text_task
                except asyncio.CancelledError:
                    pass
                except WebSocketDisconnect:
                    pass
            if input_error is not None:
                generation_failed = True
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    pass
                await self._send_error(websocket, input_error)
            try:
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
                        "total_sentences": 1,
                    }
                )
            except Exception:
                pass

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

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
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
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
