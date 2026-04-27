# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video streaming session manager and WebSocket handler.

Provides ``VideoStreamConfig``, ``VideoStreamSession`` (frame/audio buffer
with EVS filtering), and ``VideoStreamHandler`` (WebSocket session loop).
Builds standard ``ChatCompletionRequest`` so ``OmniOpenAIServingChat`` is
reused with zero changes.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_BYTES = 10 * 1024 * 1024  # 10 MB per frame


class VideoStreamConfig:
    """Per-session configuration sent by the client in ``session.config``."""

    __slots__ = (
        "model",
        "modalities",
        "max_frames",
        "num_sample_frames",
        "evs_enabled",
        "evs_threshold",
    )

    def __init__(
        self,
        model: str = "",
        modalities: list[str] | None = None,
        max_frames: int = 64,
        num_sample_frames: int = 16,
        evs_enabled: bool = True,
        evs_threshold: float = 0.95,
    ) -> None:
        self.model = model
        self.modalities = modalities if modalities is not None else ["text"]
        self.max_frames = max_frames
        self.num_sample_frames = num_sample_frames
        self.evs_enabled = evs_enabled
        self.evs_threshold = evs_threshold

    # JSON doesn't distinguish int/float, so float fields accept both.
    _FIELD_TYPES: dict[str, tuple[type, ...]] = {
        "model": (str,),
        "modalities": (list,),
        "max_frames": (int,),
        "num_sample_frames": (int,),
        "evs_enabled": (bool,),
        "evs_threshold": (int, float),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoStreamConfig:
        known = set(cls.__slots__)
        filtered = {k: v for k, v in data.items() if k in known}

        for field, accepted in cls._FIELD_TYPES.items():
            if field in filtered and not isinstance(filtered[field], accepted):
                names = "/".join(t.__name__ for t in accepted)
                raise TypeError(f"Invalid type for '{field}': expected {names}, got {type(filtered[field]).__name__}")

        return cls(**filtered)


class VideoStreamSession:
    """Manages frame and audio buffers for a single streaming session.

    Frames are optionally filtered through ``FrameSimilarityFilter`` (EVS)
    before being stored in a fixed-size ring buffer.  ``build_chat_request``
    merges buffers into a ``ChatCompletionRequest`` with ``image_url`` (+
    ``audio_url``) content blocks.
    """

    def __init__(self, config: VideoStreamConfig) -> None:
        self._config = config
        self._frame_filter: FrameSimilarityFilter | None = (
            FrameSimilarityFilter(threshold=config.evs_threshold) if config.evs_enabled else None
        )
        self._frames: deque[bytes] = deque(maxlen=config.max_frames)
        self._audio_chunks: list[bytes] = []

    def add_frame(self, jpeg_bytes: bytes) -> bool:
        """Add a JPEG frame after EVS filtering.  Returns ``True`` if kept."""
        if len(jpeg_bytes) > _MAX_FRAME_BYTES:
            raise ValueError(f"Frame too large: {len(jpeg_bytes)} bytes (limit {_MAX_FRAME_BYTES})")

        if self._frame_filter and not self._frame_filter.should_retain(jpeg_bytes):
            return False

        self._frames.append(jpeg_bytes)
        return True

    def sample_frames(self) -> list[bytes]:
        """Return up to ``num_sample_frames`` uniformly sampled frames."""
        n = len(self._frames)
        k = min(n, self._config.num_sample_frames)
        if k == 0:
            return []
        if k == 1:
            return [self._frames[-1]]
        if k >= n:
            return list(self._frames)
        indices = [int(i * (n - 1) / (k - 1)) for i in range(k)]
        return [self._frames[i] for i in indices]

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def add_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Append raw PCM 16 kHz audio bytes."""
        self._audio_chunks.append(pcm_bytes)

    def clear_audio(self) -> None:
        """Clear audio buffer after a query is submitted."""
        self._audio_chunks.clear()

    @property
    def has_audio(self) -> bool:
        return bool(self._audio_chunks)

    def build_chat_request(self, query_text: str) -> ChatCompletionRequest:
        """Build a ``ChatCompletionRequest`` from the current buffers."""
        sampled = self.sample_frames()
        content_parts: list[dict[str, Any]] = []

        for frame in sampled:
            frame_b64 = base64.b64encode(frame).decode()
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                    },
                }
            )

        if self.has_audio:
            combined_pcm = b"".join(self._audio_chunks)
            audio_b64 = base64.b64encode(combined_pcm).decode()
            content_parts.append(
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/L16;rate=16000;base64,{audio_b64}",
                    },
                }
            )

        content_parts.append({"type": "text", "text": query_text})

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content_parts},
        ]

        request_dict: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
        }

        if self.has_audio:
            request_dict["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }

        return ChatCompletionRequest(**request_dict)

    @property
    def evs_stats(self) -> dict[str, int | float] | None:
        return self._frame_filter.stats if self._frame_filter else None


class VideoStreamHandler:
    """Drives a ``VideoStreamSession`` over a FastAPI ``WebSocket``.

    Instantiate once at server startup and call ``handle_session`` for each
    incoming connection.  Uses two concurrent tasks (_reader + _processor)
    so frames keep arriving while a query is being processed.
    """

    def __init__(
        self,
        chat_handler: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._chat_handler = chat_handler
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for one WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            session = VideoStreamSession(config)
            logger.info(
                "Video stream session started: model=%s modalities=%s max_frames=%d evs=%s",
                config.model,
                config.modalities,
                config.max_frames,
                config.evs_enabled,
            )

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def _reader() -> None:
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=self._idle_timeout,
                            )
                        except asyncio.TimeoutError:
                            await self._send_error(websocket, "Idle timeout: no message received")
                            await msg_queue.put(None)
                            return

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            await self._send_error(websocket, "Invalid JSON")
                            continue

                        if not isinstance(msg, dict):
                            await self._send_error(websocket, "Messages must be JSON objects")
                            continue

                        await msg_queue.put(msg)

                        if msg.get("type") == "video.done":
                            return
                except WebSocketDisconnect:
                    await msg_queue.put(None)
                except Exception:
                    await msg_queue.put(None)
                    raise

            async def _processor() -> None:
                while True:
                    msg = await msg_queue.get()
                    if msg is None:
                        return

                    msg_type = msg.get("type")

                    if msg_type == "video.frame":
                        data_b64 = msg.get("data", "")
                        try:
                            jpeg_bytes = base64.b64decode(data_b64)
                        except Exception:
                            await self._send_error(websocket, "Invalid base64 in video.frame")
                            continue
                        try:
                            session.add_frame(jpeg_bytes)
                        except ValueError as exc:
                            await self._send_error(websocket, str(exc))
                            continue

                    elif msg_type == "audio.chunk":
                        data_b64 = msg.get("data", "")
                        try:
                            pcm_bytes = base64.b64decode(data_b64)
                        except Exception:
                            await self._send_error(websocket, "Invalid base64 in audio.chunk")
                            continue
                        session.add_audio_chunk(pcm_bytes)

                    elif msg_type == "video.query":
                        query_text = msg.get("text", "")
                        if not query_text:
                            await self._send_error(
                                websocket,
                                "video.query requires a non-empty 'text' field",
                            )
                            continue
                        await self._handle_query(websocket, session, query_text)

                    elif msg_type == "video.done":
                        evs = session.evs_stats
                        if evs is not None:
                            await websocket.send_json({"type": "response.evs_stats", **evs})
                        await websocket.send_json({"type": "session.done"})
                        return

                    else:
                        await self._send_error(websocket, f"Unknown message type: {msg_type}")

            reader_task = asyncio.create_task(_reader())
            try:
                await _processor()
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug(
                        "Reader task raised an exception",
                        exc_info=True,
                    )

        except WebSocketDisconnect:
            logger.info("Video stream: client disconnected")
        except Exception:
            logger.exception("Video stream session error")
            try:
                await self._send_error(websocket, "Internal server error")
            except Exception:
                pass

    async def _receive_config(self, websocket: WebSocket) -> VideoStreamConfig | None:
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        try:
            return VideoStreamConfig.from_dict(msg)
        except TypeError as exc:
            await self._send_error(websocket, str(exc))
            return None

    async def _handle_query(
        self,
        websocket: WebSocket,
        session: VideoStreamSession,
        query_text: str,
    ) -> None:
        """Build a ChatCompletionRequest, run it, and stream the response."""
        if session.frame_count == 0:
            await self._send_error(
                websocket,
                "No frames available. Send video.frame before video.query.",
            )
            return

        request = session.build_chat_request(query_text)
        await websocket.send_json({"type": "response.start"})

        # After response.start, the protocol contract requires us to always
        # send response.text.done — even on error / exception paths.
        text_parts: list[str] = []
        try:
            # raw_request=None: serving_chat.py guards with `if raw_request:`
            generator = await self._chat_handler.create_chat_completion(request, raw_request=None)

            if isinstance(generator, ErrorResponse):
                error_msg = generator.error.message if generator.error else "Unknown error"
                await self._send_error(websocket, error_msg)
            else:
                async for chunk in generator:
                    if isinstance(chunk, str):
                        for delta in self._parse_sse_deltas(chunk):
                            text_parts.append(delta)
                            await websocket.send_json(
                                {
                                    "type": "response.text.delta",
                                    "delta": delta,
                                }
                            )

        except Exception:
            logger.exception("Query failed")
            await self._send_error(websocket, "Query processing failed")

        full_text = "".join(text_parts)
        await websocket.send_json({"type": "response.text.done", "text": full_text})

        session.clear_audio()

    @staticmethod
    def _parse_sse_deltas(chunk: str) -> list[str]:
        """Extract content deltas from SSE-formatted chunk."""
        deltas: list[str] = []
        for line in chunk.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                data = json.loads(line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    deltas.append(delta)
            except (json.JSONDecodeError, IndexError, AttributeError):
                pass
        return deltas

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
