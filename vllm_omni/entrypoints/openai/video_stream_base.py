# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Base WebSocket handler for streaming video input understanding.

Shared session loop, frame/audio buffering, EVS pre-filter, prewarm,
interrupt handling, and engine ``generate()`` streaming. Pipeline-specific
behavior (trigger rules, prompt shape, history) is supplied by subclasses
via :class:`VideoStreamPipelineHooks`.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "...", "frame_id": "...", "pts_ms": 0}
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "video.frame.ack", ...}          # when frame_id is provided
        {"type": "video.frames.consumed", ...}    # after first engine output
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import hashlib
import io
import json
import time as _time
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai import video_stream_envs
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter
from vllm_omni.entrypoints.openai.video_stream_context import (
    text_only_message,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_BUFFER_FRAMES = 64
_MAX_AUDIO_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_MSG_QUEUE = 200
_CODEC_FRAME_SAMPLES = 1920  # CausalConv leading-edge artifact length
_BAD_FRAME = object()


def _decode_frame_bytes(raw_bytes: bytes) -> Any:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _stage0_prefix_caching_enabled(engine_client: Any) -> bool:
    """Read stage-0 (thinker) prefix caching from the engine. Called once
    at handler init: incremental prefill is a server-side capability, not a
    per-request choice."""
    engine = getattr(engine_client, "engine", None)
    stage_vllm_configs = getattr(engine, "stage_vllm_configs", None)
    stage0 = stage_vllm_configs[0] if stage_vllm_configs else None
    cache_config = getattr(stage0, "cache_config", None)
    return bool(getattr(cache_config, "enable_prefix_caching", False))


@runtime_checkable
class VideoStreamPipelineHooks(Protocol):
    """Pipeline-specific hooks for streaming video handlers."""

    def should_trigger_turn(self, trigger: "VideoStreamTurnTrigger") -> bool:
        """Return True to auto-start a turn after a new frame (no ``video.query``)."""
        ...

    def build_engine_prompt(
        self,
        config: "StreamingVideoSessionConfig",
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build OpenAI-style messages and the current user message."""
        ...

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        """Update session state after a successful turn."""
        ...

    def build_engine_prompt_prefix(
        self,
        config: "StreamingVideoSessionConfig",
        frame_buffer: list[str],
        message_history: list[dict[str, Any]],
        prewarmed_frames: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Prefix of the next :meth:`build_engine_prompt` through the last vision token.

        History plus arriving video frames, without query text or input audio.
        Return None when there are no usable frames (or to disable incremental prefill).
        """
        ...


@dataclass(frozen=True)
class VideoStreamTurnTrigger:
    """Snapshot passed to :meth:`OmniStreamingVideoHandler.should_trigger_turn`."""

    frame_count: int
    is_generating: bool
    config: "StreamingVideoSessionConfig"


class StreamingVideoSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message."""

    model: str | None = None
    modalities: list[str] = Field(
        default_factory=lambda: ["text", "audio"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    num_frames: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Max frames to sample from buffer for the model.",
    )
    max_frames: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Max frames to keep in the buffer.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt.",
    )
    use_audio_in_video: bool = Field(
        default=True,
        description=(
            "Pass use_audio_in_video to the processor when this is true and "
            "(incremental prefill is active, or this query has input audio)."
        ),
    )
    sampling_params_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-stage sampling params [thinker, talker, code2wav].",
    )
    enable_frame_filter: bool = Field(
        default=True,
        description="EVS pixel-similarity pre-filter to drop near-duplicate frames.",
    )
    frame_filter_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="EVS similarity threshold (higher = keep more frames).",
    )


class OmniStreamingVideoHandler:
    """Base handler for WebSocket streaming video sessions.

    Subclasses implement :class:`VideoStreamPipelineHooks` to customize turn
    triggering, prompt construction, and history updates.
    """

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        """Auto-trigger after ``video.frame`` when True (default: never)."""
        return False

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _build_frame_image_parts(
        frames: list[str],
        prewarmed_frames: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Decoded frame parts. Prefix and query both wait for PIL prewarm so
        every frame is ``image_pil`` plus the arrival-time uuid."""
        prewarmed = prewarmed_frames or {}
        parts: list[dict[str, Any]] = []
        for frame_b64 in frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME or cached is None:
                continue
            pil, mm_uuid = cached
            parts.append(
                {
                    "type": "image_pil",
                    "image_pil": pil,
                    "uuid": mm_uuid,
                }
            )
        return parts

    def _history_prefix_messages(
        self,
        config: StreamingVideoSessionConfig,
        message_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Shared prompt prefix (system + compressed text-only history).

        :meth:`build_engine_prompt_prefix` and :meth:`build_engine_prompt`
        MUST build this identically: any divergence before the frame parts
        voids the warmed KV from that token onward.
        """
        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        recent_history = message_history[-2:] if len(message_history) > 2 else message_history
        for hist_msg in recent_history:
            messages.append(self._text_only_message(hist_msg))
        return messages

    def build_engine_prompt_prefix(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        message_history: list[dict[str, Any]],
        prewarmed_frames: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Prefix of the next :meth:`build_engine_prompt` through the last vision token.

        History plus arriving video frames, without query text or input audio.
        A mismatch voids the warmed KV from the first diverging token.
        Return None when there are no usable frames (or override to disable).
        """
        frame_parts = self._build_frame_image_parts(frame_buffer, prewarmed_frames)
        if not frame_parts:
            return None

        messages = self._history_prefix_messages(config, message_history)
        messages.append({"role": "user", "content": frame_parts})
        return messages

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        """Per-session conversation state (default: empty OpenAI-style list)."""
        return []

    def _incremental_prefill_active(self, config: StreamingVideoSessionConfig) -> bool:
        """Incremental frame prefill: stage-0 prefix caching on, text-only output."""
        return self._incremental_prefill_supported and "audio" not in config.modalities

    def on_frame_buffered(
        self,
        raw_bytes: bytes,
        frame_b64: str,
        message_history: Any,
        config: StreamingVideoSessionConfig,
    ) -> None:
        """Hook after a frame is accepted into the session buffer."""
        del raw_bytes, frame_b64, message_history, config

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        engine_client: Any | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout
        self._engine_client = engine_client
        self._incremental_prefill_supported = _stage0_prefix_caching_enabled(engine_client)

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            frame_buffer: list[str] = []  # base64-encoded JPEG frames
            frame_metadata: list[dict[str, Any]] = []
            # Per-frame PIL cache + uuid for mm_hash reuse. Aligned with frame_buffer by index.
            frame_pil_cache: dict[str, tuple[Any, str] | object] = {}  # b64 -> (PIL.Image, uuid) or _BAD_FRAME
            frame_filter = (
                FrameSimilarityFilter(threshold=config.frame_filter_threshold) if config.enable_frame_filter else None
            )
            audio_buffer = bytearray()  # raw PCM16 16kHz mono
            message_history: Any = self.create_message_history(config)
            active_request_id: str | None = None
            prev_request_id: str | None = None  # abort target iff prev was interrupted
            prev_was_interrupted: bool = False
            interrupt_event = asyncio.Event()
            prewarm_tasks: set[asyncio.Task[Any]] = set()
            query_task: asyncio.Task[Any] | None = None
            session_alive = True
            # Session-owned decode readiness. Warmup/query wait on these
            # Events, not on the prewarm tasks, so cancelling an engine
            # request cannot cascade into JPEG decode.
            frame_ready: dict[str, asyncio.Event] = {}
            # In-flight warmup/query snapshots keep their PIL entries if FIFO
            # evicts the live buffer while they await readiness.
            pinned_frame_refs: dict[str, int] = {}

            def _drop_frame_cache(frame_b64: str) -> None:
                frame_pil_cache.pop(frame_b64, None)
                frame_ready.pop(frame_b64, None)

            def _pin_frames(frames: list[str]) -> None:
                for frame_b64 in frames:
                    pinned_frame_refs[frame_b64] = pinned_frame_refs.get(frame_b64, 0) + 1

            def _unpin_frames(frames: list[str]) -> None:
                for frame_b64 in frames:
                    remaining = pinned_frame_refs.get(frame_b64, 0) - 1
                    if remaining <= 0:
                        pinned_frame_refs.pop(frame_b64, None)
                        # Settle the eviction this pin deferred: the frame left
                        # the buffer, so its cache entry is orphaned.
                        if frame_b64 not in frame_buffer:
                            _drop_frame_cache(frame_b64)
                    else:
                        pinned_frame_refs[frame_b64] = remaining

            def _snapshot_prewarmed(frames: list[str]) -> dict[str, Any]:
                return {frame_b64: frame_pil_cache[frame_b64] for frame_b64 in frames if frame_b64 in frame_pil_cache}

            def _frames_ready(frames: list[str]) -> bool:
                for frame_b64 in frames:
                    event = frame_ready.get(frame_b64)
                    if event is None or not event.is_set():
                        return False
                return True

            async def _await_frames_ready(frames: list[str]) -> None:
                waits: list[Any] = []
                for frame_b64 in dict.fromkeys(frames):
                    event = frame_ready.get(frame_b64)
                    if event is not None and not event.is_set():
                        waits.append(event.wait())
                if waits:
                    await asyncio.gather(*waits)

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_MAX_MSG_QUEUE)

            async def _reader() -> None:
                """Receive WebSocket messages and enqueue them."""
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=self._idle_timeout,
                            )
                        except asyncio.TimeoutError:
                            await self._send_error(websocket, "Idle timeout")
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

                        msg_type = str(msg.get("type", ""))
                        if msg_type.startswith("_internal."):
                            await self._send_error(websocket, f"Unknown type: {msg_type}")
                            continue
                        if msg_type == "video.frame":
                            msg["_receiver_received_ts_ms"] = _time.monotonic() * 1000

                        await msg_queue.put(msg)
                        if msg.get("type") == "video.done":
                            return
                except WebSocketDisconnect:
                    await msg_queue.put(None)
                except Exception:
                    await msg_queue.put(None)
                    raise

            async def _cancel_active_query(*, abort_now: bool = False) -> None:
                """Signal soft interrupt for the active query."""
                nonlocal active_request_id, prev_was_interrupted, query_task
                if active_request_id is not None:
                    interrupt_event.set()
                    prev_was_interrupted = True
                    logger.info("Interrupt signaled for %s", active_request_id)
                    if abort_now and self._engine_client:
                        try:
                            await self._engine_client.abort(active_request_id)
                        except Exception:
                            logger.debug("Abort failed for %s", active_request_id, exc_info=True)
                    if query_task is not None and not query_task.done():
                        query_task.cancel()
                        await asyncio.gather(query_task, return_exceptions=True)
                    query_task = None

            # --- Incremental prefill: keep the prefix cache warmed to the latest
            # buffered frame so a query only pays for its own text suffix. ---
            warmup_task: asyncio.Task[Any] | None = None
            warmup_request_id: str | None = None
            warmed_signature: tuple[Any, ...] | None = None

            def _context_signature() -> tuple[Any, ...]:
                history_len = len(message_history) if isinstance(message_history, list) else None
                return (history_len, len(frame_buffer), frame_buffer[-1] if frame_buffer else None)

            def _maybe_start_warmup() -> None:
                nonlocal warmup_task, warmup_request_id
                if not session_alive:
                    return
                if self._engine_client is None or not self._incremental_prefill_active(config):
                    return
                if active_request_id is not None:
                    return
                if warmup_task is not None and not warmup_task.done():
                    return
                if not frame_buffer or _context_signature() == warmed_signature:
                    return
                frames = list(frame_buffer)
                # Decode is session-owned: do not start prefill until PIL is
                # already in cache, so this task is only engine work.
                if not _frames_ready(frames):
                    return
                signature = _context_signature()
                request_id = f"video-warmup-{uuid.uuid4().hex[:12]}"
                warmup_request_id = request_id
                _pin_frames(frames)

                async def _run_warmup() -> None:
                    nonlocal warmup_request_id, warmed_signature
                    cancelled = False
                    try:
                        messages = self.build_engine_prompt_prefix(
                            config, frames, message_history, _snapshot_prewarmed(frames)
                        )
                        if not messages:
                            warmed_signature = signature
                            return
                        await self._prefill_context(config, messages, request_id)
                        warmed_signature = signature
                    except asyncio.CancelledError:
                        cancelled = True
                        raise
                    except Exception:
                        # Mark attempted so a persistently failing state does
                        # not hot-loop; the next frame retriggers naturally.
                        warmed_signature = signature
                        logger.debug("Context warmup failed for %s", request_id, exc_info=True)
                    finally:
                        _unpin_frames(frames)
                        if warmup_request_id == request_id:
                            warmup_request_id = None
                        # Skip on cancel: query already owns the session and will
                        # call_soon after it finishes. Restarting here can spawn a
                        # second warmup with the same frames before active_request_id
                        # is set.
                        if not cancelled:
                            asyncio.get_running_loop().call_soon(_maybe_start_warmup)

                warmup_task = asyncio.create_task(_run_warmup())

            async def _cancel_warmup() -> None:
                nonlocal warmup_task, warmup_request_id
                request_id = warmup_request_id
                if warmup_task is not None and not warmup_task.done():
                    warmup_task.cancel()
                    await asyncio.gather(warmup_task, return_exceptions=True)
                warmup_task = None
                warmup_request_id = None
                if request_id and self._engine_client:
                    try:
                        await self._engine_client.abort(request_id)
                    except Exception:
                        logger.debug("Warmup abort failed for %s", request_id, exc_info=True)

            async def _start_query_turn(*, query_text: str) -> None:
                """Schedule a new inference turn from the current buffers."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task

                await _cancel_active_query()

                if not frame_buffer:
                    await _cancel_warmup()
                    await self._send_error(websocket, "No frames buffered")
                    return

                # Claim the session before cancelling warmup so a leftover
                # call_soon from a just-finished warmup hits the 405 gate.
                request_id = f"video-{uuid.uuid4().hex[:12]}"
                active_request_id = request_id
                interrupt_event.clear()
                # Free prefill capacity for the query; completed warmup blocks
                # stay in the prefix cache and are reused by it.
                await _cancel_warmup()

                if prev_was_interrupted and prev_request_id and self._engine_client:
                    try:
                        await self._engine_client.abort(prev_request_id)
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)
                prev_was_interrupted = False
                query_frames = list(frame_buffer)
                query_frame_metadata = list(frame_metadata)
                query_audio_buffer = bytearray(audio_buffer)
                audio_buffer.clear()
                _pin_frames(query_frames)

                async def _run_query() -> None:
                    nonlocal active_request_id, prev_request_id
                    try:
                        await _await_frames_ready(query_frames)
                        query_prewarmed_frames = _snapshot_prewarmed(query_frames)
                        process_kwargs: dict[str, Any] = {}
                        if any(metadata.get("frame_id") for metadata in query_frame_metadata):
                            process_kwargs["frame_metadata"] = query_frame_metadata
                        await self._process_query(
                            websocket,
                            config,
                            query_frames,
                            query_audio_buffer,
                            message_history,
                            query_text,
                            request_id,
                            interrupt_event,
                            query_prewarmed_frames,
                            **process_kwargs,
                        )
                    finally:
                        _unpin_frames(query_frames)
                        if active_request_id == request_id:
                            prev_request_id = request_id
                            active_request_id = None
                        # Warm frames that arrived during the turn (fires after
                        # this task completes; gated on no active query).
                        asyncio.get_running_loop().call_soon(_maybe_start_warmup)

                query_task = asyncio.create_task(_run_query())

            async def _processor() -> None:
                """Process enqueued messages."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task
                nonlocal session_alive

                while True:
                    msg = await msg_queue.get()
                    if msg is None:
                        session_alive = False
                        await _cancel_active_query(abort_now=True)
                        return

                    msg_type = msg.get("type")

                    if msg_type == "_internal.frame_decode_failed":
                        frame_data = msg.get("b64", "")
                        removed = frame_data in frame_buffer
                        if removed:
                            retained_indices = [
                                index for index, frame in enumerate(frame_buffer) if frame != frame_data
                            ]
                            frame_buffer[:] = [frame_buffer[index] for index in retained_indices]
                            frame_metadata[:] = [frame_metadata[index] for index in retained_indices]
                        if frame_pil_cache.get(frame_data) is _BAD_FRAME:
                            _drop_frame_cache(frame_data)
                        if removed:
                            await self._send_error(websocket, "Frame decode failed")

                    elif msg_type == "video.frame":
                        frame_data = msg.get("data", "")
                        if not frame_data:
                            continue
                        if len(frame_data) > _MAX_FRAME_SIZE:
                            await self._send_error(websocket, "Frame too large")
                            continue
                        try:
                            raw_bytes = base64.b64decode(frame_data, validate=True)
                        except Exception:
                            await self._send_error(websocket, "Invalid image data")
                            continue
                        if frame_filter is not None:
                            try:
                                if not frame_filter.should_retain(raw_bytes):
                                    await self._send_frame_ack(
                                        websocket,
                                        msg,
                                        accepted=False,
                                        buffered_frames=len(frame_buffer),
                                        reason="filtered",
                                    )
                                    continue
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                        max_buf = config.max_frames
                        dropped_frame_id: str | None = None
                        if len(frame_buffer) >= max_buf:
                            dropped = frame_buffer.pop(0)
                            dropped_metadata = frame_metadata.pop(0)
                            dropped_frame_id = dropped_metadata.get("frame_id")
                            if dropped not in pinned_frame_refs:
                                _drop_frame_cache(dropped)
                        frame_buffer.append(frame_data)
                        frame_metadata.append(
                            {
                                "frame_id": msg.get("frame_id"),
                                "pts_ms": msg.get("pts_ms"),
                                "source_pts_ms": msg.get("source_pts_ms"),
                                "quality_profile": msg.get("quality_profile"),
                                "capture_ts_ms": msg.get("capture_ts_ms"),
                                "receiver_received_ts_ms": msg.get("_receiver_received_ts_ms"),
                            }
                        )
                        self.on_frame_buffered(raw_bytes, frame_data, message_history, config)
                        await self._send_frame_ack(
                            websocket,
                            msg,
                            accepted=True,
                            buffered_frames=len(frame_buffer),
                            dropped_frame_id=dropped_frame_id,
                        )
                        # Prewarm: decode PIL off the event loop so query-time chat_template
                        # can skip base64+Image.open. uuid=md5 lets mm_cache dedupe identical frames.
                        if frame_data not in frame_pil_cache and frame_data not in frame_ready:
                            mm_uuid = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()
                            frame_ready[frame_data] = asyncio.Event()

                            async def _prewarm(b64: str, b: bytes, u: str) -> None:
                                cancelled = False
                                try:
                                    pil = await asyncio.to_thread(_decode_frame_bytes, b)
                                    if b64 in frame_ready:
                                        frame_pil_cache[b64] = (pil, u)
                                except asyncio.CancelledError:
                                    cancelled = True
                                    raise
                                except Exception:
                                    if b64 not in frame_ready:
                                        return
                                    frame_pil_cache[b64] = _BAD_FRAME
                                    logger.warning("prewarm decode failed for frame (len=%d)", len(b))
                                    try:
                                        msg_queue.put_nowait({"type": "_internal.frame_decode_failed", "b64": b64})
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            "frame decode failure event dropped because message queue is full"
                                        )
                                finally:
                                    event = frame_ready.get(b64)
                                    if event is not None:
                                        event.set()
                                    if not cancelled:
                                        asyncio.get_running_loop().call_soon(_maybe_start_warmup)

                            task = asyncio.create_task(_prewarm(frame_data, raw_bytes, mm_uuid))
                            prewarm_tasks.add(task)
                            task.add_done_callback(prewarm_tasks.discard)
                        elif frame_data in frame_pil_cache:
                            frame_ready.setdefault(frame_data, asyncio.Event()).set()

                        is_generating = active_request_id is not None or (
                            query_task is not None and not query_task.done()
                        )
                        if self.should_trigger_turn(
                            VideoStreamTurnTrigger(
                                frame_count=len(frame_buffer),
                                is_generating=is_generating,
                                config=config,
                            )
                        ):
                            await _start_query_turn(query_text="")
                        else:
                            _maybe_start_warmup()

                    elif msg_type == "audio.chunk":
                        data_b64 = msg.get("data", "")
                        try:
                            pcm_bytes = base64.b64decode(data_b64)
                        except Exception:
                            continue
                        if len(audio_buffer) + len(pcm_bytes) > _MAX_AUDIO_BUFFER_BYTES:
                            await self._send_error(websocket, "Audio buffer overflow")
                            audio_buffer.clear()
                            continue
                        audio_buffer.extend(pcm_bytes)

                    elif msg_type == "video.query":
                        query_text = msg.get("text", "")
                        audio_data_b64 = msg.get("audio_data")
                        if audio_data_b64:
                            try:
                                decoded = base64.b64decode(audio_data_b64)
                                if len(audio_buffer) + len(decoded) <= _MAX_AUDIO_BUFFER_BYTES:
                                    audio_buffer.extend(decoded)
                                else:
                                    await self._send_error(websocket, "Audio buffer overflow")
                                    audio_buffer.clear()
                            except Exception:
                                pass

                        await _start_query_turn(query_text=query_text)

                    elif msg_type == "video.done":
                        session_alive = False
                        if query_task is not None and not query_task.done():
                            await asyncio.gather(query_task, return_exceptions=True)
                            query_task = None
                        await websocket.send_json({"type": "session.done"})
                        return

                    elif msg_type == "ping":
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass

                    else:
                        await self._send_error(websocket, f"Unknown type: {msg_type}")

            reader_task = asyncio.create_task(_reader())
            try:
                await _processor()
            finally:
                session_alive = False
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
                for t in list(prewarm_tasks):
                    t.cancel()
                if prewarm_tasks:
                    await asyncio.gather(*prewarm_tasks, return_exceptions=True)
                await _cancel_warmup()
                if query_task is not None and not query_task.done():
                    await _cancel_active_query(abort_now=True)

        except WebSocketDisconnect:
            logger.info("Streaming video: client disconnected")
        except Exception as e:
            logger.exception("Streaming video session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(self, websocket: WebSocket) -> StreamingVideoSessionConfig | None:
        """Wait for and validate the session.config message."""
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

        config_data = {k: v for k, v in msg.items() if k != "type"}
        alias_map = {
            "num_sample_frames": "num_frames",
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data[old_key]

        try:
            config = StreamingVideoSessionConfig(**config_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _process_query(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: dict[str, Any],
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        engine_kwargs: dict[str, Any] = {}
        if frame_metadata:
            engine_kwargs["frame_metadata"] = frame_metadata
        await self._process_query_engine(
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
            **engine_kwargs,
        )

    # ------------------------------------------------------------------
    # Engine-client path (async_chunk audio streaming)
    # ------------------------------------------------------------------

    async def _process_query_engine(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: dict[str, Any],
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Direct engine_client.generate() path for async_chunk audio."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        engine_client = self._engine_client
        if engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        reuse_active = self._incremental_prefill_active(config)
        messages, user_message = self.build_engine_prompt(
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        # Incremental: same kwargs as warmup so frame mm hashes stay a prefix.
        # Legacy: only when this query actually has input audio.
        if config.use_audio_in_video and (reuse_active or len(audio_buffer) > 0):
            request_kwargs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = config.sampling_params_list

        try:
            chat_request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(websocket, f"Failed to build request: {e}")
            return

        try:
            engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
        except Exception as e:
            await self._send_error(websocket, f"Preprocess failed: {e}")
            return
        decoded_ready_ts_ms = _time.monotonic() * 1000
        if reuse_active:
            selected_metadata = list(frame_metadata or [])
        else:
            selected_metadata = self._sample_frame_metadata(frame_metadata or [], config.num_frames)
        model_selected_ts_ms = _time.monotonic() * 1000

        await websocket.send_json({"type": "response.start"})
        text_parts: list[str] = []
        text_done_sent = False
        audio_chunk_count = 0
        # Number of per-step tensors in OmniRequestOutput.audio_data already
        # drained. Used by the fast path to skip already-emitted history.
        audio_chunks_drained = 0
        previous_text = ""
        interrupted = False
        frames_consumed_sent = False
        t_start = _time.monotonic()
        t_first_text = None
        t_first_audio = None

        # Wire-level async-chunk switch. "off" means
        # buffer all deltas server-side and flush once at the end; the engine
        # pipeline still overlaps internally.
        async_chunk_mode = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK
        streaming = async_chunk_mode == "on"
        audio_tail_tensors: list[Any] = []

        try:
            result_gen = engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                output_modalities=config.modalities,
            )

            async for output in result_gen:
                # Soft interrupt: drain without sending
                if interrupt_event.is_set():
                    if not interrupted:
                        logger.info("Generation interrupted — draining")
                        interrupted = True
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

                if not frames_consumed_sent and frame_metadata:
                    await websocket.send_json(
                        {
                            "type": "video.frames.consumed",
                            "request_id": request_id,
                            "model_selected_ts_ms": model_selected_ts_ms,
                            "frame_ids": [
                                metadata["frame_id"]
                                for metadata in selected_metadata
                                if isinstance(metadata.get("frame_id"), str)
                            ],
                            "frames": [
                                {
                                    "frame_id": metadata.get("frame_id"),
                                    "pts_ms": metadata.get("pts_ms"),
                                    "source_pts_ms": metadata.get("source_pts_ms"),
                                    "quality_profile": metadata.get("quality_profile"),
                                    "receiver_received_ts_ms": metadata.get("receiver_received_ts_ms"),
                                    "decoded_ready_ts_ms": decoded_ready_ts_ms,
                                }
                                for metadata in selected_metadata
                            ],
                            "latest_pts_ms": selected_metadata[-1].get("pts_ms") if selected_metadata else None,
                        }
                    )
                    frames_consumed_sent = True

                out_type = getattr(output, "final_output_type", "text")

                if out_type == "audio":
                    if streaming and not text_done_sent:
                        full_text = "".join(text_parts)
                        await websocket.send_json({"type": "response.text.done", "text": full_text})
                        text_done_sent = True

                    if t_first_audio is None:
                        t_first_audio = _time.monotonic()
                    audio_chunk_count += 1
                    if streaming:
                        b64, audio_chunks_drained = self._extract_audio_delta_b64(
                            output,
                            audio_chunks_drained,
                        )
                        if b64:
                            await websocket.send_json(
                                {
                                    "type": "response.audio.delta",
                                    "data": b64,
                                    "format": "wav",
                                }
                            )
                    else:
                        audio_data = self._get_audio_data(output)
                        if audio_data is not None:
                            if isinstance(audio_data, list):
                                audio_tail_tensors = list(audio_data)
                            else:
                                audio_tail_tensors = [audio_data]
                else:
                    delta_text, previous_text = self._extract_text_delta(
                        output,
                        previous_text,
                    )
                    if delta_text:
                        if t_first_text is None:
                            t_first_text = _time.monotonic()
                        text_parts.append(delta_text)
                        if streaming:
                            await websocket.send_json({"type": "response.text.delta", "delta": delta_text})

            if not text_done_sent:
                full_text = "".join(text_parts)
                await websocket.send_json({"type": "response.text.done", "text": full_text})
                text_done_sent = True

            if not streaming and audio_tail_tensors:
                try:
                    coalesced = (
                        audio_tail_tensors[0] if len(audio_tail_tensors) == 1 else torch.cat(audio_tail_tensors, dim=-1)
                    )
                    tail_np = self._tensor_to_1d_np(coalesced)
                    b64, _ = self._encode_tail(
                        tail_np,
                        0,
                        new_drained=len(audio_tail_tensors),
                        is_first=True,
                    )
                    if b64:
                        await websocket.send_json(
                            {
                                "type": "response.audio.delta",
                                "data": b64,
                                "format": "wav",
                            }
                        )
                except Exception:
                    logger.exception("Failed to coalesce off-path audio")

            if audio_chunk_count > 0:
                await websocket.send_json({"type": "response.audio.done"})

            response_text = "".join(text_parts)
            self.on_turn_complete(message_history, user_message, response_text)

            t_end = _time.monotonic()
            logger.info(
                "[TIMING] mode=%s total=%.2fs first_text=%.2fs first_audio=%.2fs audio_chunks=%d",
                async_chunk_mode,
                t_end - t_start,
                (t_first_text - t_start) if t_first_text else -1,
                (t_first_audio - t_start) if t_first_audio else -1,
                audio_chunk_count,
            )

        except Exception:
            logger.exception("Engine query failed")
            await self._send_error(websocket, "Query processing failed")

        if not text_done_sent:
            full_text = "".join(text_parts)
            await websocket.send_json({"type": "response.text.done", "text": full_text})

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_to_wav_b64(pcm_data: bytes, sample_rate: int = 16000) -> str:
        """Wrap raw PCM16 mono in a WAV container and return base64."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(buf.getvalue()).decode()

    @classmethod
    def _extract_audio_delta_b64(
        cls,
        result: OmniRequestOutput,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Return (base64 WAV of new samples, updated chunks_drained).

        `chunks_drained` is the number of per-step tensors in
        ``audio_data`` that have already been emitted. Each engine step appends
        one tensor, so new samples are ``audio_data[chunks_drained:]`` — no
        matter how many steps accumulated between reads (handles backpressure
        cleanly, unlike a simple ``audio_data[-1]``).

        Two paths, selected at runtime by ``VLLM_VIDEO_AUDIO_DELTA_MODE``:
          * fast — only D2H the new tail. Per-call cost ∝ new chunks.
          * slow — full cat + D2H each call. Per-call cost ∝ total history.
                   Retained for A/B; remove once downstream callers confirm.
        """
        audio_data = cls._get_audio_data(result)
        if audio_data is None:
            return None, chunks_drained

        if video_stream_envs.VLLM_VIDEO_AUDIO_DELTA_MODE == "slow":
            return cls._delta_slow(audio_data, chunks_drained)
        return cls._delta_fast(audio_data, chunks_drained)

    @staticmethod
    def _get_audio_data(result: OmniRequestOutput):
        """Navigate OmniRequestOutput → multimodal_output['audio']. None on miss."""
        request_output = result
        if request_output is None:
            return None
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return None
        mm_output = getattr(outputs[0], "multimodal_output", None)
        if not isinstance(mm_output, Mapping):
            return None
        return mm_output.get("audio")

    @classmethod
    def _delta_fast(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Emit only tensors appended since the last call."""
        # Single tensor: output_processor hands us one tensor before it becomes a
        # list (see output_processor.py:89). Treat it as chunk #0.
        if not isinstance(audio_data, list):
            if chunks_drained >= 1:
                return None, chunks_drained
            tail_np = cls._tensor_to_1d_np(audio_data)
            return cls._encode_tail(tail_np, chunks_drained, new_drained=1, is_first=True)

        n = len(audio_data)
        if n <= chunks_drained:
            return None, chunks_drained

        new_chunks = audio_data[chunks_drained:]
        tail = new_chunks[0] if len(new_chunks) == 1 else torch.cat(new_chunks, dim=-1)
        tail_np = cls._tensor_to_1d_np(tail)
        return cls._encode_tail(tail_np, chunks_drained, new_drained=n, is_first=(chunks_drained == 0))

    @classmethod
    def _delta_slow(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Pre-fix behaviour: concat everything each call and slice on CPU."""
        if isinstance(audio_data, list):
            if not audio_data:
                return None, chunks_drained
            audio_tensor = torch.cat(audio_data, dim=-1)
            new_drained = len(audio_data)
        else:
            audio_tensor = audio_data
            new_drained = 1

        full_np = cls._tensor_to_1d_np(audio_tensor)
        if full_np is None:
            return None, chunks_drained
        # chunks_drained doesn't map directly to sample offset without tracking
        # per-chunk lengths, so we re-derive: replay the tail that corresponds
        # to chunks appended since last call by slicing off the part produced
        # by the already-drained prefix. For slow path this is intentionally
        # wasteful — the point is to reproduce the pre-fix hot loop.
        if chunks_drained == 0:
            tail_np = full_np
        else:
            # Recover prefix length by re-concatenating the already-drained
            # prefix tensors (cost intentionally identical to the baseline
            # implementation this was lifted from).
            if isinstance(audio_data, list) and chunks_drained < len(audio_data):
                prefix_len = sum(int(t.shape[-1]) for t in audio_data[:chunks_drained])
                tail_np = full_np[prefix_len:]
            else:
                tail_np = full_np[0:0]
        return cls._encode_tail(tail_np, chunks_drained, new_drained=new_drained, is_first=(chunks_drained == 0))

    @classmethod
    def _encode_tail(
        cls,
        tail_np,
        old_drained: int,
        *,
        new_drained: int,
        is_first: bool,
    ) -> tuple[str | None, int]:
        """Strip the CausalConv leading artifact on first emit, then b64-encode."""
        if tail_np is None or len(tail_np) == 0:
            return None, new_drained
        if is_first and len(tail_np) > _CODEC_FRAME_SAMPLES * 2:
            tail_np = tail_np[_CODEC_FRAME_SAMPLES:]
        if len(tail_np) == 0:
            return None, new_drained
        try:
            return cls._encode_audio_wav_b64(tail_np), new_drained
        except Exception:
            logger.exception("Failed to encode audio delta WAV")
            return None, old_drained

    @staticmethod
    def _tensor_to_1d_np(t):
        """Tensor → flat float32 numpy on CPU. None on failure."""
        if t is None or not hasattr(t, "float"):
            return None
        arr = t.float().detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    @staticmethod
    def _encode_audio_wav_b64(audio_np) -> str:
        """Encode numpy float32 audio to base64 WAV (24kHz)."""
        from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

        audio_obj = CreateAudio(
            audio_tensor=audio_np,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            base64_encode=True,
        )
        mixin = AudioMixin()
        resp = mixin.create_audio(audio_obj)
        audio_b64 = resp.audio_data
        return audio_b64.decode() if isinstance(audio_b64, bytes) else audio_b64

    @staticmethod
    def _extract_text_delta(
        result: OmniRequestOutput,
        previous_text: str,
    ) -> tuple[str, str]:
        """Extract incremental text delta from OmniRequestOutput."""
        if result.final_output_type != "text":
            return "", previous_text

        request_output = result
        if request_output is None:
            return "", previous_text

        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return "", previous_text

        text = getattr(outputs[0], "text", None)
        if not isinstance(text, str) or not text:
            return "", previous_text

        if text.startswith(previous_text):
            return text[len(previous_text) :], text
        return text, text

    # ------------------------------------------------------------------
    # Incremental prefill (context warmup)
    # ------------------------------------------------------------------

    async def _prefill_context(
        self,
        config: StreamingVideoSessionConfig,
        messages: list[dict[str, Any]],
        request_id: str,
    ) -> None:
        """max_tokens=1 generate so arriving frames land in the prefix cache."""
        from vllm import SamplingParams
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        engine_client = self._engine_client
        if engine_client is None:
            return

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if config.use_audio_in_video:
            request_kwargs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }
        chat_request = ChatCompletionRequest(**request_kwargs)
        engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
        result_gen = engine_client.generate(
            prompt=engine_prompt,
            request_id=request_id,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            output_modalities=config.modalities,
        )
        async for _ in result_gen:
            pass

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    async def _preprocess_to_engine_prompt(self, request) -> Any:
        """Use the chat handler's preprocessing to build an engine prompt."""
        handler = self._chat_service
        renderer = handler.renderer

        _conversation, engine_prompts = await handler._preprocess_chat(
            request,
            request.messages,
            default_template=getattr(request, "chat_template", None) or handler.chat_template,
            default_template_content_format=handler.chat_template_content_format,
            renderer=renderer,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            add_special_tokens=request.add_special_tokens,
        )
        return engine_prompts[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    _text_only_message = staticmethod(text_only_message)

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass

    @staticmethod
    def _sample_frame_metadata(
        frame_metadata: list[dict[str, Any]],
        num_frames: int,
    ) -> list[dict[str, Any]]:
        if len(frame_metadata) <= num_frames:
            return list(frame_metadata)
        stride = max(1, len(frame_metadata) // num_frames)
        indices = [index * stride for index in range(num_frames - 1)] + [len(frame_metadata) - 1]
        return [frame_metadata[index] for index in indices]

    @staticmethod
    async def _send_frame_ack(
        websocket: WebSocket,
        message: Mapping[str, Any],
        *,
        accepted: bool,
        buffered_frames: int,
        reason: str | None = None,
        dropped_frame_id: str | None = None,
    ) -> None:
        frame_id = message.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            return
        ack: dict[str, Any] = {
            "type": "video.frame.ack",
            "frame_id": frame_id,
            "pts_ms": message.get("pts_ms"),
            "capture_ts_ms": message.get("capture_ts_ms"),
            "accepted": accepted,
            "buffered_frames": buffered_frames,
            "server_receive_ts_ms": message.get("_receiver_received_ts_ms", _time.monotonic() * 1000),
        }
        if reason is not None:
            ack["reason"] = reason
        if dropped_frame_id is not None:
            ack["dropped_frame_id"] = dropped_frame_id
        await websocket.send_json(ack)
