# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WebSocket handler for realtime video streaming.

Supports both text-to-video (T2V) and video-to-video (V2V) streaming.
Uses MessagePack binary protocol for efficient frame transfer.
GPU work is delegated to DiffusionWorker via collective_rpc.

Protocol:
    Client -> Server:
        {"type": "config", "mode": "t2v"|"v2v", "prompt": "...", ...}
        {"type": "prompt", "content": "new prompt"}
        {"type": "video", "frames": [bytes, ...]}

    Server -> Client:
        {"type": "frame", "content": bytes}
        {"type": "status", "content": "..."}
        {"type": "error", "content": "..."}
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections import deque
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

try:
    from msgpack import packb, unpackb
except ImportError:
    packb = None
    unpackb = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

_CONFIG_TIMEOUT = 30.0
_FRAME_WAIT_INTERVAL = 0.01
_VAE_TEMPORAL_COMPRESSION = 4


class GenerateSession:
    """Session state for a single WebSocket realtime video connection.

    Only holds API-layer state. GPU-side state (KV caches, latents,
    decoder cache) is managed by the worker via session_id.
    """

    def __init__(self) -> None:
        self.id: str = uuid4().hex
        self.mode: str | None = None
        self.prompt: str | None = None
        self.height: int = 480
        self.width: int = 832
        self.num_inference_steps: int | None = None
        self.num_frames_per_block: int = 3
        self.action_queue: deque = deque(maxlen=1)
        self.video_frame_queue: deque = deque()
        self.generate_chunk_cnt: int = 0

    @property
    def required_video_frames(self) -> int:
        n = self.num_frames_per_block
        r = _VAE_TEMPORAL_COMPRESSION
        if self.generate_chunk_cnt == 0:
            return (n - 1) * r + 1
        return n * r

    def has_pending_video_frames(self) -> bool:
        return len(self.video_frame_queue) >= self.required_video_frames

    def sample_video_frames_bytes(self) -> list[bytes] | None:
        """Pop exactly the required number of frames from the queue.

        Only consumes `required_video_frames` frames, leaving the rest
        for subsequent blocks. This avoids discarding buffered frames
        when the client sends ahead.
        """
        required = self.required_video_frames
        if len(self.video_frame_queue) < required:
            return None

        frames = [self.video_frame_queue.popleft() for _ in range(required)]

        result = []
        for frame_bytes in frames:
            if isinstance(frame_bytes, bytes):
                result.append(frame_bytes)
            elif Image is not None:
                buf = io.BytesIO()
                frame_bytes.save(buf, format="JPEG", quality=90)
                result.append(buf.getvalue())
        return result

    def get_current_prompt(self) -> str:
        if self.action_queue:
            action = self.action_queue.popleft()
            self.prompt = action
        return self.prompt

    def generate_chunk_completed(self) -> None:
        self.generate_chunk_cnt += 1

    def dispose(self) -> None:
        self.action_queue.clear()
        self.video_frame_queue.clear()


class RealtimeVideoHandler:
    """Handles WebSocket sessions for realtime video generation.

    Delegates GPU work to DiffusionWorker via engine_client.collective_rpc().
    """

    def __init__(self, engine_client: Any) -> None:
        self._engine = engine_client

    async def _rpc(self, method: str, *args: Any) -> Any:
        """Call a worker method via collective_rpc, return rank-0 result.

        The result goes through multiple wrapping layers:
          worker result → executor [result] → stage_client [result] →
          orchestrator [[result]] → AsyncOmni [[result]]
        We unwrap until we reach a non-list scalar.
        """
        results = await self._engine.collective_rpc(method=method, args=args)
        # Unwrap nested list layers: [[value]] → [value] → value
        while isinstance(results, list) and len(results) == 1:
            results = results[0]
        return results

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session handler for a WebSocket connection."""
        await websocket.accept()
        session = GenerateSession()
        worker_session_created = False

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            session.mode = config.get("mode", "t2v")
            session.prompt = config.get("prompt", "")
            session.height = config.get("height", 480)
            session.width = config.get("width", 832)
            session.num_inference_steps = config.get("num_inference_steps")
            session.num_frames_per_block = config.get("num_frames_per_block", 3)

            if session.mode == "t2v" and config.get("first_frame"):
                await self._send_error(websocket, "first_frame not allowed in T2V mode")
                return

            await self._rpc(
                "realtime_create_session",
                session.id,
                {
                    "num_frames_per_block": config.get("num_frames_per_block", 3),
                    "kv_cache_num_frames": config.get("kv_cache_num_frames", 3),
                    "v2v_strength": config.get("v2v_strength"),
                    "frame_format": config.get("frame_format", "jpeg"),
                    "frame_quality": config.get("frame_quality", 95),
                },
            )
            worker_session_created = True

            await self._send_status(websocket, "session_started")

            gen_task = asyncio.create_task(self._generate_loop(websocket, session))
            listen_task = asyncio.create_task(self._listen_actions(websocket, session))

            done, pending = await asyncio.wait({gen_task, listen_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    logger.error(
                        "Session %s task error: %s",
                        session.id,
                        task.exception(),
                    )

        except WebSocketDisconnect:
            logger.info("Session %s disconnected", session.id)
        except Exception as e:
            logger.error("Session %s error: %s", session.id, e)
            try:
                await self._send_error(websocket, str(e))
            except Exception:
                pass
        finally:
            if worker_session_created:
                try:
                    await self._rpc("realtime_dispose_session", session.id)
                except Exception:
                    logger.warning(
                        "Session %s: failed to dispose worker session",
                        session.id,
                    )
            session.dispose()

    async def _receive_config(self, websocket: WebSocket) -> dict | None:
        """Wait for initial config message."""
        try:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=_CONFIG_TIMEOUT)
            if unpackb is not None:
                msg = unpackb(data, raw=False)
            else:
                import json

                msg = json.loads(data)

            if not isinstance(msg, dict) or msg.get("type") != "config":
                await self._send_error(
                    websocket,
                    'Expected {"type": "config", ...} as first message',
                )
                return None
            return msg
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Config timeout")
            return None
        except Exception as e:
            await self._send_error(websocket, f"Config parse error: {e}")
            return None

    async def _generate_loop(
        self,
        websocket: WebSocket,
        session: GenerateSession,
    ) -> None:
        """Main generation loop: produces video blocks continuously."""
        while True:
            is_v2v = session.mode == "v2v"

            if is_v2v:
                while not session.has_pending_video_frames():
                    await asyncio.sleep(_FRAME_WAIT_INTERVAL)

            prompt = session.get_current_prompt()

            video_frames_bytes = None
            if is_v2v:
                video_frames_bytes = session.sample_video_frames_bytes()
                if video_frames_bytes is None:
                    continue

            try:
                result = await self._rpc(
                    "realtime_generate_block",
                    session.id,
                    prompt,
                    session.height,
                    session.width,
                    session.num_inference_steps,
                    video_frames_bytes,
                )

                if isinstance(result, list):
                    for frame_bytes in result:
                        await self._send_frame(websocket, frame_bytes)
                else:
                    await self._send_frame(websocket, result)
                session.generate_chunk_completed()

            except Exception as e:
                logger.error(
                    "Session %s generation error at block %d: %s",
                    session.id,
                    session.generate_chunk_cnt,
                    e,
                )
                await self._send_error(websocket, f"Generation error: {e}")
                break

    async def _listen_actions(
        self,
        websocket: WebSocket,
        session: GenerateSession,
    ) -> None:
        """Listen for prompt/video updates from client."""
        async for data in websocket.iter_bytes():
            try:
                if unpackb is not None:
                    msg = unpackb(data, raw=False)
                else:
                    import json

                    msg = json.loads(data)

                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "prompt":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        session.action_queue.append(content.strip())

                elif msg_type == "video":
                    if session.mode != "v2v":
                        await self._send_error(
                            websocket,
                            "Video frames only accepted in V2V mode",
                        )
                        continue
                    frames_data = msg.get("frames", [])
                    if frames_data:
                        session.video_frame_queue.extend(frames_data)

            except Exception as e:
                logger.warning("Session %s action parse error: %s", session.id, e)

    async def _send_frame(self, websocket: WebSocket, content: bytes) -> None:
        msg = {"type": "frame", "content": content}
        if packb is not None:
            await websocket.send_bytes(packb(msg, use_bin_type=True))
        else:
            await websocket.send_bytes(content)

    async def _send_status(self, websocket: WebSocket, status: str) -> None:
        msg = {"type": "status", "content": status}
        if packb is not None:
            await websocket.send_bytes(packb(msg, use_bin_type=True))
        else:
            import json

            await websocket.send_text(json.dumps(msg))

    async def _send_error(self, websocket: WebSocket, error: str) -> None:
        msg = {"type": "error", "content": error}
        try:
            if packb is not None:
                await websocket.send_bytes(packb(msg, use_bin_type=True))
            else:
                import json

                await websocket.send_text(json.dumps(msg))
        except Exception:
            pass
