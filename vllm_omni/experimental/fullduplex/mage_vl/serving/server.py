# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Production WebSocket transport for Mage-VL full-duplex sessions."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import inspect
import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from vllm_omni.experimental.fullduplex.core import protocol as event_protocol
from vllm_omni.experimental.fullduplex.core.adapter import DuplexAdapter
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.mage_vl import MageVLDuplexAdapter, MageVLDuplexRuntime

AdapterFactory = Callable[[], DuplexAdapter | Any]


@dataclass(frozen=True)
class MageVLServingConfig:
    model: str = "microsoft/Mage-VL"
    max_sessions: int = 32
    idle_timeout_s: float = 300.0
    max_message_bytes: int = 8 * 1024 * 1024
    auth_token: str | None = None

    def __post_init__(self) -> None:
        if self.max_sessions <= 0 or self.idle_timeout_s <= 0 or self.max_message_bytes <= 0:
            raise ValueError("max_sessions, idle_timeout_s, and max_message_bytes must be positive")


class _SessionLease:
    def __init__(self, session_id: str, runtime: MageVLDuplexRuntime) -> None:
        self.session_id = session_id
        self.runtime = runtime
        self.last_activity = time.monotonic()

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class _SessionRegistry:
    def __init__(self, config: MageVLServingConfig, factory: AdapterFactory) -> None:
        self.config = config
        self.factory = factory
        self._leases: dict[str, _SessionLease] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, requested_id: str | None) -> _SessionLease | None:
        async with self._lock:
            if len(self._leases) >= self.config.max_sessions:
                return None
            session_id = requested_id or f"mage-{uuid.uuid4().hex}"
            if not _valid_session_id(session_id) or session_id in self._leases:
                return None
            adapter = self.factory()
            if inspect.isawaitable(adapter):
                adapter = await adapter
            if not isinstance(adapter, MageVLDuplexAdapter):
                raise TypeError("adapter factory must return a MageVLDuplexAdapter")
            capability = adapter.capabilities()
            session = DuplexSession(
                session_id,
                DuplexSessionConfig(
                    input_modalities=tuple(capability.input_modalities),
                    output_modalities=tuple(capability.output_modalities),
                    proactive=capability.proactive,
                ),
            )
            lease = _SessionLease(session_id, MageVLDuplexRuntime(session, adapter))
            self._leases[session_id] = lease
            return lease

    async def release(self, session_id: str) -> None:
        async with self._lock:
            self._leases.pop(session_id, None)

    async def close(self) -> None:
        async with self._lock:
            leases = list(self._leases.values())
            self._leases.clear()
        for lease in leases:
            with contextlib.suppress(asyncio.CancelledError):
                await lease.runtime.close()

    @property
    def active_count(self) -> int:
        return len(self._leases)


def _valid_session_id(value: str) -> bool:
    return 0 < len(value) <= 128 and all(char.isalnum() or char in "-_." for char in value)


def _authorized(websocket: WebSocket, token: str | None) -> bool:
    if token is None:
        return True
    return secrets.compare_digest(websocket.headers.get("authorization", ""), f"Bearer {token}")


def _event_from_text(text: str, max_bytes: int) -> dict[str, Any]:
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("message exceeds configured max_message_bytes")
    event = json.loads(text)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise ValueError("event must be a JSON object with a string type")
    supported = {
        event_protocol.INPUT_APPEND,
        event_protocol.INPUT_COMMIT,
        event_protocol.RESPONSE_CREATE,
        event_protocol.RESPONSE_CANCEL,
        event_protocol.PLAYBACK_ACK,
        event_protocol.CLOSE,
    }
    if event["type"] not in supported:
        raise ValueError(f"unsupported event type: {event['type']}")
    if event["type"] == event_protocol.INPUT_APPEND and not isinstance(event.get("modality"), str):
        raise ValueError("input.append requires a modality")
    return event


def load_adapter_factory(path: str) -> AdapterFactory:
    module_name, separator, attribute_name = path.partition(":")
    if not separator:
        raise ValueError("adapter path must use module:attribute syntax")
    value = getattr(importlib.import_module(module_name), attribute_name)
    if isinstance(value, DuplexAdapter):
        return lambda: value
    if not callable(value):
        raise TypeError(f"adapter target {path!r} is not callable")
    return value


def create_app(adapter_factory: AdapterFactory, config: MageVLServingConfig | None = None) -> FastAPI:
    config = config or MageVLServingConfig()
    registry = _SessionRegistry(config, adapter_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await registry.close()

    app = FastAPI(title="Mage-VL Duplex Server", version="1", lifespan=lifespan)
    app.state.session_registry = registry

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return JSONResponse({"status": "ready", "active_sessions": registry.active_count})

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": config.model, "object": "model", "owned_by": "microsoft"}]}

    @app.websocket("/v1/mage-vl/duplex")
    async def duplex(websocket: WebSocket) -> None:
        if not _authorized(websocket, config.auth_token):
            await websocket.close(code=4401, reason="unauthorized")
            return
        await websocket.accept()
        lease = await registry.acquire(websocket.query_params.get("session_id"))
        if lease is None:
            await websocket.close(code=4429, reason="session unavailable")
            return

        async def inputs() -> AsyncIterator[dict[str, Any]]:
            while True:
                event = _event_from_text(
                    await asyncio.wait_for(websocket.receive_text(), config.idle_timeout_s), config.max_message_bytes
                )
                lease.touch()
                yield event
                if event.get("type") == event_protocol.CLOSE:
                    return

        async def emit(event: dict[str, Any]) -> None:
            await websocket.send_json(event)

        capability = lease.runtime.capabilities
        try:
            await websocket.send_json(
                {
                    "type": "session.created",
                    "session_id": lease.session_id,
                    "model": config.model,
                    "capabilities": {
                        "input_modalities": sorted(capability.input_modalities),
                        "output_modalities": sorted(capability.output_modalities),
                        "proactive": capability.proactive,
                    },
                }
            )
            await lease.runtime.run(inputs(), emit)
        except WebSocketDisconnect:
            await lease.runtime.cancel_response()
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "message": "idle timeout"})
        except (ValueError, json.JSONDecodeError) as error:
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "message": str(error)})
            await lease.runtime.cancel_response()
        finally:
            await registry.release(lease.session_id)
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Mage-VL full-duplex WebSocket sessions.")
    parser.add_argument("--adapter", help="optional module:attribute adapter factory")
    parser.add_argument("--model", default="microsoft/Mage-VL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--idle-timeout", type=float, default=300.0)
    parser.add_argument("--max-message-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--auth-token")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-impl", choices=("flash_attention_2", "sdpa", "eager"), default="sdpa")
    parser.add_argument("--video-backend", choices=("frames", "codec"), default="frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=4)
    args = parser.parse_args()
    if args.adapter:
        adapter_factory = load_adapter_factory(args.adapter)
    else:
        from vllm_omni.experimental.fullduplex.mage_vl.serving.backend import MageVLTransformersBackend

        backend = MageVLTransformersBackend(
            args.model,
            device=args.device,
            attn_impl=args.attn_impl,
            video_backend=args.video_backend,
            num_frames=args.num_frames,
            target_fps=args.target_fps,
            max_new_tokens=args.max_new_tokens,
            gate_threshold=args.gate_threshold,
            window_size=args.window_size,
        )
        backend.load()
        adapter_factory = backend.adapter_factory
    app = create_app(
        adapter_factory,
        MageVLServingConfig(
            model=args.model,
            max_sessions=args.max_sessions,
            idle_timeout_s=args.idle_timeout,
            max_message_bytes=args.max_message_bytes,
            auth_token=args.auth_token,
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
