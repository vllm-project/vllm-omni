# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm_omni.experimental.fullduplex.joyvl.bridges.backend import OpenAIBackend
from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import (
    DelegationBridge,
    ImageEditDelegationBridge,
    ImageGenDelegationBridge,
    OpenAIDelegationBridge,
    RoutingDelegationBridge,
    StubDelegationBridge,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import to_token_form
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import Summarizer
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.session import InteractionSession, StepResult


class SessionManager:
    def __init__(self, config: InteractionConfig) -> None:
        self.config = config
        self._backend = OpenAIBackend(
            config.main_backend_url, config.main_model, config.api_key, config.request_timeout_seconds
        )
        self._summarizer: Summarizer | None = None
        if config.enable_memory:
            summarizer_backend = OpenAIBackend(
                config.resolved_summarizer_url,
                config.resolved_summarizer_model,
                config.api_key,
                config.request_timeout_seconds,
            )
            self._summarizer = Summarizer(
                summarizer_backend,
                key_frames_per_chunk=config.mid_term_key_frames,
                mid_term_max_tokens=config.mid_term_max_tokens,
                long_term_max_tokens=config.long_term_max_tokens,
                max_pixels=config.max_pixels,
            )
        self._delegation = self._build_delegation(config) if config.enable_delegation else None
        self._sessions: dict[str, InteractionSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _build_delegation(config: InteractionConfig) -> DelegationBridge | None:
        # Returns None (delegation off) unless a real backend is configured. The stub is
        # only returned on explicit opt-in (--delegation-kind stub) for tests/demos — it
        # would otherwise fold fake answers into session memory.
        kind = config.delegation_kind
        url = config.delegation_backend_url

        def chat_bridge(backend_url: str) -> OpenAIDelegationBridge:
            return OpenAIDelegationBridge(
                backend_url,
                config.resolved_delegation_model,
                config.resolved_delegation_api_key,
                max_tokens=config.delegation_max_tokens,
                timeout=config.request_timeout_seconds,
            )

        if kind == "stub":
            return StubDelegationBridge()
        if kind == "router":
            if not (url or config.delegation_image_url or config.delegation_edit_url):
                return None
            return RoutingDelegationBridge(
                chat=chat_bridge(url) if url else None,
                image=ImageGenDelegationBridge(config.delegation_image_url, timeout=config.request_timeout_seconds)
                if config.delegation_image_url
                else None,
                edit=ImageEditDelegationBridge(
                    config.delegation_edit_url,
                    config.delegation_edit_model or config.main_model,
                    timeout=config.request_timeout_seconds,
                )
                if config.delegation_edit_url
                else None,
            )
        if url and kind == "image":
            return ImageGenDelegationBridge(url, timeout=config.request_timeout_seconds)
        if url and kind == "edit":
            return ImageEditDelegationBridge(
                url, config.resolved_delegation_model, timeout=config.request_timeout_seconds
            )
        if url:
            return chat_bridge(url)
        return None

    def _get(self, session_id: str) -> InteractionSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = InteractionSession(
                session_id,
                self.config,
                self._backend,
                summarizer=self._summarizer,
                delegation=self._delegation,
            )
            self._sessions[session_id] = session
            self._locks[session_id] = asyncio.Lock()
        return session

    async def step(self, session_id: str, frames: list[str], query: str | None) -> StepResult:
        await self._evict_expired()
        # capture session + lock together (no await between) so a concurrent reset cannot
        # swap them out from under us; the lock then serializes step against reset.
        session = self._get(session_id)
        lock = self._locks[session_id]
        async with lock:
            return await session.step(frames, query)

    async def reset(self, session_id: str) -> None:
        # Acquire the per-session lock so reset waits for any in-flight step() to finish
        # instead of mutating/dropping a session mid-request.
        lock = self._locks.get(session_id)
        if lock is None:
            return
        async with lock:
            session = self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
            if session is not None:
                await session.reset()

    async def set_persona(self, session_id: str, persona: str) -> bool:
        session = self._get(session_id)
        lock = self._locks[session_id]
        async with lock:
            return session.set_persona(persona)

    async def _evict_expired(self) -> None:
        ttl = self.config.session_timeout_seconds
        if ttl <= 0:
            return
        now = time.monotonic()
        expired = [
            sid
            for sid, sess in self._sessions.items()
            if now - sess.last_access > ttl and not self._locks[sid].locked()
        ]
        for sid in expired:
            await self.reset(sid)

    async def aclose(self) -> None:
        for sid in list(self._sessions):
            lock = self._locks.get(sid)
            if lock is not None:
                async with lock:
                    session = self._sessions.pop(sid, None)
                    self._locks.pop(sid, None)
                    if session is not None:
                        await session.aclose()
        await self._backend.aclose()
        if self._summarizer is not None:
            await self._summarizer.aclose()
        if self._delegation is not None:
            await self._delegation.aclose()


def _extract_frames_and_query(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    messages = payload.get("messages") or []
    frames: list[str] = []
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content or []:
            ptype = part.get("type")
            if ptype == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    frames.append(url)
            elif ptype == "text" and part.get("text"):
                texts.append(part["text"])
    query = "\n".join(t.strip() for t in texts if t.strip()) or None
    return frames, query


def _session_id(request: Request, payload: dict[str, Any]) -> str:
    return (
        request.headers.get("x-streaming-session")
        or request.headers.get("x-session-id")
        or payload.get("session_id")
        or payload.get("user")
        or "default"
    )


def _completion_response(model: str, result: StepResult) -> dict[str, Any]:
    action = result.action
    memory = {"long_term_memory": result.long_term_memory, "mid_term_summaries": result.mid_term_summaries}
    return {
        "id": f"intchat-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": to_token_form(action)},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
        "interaction": {
            "action": action.action.value,
            "spoke": action.spoke,
            "text": action.text,
            "delegated_question": action.delegated_question,
            "delegation": result.delegation,
            "chunk_index": result.chunk_index,
            "frame_index": result.frame_index,
            "inference_skipped": result.inference_skipped,
            "latency_ms": result.latency_ms,
            "memory": memory,
        },
    }


def create_app(config: InteractionConfig) -> FastAPI:
    manager = SessionManager(config)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # close httpx/AsyncOpenAI clients and cancel+await pending tasks on shutdown/reload
            await manager.aclose()

    app = FastAPI(title="vLLM-Omni Interaction Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": config.main_model, "object": "model", "owned_by": "vllm-omni"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        frames, query = _extract_frames_and_query(payload)
        if not frames:
            return JSONResponse({"error": "interaction server requires at least one image_url frame"}, status_code=400)
        result = await manager.step(_session_id(request, payload), frames, query)
        return JSONResponse(_completion_response(config.main_model, result))

    @app.post("/reset")
    @app.post("/v1/streaming/reset")
    async def reset(request: Request) -> dict[str, str]:
        payload = await request.json() if await request.body() else {}
        await manager.reset(_session_id(request, payload))
        return {"status": "reset"}

    @app.post("/v1/streaming/persona")
    async def persona(request: Request) -> dict[str, Any]:
        payload = await request.json() if await request.body() else {}
        ok = await manager.set_persona(_session_id(request, payload), payload.get("persona", "default"))
        return {"status": "ok" if ok else "unknown_persona"}

    return app


def _build_config(args: argparse.Namespace) -> InteractionConfig:
    config = InteractionConfig(
        main_backend_url=args.main_backend_url,
        main_model=args.main_model,
        persona=args.persona,
        enable_memory=not args.no_memory,
        summarizer_backend_url=args.summarizer_backend_url,
        summarizer_model=args.summarizer_model,
        enable_delegation=not args.no_delegation,
        delegation_backend_url=args.delegation_backend_url,
        delegation_model=args.delegation_model,
        delegation_api_key=args.delegation_api_key,
        delegation_kind=args.delegation_kind,
        delegation_image_url=args.delegation_image_url,
        delegation_edit_url=args.delegation_edit_url,
        delegation_edit_model=args.delegation_edit_model,
        force_silence_before_query=not args.no_force_silence,
    )
    if args.chunk_frames is not None:
        config.chunk_frames = args.chunk_frames
    config.response_dedup_threshold = args.response_dedup_threshold
    config.sampling.max_tokens = args.max_tokens
    config.sampling.temperature = args.temperature
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM-Omni streaming interaction server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--main-backend-url", default="http://127.0.0.1:8061/v1")
    parser.add_argument("--main-model", default="JoyAI-VL-Interaction-Preview")
    parser.add_argument("--persona", default="default", choices=["default", "silent", "talkative"])
    parser.add_argument("--summarizer-backend-url", default=None)
    parser.add_argument("--summarizer-model", default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument(
        "--response-dedup-threshold",
        type=float,
        default=1.0,
        help="1.0 drops only exact repeats (reference); < 1.0 also drops near-duplicate narration",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-memory", action="store_true", help="disable mid/long-term summarizer memory")
    parser.add_argument("--no-delegation", action="store_true", help="disable the delegation bridge")
    parser.add_argument(
        "--delegation-backend-url",
        default=None,
        help="OpenAI-compatible endpoint for the background brain; unset falls back to the stub",
    )
    parser.add_argument("--delegation-model", default=None, help="model name for the background brain")
    parser.add_argument(
        "--delegation-api-key",
        default=None,
        help="API key for the background brain endpoint (e.g. an Anthropic key for a claude-* model)",
    )
    parser.add_argument(
        "--delegation-kind",
        default="chat",
        choices=["chat", "image", "edit", "router", "stub"],
        help="chat = text/VL brain; image = text-to-image; edit = restyle the frame; "
        "router = dispatch by request; stub = canned demo/test answers. chat/image/edit/router "
        "need a backend URL — without one, delegation stays off.",
    )
    parser.add_argument("--delegation-image-url", default=None, help="router mode: text-to-image endpoint")
    parser.add_argument("--delegation-edit-url", default=None, help="router mode: image-edit endpoint")
    parser.add_argument("--delegation-edit-model", default=None, help="router mode: image-edit model name")
    parser.add_argument("--no-force-silence", action="store_true", help="run the model before any user query")
    args = parser.parse_args()

    app = create_app(_build_config(args))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
