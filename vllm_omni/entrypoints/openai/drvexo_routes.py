# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in packaged endpoints. Standard vLLM endpoints remain available."""

import ipaddress
import json
from collections.abc import Mapping
from typing import Literal

import anyio
import httpx
import pybase64 as base64
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from vllm.entrypoints.serve.utils.api_utils import with_cancellation

from .drvexo_decision import general_decision
from .drvexo_speech import collect_sentence, stream_sentence
from .drvexo_streaming import stream_reply


class SpeechText(BaseModel):
    text: str = Field(min_length=1, max_length=512)
    language: Literal["Auto", "Chinese", "English"] = "Auto"
    model: str | None = None


class StreamChat(BaseModel):
    request: dict
    audio_output: bool = True
    speech_language: Literal["Auto", "Chinese", "English"] = "Auto"


def local_backend(scope: Mapping[str, object]) -> str:
    """Use the actual HTTP listening socket, never Host/forwarded headers."""
    server = scope.get("server")
    if scope.get("scheme", "http") != "http":
        raise HTTPException(503, "DRVEXO chat streaming requires a local HTTP listener; TLS is not supported")
    if not isinstance(server, (tuple, list)) or len(server) != 2:
        raise HTTPException(503, "DRVEXO chat streaming requires a TCP listener; Unix sockets are not supported")
    host, port = server
    if not isinstance(host, str) or not isinstance(port, int) or not 0 < port <= 65535:
        raise HTTPException(503, "DRVEXO chat streaming has no valid listening socket")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise HTTPException(503, "DRVEXO chat streaming requires a numeric listening address") from exc
    if address.is_unspecified:
        address = ipaddress.ip_address("::1" if address.version == 6 else "127.0.0.1")
    host = f"[{address}]" if address.version == 6 else str(address)
    return f"http://{host}:{port}"


def backend_headers(request: Request) -> dict[str, str]:
    """Keep end-to-end auth/tracing headers when re-entering this server."""
    excluded = {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "forwarded",
    }
    excluded.update(part.strip().lower() for part in request.headers.get("connection", "").split(","))
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in excluded and not key.lower().startswith("x-forwarded-")
    }


def stream_payload(request):
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list) or not request["messages"]:
        raise ValueError("Provide a normal chat request with nonempty messages")
    if request.get("n") not in (None, 1):
        raise ValueError("Sentence streaming requires one response choice")
    if request.get("tools") or request.get("functions"):
        raise ValueError("Tool-call speech streaming is not supported")
    payload = {**request, "stream": True, "modalities": ["text"], "stream_options": {"include_usage": True}}
    payload.pop("audio", None)
    # No system/user/history rewriting, answer hints or intent interception.
    return payload


async def sentence_speech(body: SpeechText, raw_request: Request):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.sampling_params import RequestOutputKind

    if not body.text.strip():
        raise HTTPException(400, "Nonempty text required")
    serving = raw_request.app.state.openai_serving_chat
    final = await collect_sentence(serving.engine_client, body.text, body.language, RequestOutputKind.FINAL_ONLY)
    req = ChatCompletionRequest(
        model=body.model or serving.models.model_name(None),
        messages=[{"role": "user", "content": body.text}],
        modalities=["audio"],
        audio={"format": "wav"},
        stream=False,
    )
    choices = serving._create_audio_choice(final, "assistant", req, stream=False)
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].message.audio is None:
        raise HTTPException(503, "Invalid sentence waveform")
    raw = base64.b64decode(choices[0].message.audio.data, validate=True)
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise HTTPException(503, "Invalid sentence WAV envelope")
    return Response(
        raw,
        media_type="audio/wav",
        headers={"X-Speech-Source": "caller_text", "X-Speech-Entry-Stage": "1", "X-Speech-Language": body.language},
    )


async def sentence_speech_stream(body: SpeechText, raw_request: Request):
    if not body.text.strip():
        raise HTTPException(400, "Nonempty text required")

    async def generate():
        source = stream_sentence(raw_request.app.state.openai_serving_chat.engine_client, body.text, body.language)
        try:
            async for chunk in source:
                wav = chunk.pop("wav")
                yield json.dumps({"type": "audio", "audio_data": base64.b64encode(wav).decode(), **chunk}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        finally:
            with anyio.CancelScope(shield=True):
                await source.aclose()

    return StreamingResponse(
        generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    )


async def chat_stream(body: StreamChat, raw_request: Request):
    try:
        backend = local_backend(raw_request.scope)
        payload = stream_payload(body.request)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # This internal HTTP hop intentionally re-enters middleware/authentication.
    headers = backend_headers(raw_request)

    async def generate():
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=10), headers=headers, trust_env=False
        ) as client:
            source = stream_reply(client, backend, payload, body.audio_output, body.speech_language)
            try:
                async for line in source:
                    yield line
            finally:
                with anyio.CancelScope(shield=True):
                    await source.aclose()

    return StreamingResponse(
        generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    )


# Mounted only for the DRVEXO pipeline after app-state initialization.


router = APIRouter()
router.add_api_route("/v1/drvexo/decision", with_cancellation(general_decision), methods=["POST"])
router.add_api_route("/v1/drvexo/speech", with_cancellation(sentence_speech), methods=["POST"])
router.add_api_route("/v1/drvexo/speech/stream", with_cancellation(sentence_speech_stream), methods=["POST"])
router.add_api_route("/v1/drvexo/chat/stream", chat_stream, methods=["POST"])
