# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in packaged endpoints. Standard vLLM endpoints remain available."""

import json
from typing import Literal

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


def local_backend(scope):
    # Use the actual listening socket, never Host/forwarded headers or user URL.
    server = scope.get("server")
    if not server or len(server) != 2 or not isinstance(server[1], int) or not 0 < server[1] <= 65535:
        raise ValueError("Missing actual server port")
    return f"http://127.0.0.1:{server[1]}"


def stream_payload(request):
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list) or not request["messages"]:
        raise ValueError("Provide a normal chat request with nonempty messages")
    if request.get("n", 1) != 1:
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

    serving = raw_request.app.state.openai_serving_chat
    if not body.text.strip():
        raise HTTPException(400, "Nonempty text required")
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
    async def generate():
        source = stream_sentence(raw_request.app.state.openai_serving_chat.engine_client, body.text, body.language)
        try:
            async for chunk in source:
                wav = chunk.pop("wav")
                yield json.dumps({"type": "audio", "audio_data": base64.b64encode(wav).decode(), **chunk}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        finally:
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
    # Preserve optional API authentication when calling the same local server.
    authorization = raw_request.headers.get("authorization")
    headers = {"Authorization": authorization} if authorization else {}

    async def generate():
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=10), headers=headers, trust_env=False
        ) as client:
            source = stream_reply(client, backend, payload, body.audio_output, body.speech_language)
            try:
                async for line in source:
                    yield line
            finally:
                await source.aclose()

    return StreamingResponse(
        generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    )


# Mounted only for the DRVEXO pipeline after app-state initialization.


router = APIRouter()
router.add_api_route("/v1/drvexo/decision", with_cancellation(general_decision), methods=["POST"])
router.add_api_route("/v1/drvexo/speech", with_cancellation(sentence_speech), methods=["POST"])
router.add_api_route("/v1/drvexo/speech/stream", sentence_speech_stream, methods=["POST"])
router.add_api_route("/v1/drvexo/chat/stream", chat_stream, methods=["POST"])
