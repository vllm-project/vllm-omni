# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import wave

import aiohttp
from aiohttp import web

_HEADER = struct.Struct(">iii")
_SAMPLE_RATE = 16000
_DEFAULT_PROMPT = "Transcribe the audio."
_ASR_TEXT_TAG = "<asr_text>"


def _clean_transcript(raw: str) -> str:
    """Qwen3-ASR emits ``language <Lang><asr_text><transcript>``; keep the transcript."""
    text = raw or ""
    marker = text.rfind(_ASR_TEXT_TAG)
    if marker != -1:
        text = text[marker + len(_ASR_TEXT_TAG) :]
    return text.replace("</asr_text>", "").strip()


def _pcm_to_wav_b64(pcm: bytes, sample_rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def _result(text: str, *, final: bool) -> dict:
    return {
        "asr_response": {
            "event_type": "IS_FINAL" if final else "IS_PARTIAL",
            "recognition_result": {"hypothesis": [{"text": text, "confidence": None}]},
        },
        "mid": "",
        "code": 0,
        "msg": "ok",
    }


async def _transcribe(cfg: dict, pcm: bytes) -> str:
    if not pcm:
        return ""
    wav_b64 = _pcm_to_wav_b64(pcm, cfg["sample_rate"])
    body = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                    {"type": "text", "text": cfg["prompt"]},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": cfg["max_tokens"],
    }
    url = cfg["backend_url"].rstrip("/") + "/v1/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body) as resp:
            payload = await resp.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return _clean_transcript(choices[0].get("message", {}).get("content") or "")


async def _handle(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    cfg = request.app["cfg"]
    pcm = bytearray()
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) >= _HEADER.size:
                seqid, _, _ = _HEADER.unpack(msg.data[: _HEADER.size])
                pcm.extend(msg.data[_HEADER.size :])
                if seqid < 0:
                    text = await _transcribe(cfg, bytes(pcm))
                    await ws.send_str(json.dumps(_result(text, final=True), ensure_ascii=False))
                    pcm.clear()
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
    finally:
        if not ws.closed:
            await ws.close()
    return ws


def create_app(backend_url: str, model: str, sample_rate: int, prompt: str, max_tokens: int) -> web.Application:
    app = web.Application()
    app["cfg"] = {
        "backend_url": backend_url,
        "model": model,
        "sample_rate": sample_rate,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    app.router.add_get("/v1/asr", _handle)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="JoyVL webui <-> Qwen3-ASR (chat input_audio) bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8094")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--sample-rate", type=int, default=_SAMPLE_RATE)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    app = create_app(args.backend_url, args.model, args.sample_rate, args.prompt, args.max_tokens)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
