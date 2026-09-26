# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import aiohttp
from aiohttp import web

try:
    import cn2an  # optional: Arabic-digit -> Chinese number normalization
except ImportError:
    cn2an = None

logger = logging.getLogger("tts_bridge")


def _normalize(text: str) -> str:
    """Spell out Arabic digits in Chinese so short counting outputs synthesize
    intelligibly: bare "1次" garbles on Qwen3-TTS (and others), while "一次"
    is clean. ``cn2an.transform`` rewrites digits in place ("第3次"->"第三次")
    and is pure-Python (~microseconds). No-op if cn2an is unavailable."""
    if not cn2an or not text:
        return text
    try:
        return cn2an.transform(text, "an2cn")
    except Exception as err:  # never let TN break synthesis
        logger.debug("cn2an normalize skipped: %s", err)
        return text


async def _pump_backend_to_front(back: aiohttp.ClientWebSocketResponse, front: web.WebSocketResponse) -> None:
    async for msg in back:
        if msg.type == aiohttp.WSMsgType.BINARY:
            await front.send_bytes(msg.data)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            event = json.loads(msg.data).get("type")
            if event == "session.done":
                await front.send_str(json.dumps({"type": "response.done"}))
                return
            if event == "error":
                await front.send_str(json.dumps({"type": "error", "error": "tts backend error"}))
                return
        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            return


async def _handle(request: web.Request) -> web.WebSocketResponse:
    front = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await front.prepare(request)
    cfg = request.app["cfg"]

    session = aiohttp.ClientSession()
    back: aiohttp.ClientWebSocketResponse | None = None
    pump: asyncio.Task | None = None
    parts: list[str] = []
    try:
        async for msg in front:
            if msg.type != web.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)

            if "config" in data and back is None:
                voice = (data["config"] or {}).get("voice") or cfg["voice"]
                back = await session.ws_connect(cfg["backend_url"], max_msg_size=0)
                await back.send_str(
                    json.dumps({"type": "session.config", "response_format": cfg["response_format"], "voice": voice})
                )
                pump = asyncio.create_task(_pump_backend_to_front(back, front))
            elif data.get("type") == "input_text.append" and back is not None:
                parts.append(data.get("text", ""))
            elif data.get("type") == "input_text.commit" and back is not None:
                # Normalize the full utterance once (digits may span chunks), then send.
                await back.send_str(json.dumps({"type": "input.text", "text": _normalize("".join(parts))}))
                parts = []
                await back.send_str(json.dumps({"type": "input.done"}))
                if pump is not None:
                    await pump
                break
    except Exception as err:
        logger.warning("tts bridge error: %s", err)
    finally:
        if pump is not None and not pump.done():
            pump.cancel()
        if back is not None and not back.closed:
            await back.close()
        await session.close()
        if not front.closed:
            await front.close()
    return front


def create_app(backend_url: str, voice: str, response_format: str) -> web.Application:
    app = web.Application()
    app["cfg"] = {"backend_url": backend_url, "voice": voice, "response_format": response_format}
    app.router.add_get("/v1/tts", _handle)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="JoyVL webui <-> Qwen3-TTS websocket bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--backend-url", default="ws://127.0.0.1:8091/v1/audio/speech/stream")
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--response-format", default="pcm")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(args.backend_url, args.voice, args.response_format), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
