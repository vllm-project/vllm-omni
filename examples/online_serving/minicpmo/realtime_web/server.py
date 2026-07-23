"""Static host and same-origin Realtime proxy for the full-duplex demo."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).parent / "app"
STATIC_DIR = APP_DIR / "static"


def _join_ws_url(base: str, path: str, query: str) -> str:
    return base.rstrip("/") + path + (("?" + query) if query else "")


async def _pump_client_to_backend(client: WebSocket, backend) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                await backend.close()
                return
            if message.get("text") is not None:
                await backend.send(message["text"])
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def _pump_backend_to_client(client: WebSocket, backend) -> None:
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                await client.send_text(message)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise


def _expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


def build_app(
    *,
    ws_backend: str = "ws://127.0.0.1:8099",
    model: str = "openbmb/MiniCPM-o-4_5",
    public_realtime_url: str | None = None,
    ref_audio: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Experimental Full-Duplex Web Demo")
    index_path = APP_DIR / "index.html"
    app_version_hash = hashlib.sha256()
    for asset_path in (
        STATIC_DIR / "app.js",
        STATIC_DIR / "pcm_worklet.js",
        STATIC_DIR / "playback_worklet.js",
    ):
        app_version_hash.update(asset_path.read_bytes())
    app_version = app_version_hash.hexdigest()[:12]

    ref_audio_uri: str | None = None
    if ref_audio:
        ref_path = Path(ref_audio)
        if not ref_path.is_file():
            raise SystemExit(f"--ref-audio not found: {ref_audio}")
        encoded = base64.b64encode(ref_path.read_bytes()).decode("ascii")
        ref_audio_uri = f"data:audio/wav;base64,{encoded}"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        config = json.dumps(
            {
                "model": model,
                "realtimePath": public_realtime_url or "v1/realtime",
                "refAudio": ref_audio_uri,
                "appVersion": app_version,
            },
            ensure_ascii=True,
        )
        html = (
            index_path.read_text(encoding="utf-8")
            .replace("__FULL_DUPLEX_CONFIG__", config)
            .replace("__FULL_DUPLEX_APP_VERSION__", app_version)
        )
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def healthz() -> Response:
        return Response(content="ok", media_type="text/plain")

    @app.websocket("/v1/realtime")
    async def realtime_proxy(websocket: WebSocket) -> None:
        await websocket.accept()
        query = urlencode(websocket.query_params.multi_items())
        backend_url = _join_ws_url(ws_backend, "/v1/realtime", query)
        logger.info("Proxying Realtime WebSocket to %s", backend_url)
        try:
            async with websockets.connect(
                backend_url,
                max_size=64 * 1024 * 1024,
            ) as backend:
                tasks = {
                    asyncio.create_task(_pump_client_to_backend(websocket, backend)),
                    asyncio.create_task(_pump_backend_to_client(websocket, backend)),
                }
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, BaseException) and not _expected_proxy_close(result):
                        raise result
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            return
        except Exception:
            logger.exception("Realtime WebSocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--ws-backend", default="ws://127.0.0.1:8099")
    parser.add_argument(
        "--public-realtime-url",
        help="Browser-visible ws:// or wss:// Realtime URL; defaults to the same-origin proxy.",
    )
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument(
        "--ref-audio",
        required=True,
        help=(
            "Reference voice wav for TTS voice cloning, e.g. the "
            "official MiniCPM-o-Demo assets/ref_audio/ref_minicpm_signature.wav"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        build_app(
            ws_backend=args.ws_backend,
            model=args.model,
            public_realtime_url=args.public_realtime_url,
            ref_audio=args.ref_audio,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
