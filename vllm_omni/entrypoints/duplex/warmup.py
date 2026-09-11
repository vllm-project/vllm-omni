# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup warmup probe for duplex ``/v1/realtime``.

Moved out of ``api_server.py`` under P0.2 of #5227. The
``omni_run_server_worker`` scheduling and ``/v1/realtime`` hold-clients
preamble stay in ``api_server.py`` until P0.3.
"""

import asyncio
import json
import time
import uuid

import httpx
import pybase64 as base64
from vllm.logger import init_logger

logger = init_logger(__name__)


async def _warmup_duplex_realtime(app, args, warmup_frames: int) -> None:
    """Run silent frames through a throwaway realtime session at startup.

    Self-connects over the server's own websocket endpoint so the warmup
    exercises the exact production path (serving adapter, duplex control
    plane, every pipeline stage). One-time costs — kernel JIT compilation,
    first prefill/decode paths, codec caches — are paid here instead of by
    the first real client; ``/v1/realtime`` holds other connections until
    the warmup finishes (see ``duplex_warmup_done``).
    """
    warmup_done = getattr(app.state, "duplex_warmup_done", None)
    try:
        try:
            import websockets
        except ImportError:
            logger.warning("Duplex warmup skipped: the 'websockets' package is not installed.")
            return
        served = getattr(args, "served_model_name", None)
        if isinstance(served, list | tuple) and served:
            model_name = served[0]
        elif isinstance(served, str) and served:
            model_name = served
        else:
            model_name = args.model
        adapter = getattr(getattr(app.state, "openai_serving_duplex", None), "_serving_runtime_adapter", None)
        frame_samples = int(getattr(adapter, "silence_continuation_samples", 16000))
        silence = base64.b64encode(bytes(frame_samples * 4)).decode("ascii")
        from vllm_omni.experimental.fullduplex.client import build_realtime_url

        url = (
            build_realtime_url(f"ws://127.0.0.1:{args.port}/v1/realtime", model_name, autostart=False)
            + "&vllm_omni_warmup=1"
        )
        logger.info("Duplex warmup starting: %d silent frames of %d samples.", warmup_frames, frame_samples)
        start = time.time()
        # The socket is bound before uvicorn finishes starting, so a plain
        # connect can be accepted and then reset. Wait for /health first.
        async with httpx.AsyncClient() as http:
            for _ in range(120):
                try:
                    if (await http.get(f"http://127.0.0.1:{args.port}/health", timeout=2)).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                logger.warning("Duplex warmup: /health never became ready; skipping warmup.")
                return
        async with websockets.connect(url, max_size=64 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "session_id": f"warmup-{uuid.uuid4().hex[:8]}",
                            "model": model_name,
                            "modalities": ["audio", "text"],
                            "input_audio_format": "pcm_f32le",
                            "output_audio_format": "pcm16",
                            "idle_timeout_s": 60,
                            "turn_detection": None,
                            "extra_body": {"auto_response": True},
                        },
                    }
                )
            )
            saw_audio = False
            sent = 0

            async def _recv_until(predicate, timeout_s: float) -> bool:
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.05, deadline - time.monotonic()))
                    except (TimeoutError, asyncio.TimeoutError):
                        return False
                    event = json.loads(raw)
                    if predicate(event):
                        return True
                return False

            if not await _recv_until(lambda e: e.get("type") == "session.created", 30):
                logger.warning("Duplex warmup: no session.created within 30 s; aborting warmup.")
                return
            while sent < warmup_frames:
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": silence}))
                sent += 1
                await asyncio.sleep(0.08)
            # Wait for the pipeline's first audio output so every stage ran.
            saw_audio = await _recv_until(lambda e: e.get("type") == "response.output_audio.delta", 30)
            # Close the session EXPLICITLY: a bare websocket close parks the
            # session in its disconnect grace window, where it would keep
            # occupying the max_sessions=1 slot and block the first client.
            await ws.send(json.dumps({"type": "session.close"}))
            await _recv_until(lambda e: e.get("type") == "session.closed", 10)
        logger.info(
            "Duplex warmup finished in %.1f s (%d silent frames, first audio %s).",
            time.time() - start,
            sent,
            "received" if saw_audio else "NOT received",
        )
    except Exception:
        logger.exception("Duplex warmup failed; continuing to serve (first session pays cold-start costs).")
    finally:
        if warmup_done is not None:
            warmup_done.set()


__all__ = ["_warmup_duplex_realtime"]
