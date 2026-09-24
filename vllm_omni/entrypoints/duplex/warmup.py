# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup warmup probe for duplex ``/v1/realtime``.

Moved out of ``api_server.py`` under P0.2 of #5227. The
``omni_run_server_worker`` scheduling and ``/v1/realtime`` hold-clients
preamble stay in ``api_server.py`` until P0.3.

Video-required models default to one short audio-plus-frame turn so the four
pipeline stages compile real shapes before a client is accepted. That is
separate from the per-stage JIT kernel registry, which can finish in ~0.01 s
without compiling anything.
"""

import asyncio
import json
import math
import struct
import time
import uuid
from io import BytesIO

import httpx
import pybase64 as base64
from vllm.logger import init_logger

logger = init_logger(__name__)

# First-request compiles (Triton mRoPE, vision pos-embed, Talker torch.compile,
# Code2Wav) landed ~30–40 s after the socket was already "ready" on a cold
# process, then the short reply still has to be spoken. Hold clients past that.
DUPLEX_WARMUP_CLIENT_WAIT_S = 180
DUPLEX_WARMUP_AUDIO_WAIT_S = 150
_WARMUP_SAMPLE_RATE_HZ = 16000
_WARMUP_AUDIO_SECONDS = 0.4
# 640x480 is divisible by Qwen3-VL's patch*merge (32) and above the
# processor's shortest_edge, so the vision tower sees a real camera-sized grid
# rather than an upscaled thumbnail.
_WARMUP_FRAME_WIDTH = 640
_WARMUP_FRAME_HEIGHT = 480
_WARMUP_INSTRUCTIONS = "启动暖机。请只用这四个字回答，不要输出 silent：暖机完成。"


def lookup_duplex_plugin(engine_client: object) -> object | None:
    engine = getattr(engine_client, "engine", None)
    plugin = getattr(engine, "plugin", None) if engine is not None else None
    if plugin is None:
        plugin = getattr(engine_client, "plugin", None)
    return plugin


def startup_warmup_kind(plugin: object | None, warmup_frames: int) -> str | None:
    """Which throwaway realtime session to run before admitting clients.

    ``video_turn`` is one short audio chunk plus one image. It is the default
    for models that require video, including when ``warmup_frames`` is 0.
    ``silent_frames`` is the older audio-only path and stays opt-in.
    A negative ``warmup_frames`` disables both. ``None`` means do not hold clients.
    """
    if warmup_frames < 0:
        return None
    required: object = frozenset()
    if plugin is not None and hasattr(plugin, "capabilities"):
        caps = plugin.capabilities(max_sessions=1)
        required = getattr(caps, "required_input_modalities", frozenset())
    if isinstance(required, frozenset | set) and "video" in required:
        return "video_turn"
    if warmup_frames > 0:
        return "silent_frames"
    return None


def _short_pcm_f32le_b64() -> str:
    """Non-silent PCM so Stage0 ASR actually runs.

    All-zero audio is either rejected or replaced before the ASR forward, which
    would leave the mRoPE kernel uncompiled.
    """
    n = int(_WARMUP_SAMPLE_RATE_HZ * _WARMUP_AUDIO_SECONDS)
    pcm = bytearray()
    for i in range(n):
        sample = 0.2 * math.sin(2.0 * math.pi * 220.0 * i / _WARMUP_SAMPLE_RATE_HZ)
        pcm += struct.pack("<f", sample)
    return base64.b64encode(bytes(pcm)).decode("ascii")


def _warmup_jpeg_b64() -> str:
    from PIL import Image

    image = Image.new("RGB", (_WARMUP_FRAME_WIDTH, _WARMUP_FRAME_HEIGHT), (255, 255, 255))
    image.paste((0, 0, 0), (_WARMUP_FRAME_WIDTH // 2, 0, _WARMUP_FRAME_WIDTH, _WARMUP_FRAME_HEIGHT))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _warmup_duplex_realtime(app, args, warmup_frames: int) -> None:
    """Run one throwaway realtime session before real clients are admitted.

    Video-required models send one short non-silent audio chunk and one image
    so each stage compiles the shape a real turn uses. Audio-primary models
    keep the opt-in silent-frame path. ``/v1/realtime`` holds other connections
    until this finishes (see ``duplex_warmup_done``).
    """
    warmup_done = getattr(app.state, "duplex_warmup_done", None)
    try:
        try:
            import websockets
        except ImportError:
            logger.warning("Duplex warmup skipped: the 'websockets' package is not installed.")
            return
        engine_client = getattr(app.state, "engine_client", None)
        plugin = lookup_duplex_plugin(engine_client) if engine_client is not None else None
        kind = startup_warmup_kind(plugin, warmup_frames)
        if kind is None:
            return
        served = getattr(args, "served_model_name", None)
        if isinstance(served, list | tuple) and served:
            model_name = served[0]
        elif isinstance(served, str) and served:
            model_name = served
        else:
            model_name = args.model
        # One silence unit as the engine-side plugin defines it (DuplexOmniEngine.plugin).
        frame_samples = int(getattr(plugin, "silence_continuation_samples", 16000))
        native_append = plugin.capabilities(max_sessions=1).supports_core_resumable_request if plugin else True
        from vllm_omni.clients.duplex import build_realtime_url

        url = (
            build_realtime_url(f"ws://127.0.0.1:{args.port}/v1/realtime", model_name, autostart=False)
            + "&vllm_omni_warmup=1"
        )
        if kind == "video_turn":
            logger.info(
                "Duplex warmup starting: one %.2f s audio chunk and one %dx%d frame.",
                _WARMUP_AUDIO_SECONDS,
                _WARMUP_FRAME_WIDTH,
                _WARMUP_FRAME_HEIGHT,
            )
        else:
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
        session: dict[str, object] = {
            "session_id": f"warmup-{uuid.uuid4().hex[:8]}",
            "model": model_name,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "idle_timeout_s": 60,
            "turn_detection": None,
            "extra_body": {"auto_response": native_append},
        }
        if kind == "video_turn":
            # The default AURA prompt answers with <|silent|> unless the user
            # asked for something. This session is thrown away; force one short
            # spoken reply so Talker and Code2Wav compile too. auto_response
            # matches the web demo: a commit must actually start a turn.
            session["instructions"] = _WARMUP_INSTRUCTIONS
            session["extra_body"] = {"auto_response": True}
        async with websockets.connect(url, max_size=64 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": session}))
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
            if kind == "video_turn":
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": _short_pcm_f32le_b64(),
                            "format": "pcm_f32le",
                            "sample_rate_hz": _WARMUP_SAMPLE_RATE_HZ,
                            "is_speech": True,
                            "video_frames": [_warmup_jpeg_b64()],
                        }
                    )
                )
                sent = 1
            else:
                silence = base64.b64encode(bytes(frame_samples * 4)).decode("ascii")
                while sent < warmup_frames:
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": silence}))
                    sent += 1
                    await asyncio.sleep(0.08)
            if not native_append:
                commit: dict[str, object] = {"type": "input_audio_buffer.commit", "final": True}
                if kind == "video_turn":
                    commit["create_response"] = True
                await ws.send(json.dumps(commit))
            # First output audio means Code2Wav ran, so the Talker compile
            # ahead of it has run too. Do not wait out the whole utterance.
            saw_audio = await _recv_until(
                lambda e: e.get("type") == "response.output_audio.delta",
                DUPLEX_WARMUP_AUDIO_WAIT_S,
            )
            # Close the session EXPLICITLY: a bare websocket close parks the
            # session in its disconnect grace window, where it would keep
            # occupying the max_sessions=1 slot and block the first client.
            await ws.send(json.dumps({"type": "session.close"}))
            await _recv_until(lambda e: e.get("type") == "session.closed", 10)
        if kind == "video_turn":
            logger.info(
                "Duplex warmup finished in %.1f s (one audio+frame turn, first audio %s).",
                time.time() - start,
                "received" if saw_audio else "NOT received",
            )
        else:
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


__all__ = [
    "DUPLEX_WARMUP_CLIENT_WAIT_S",
    "_warmup_duplex_realtime",
    "lookup_duplex_plugin",
    "startup_warmup_kind",
]
