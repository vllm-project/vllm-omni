# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup warmup probe for duplex ``/v1/realtime``.

Moved out of ``api_server.py`` under P0.2 of #5227. The
``omni_run_server_worker`` scheduling stays in ``api_server.py`` until P0.3;
the hold-clients preamble is ``wait_for_duplex_warmup`` here.

Video-required models default to one short audio-plus-frame turn so the four
pipeline stages compile real shapes before a client session is admitted. That is
separate from the per-stage JIT kernel registry, which can finish in ~0.01 s
without compiling anything.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, TypedDict, cast

import httpx
import pybase64 as base64
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.config import DuplexCapabilities
    from vllm_omni.engine.duplex.process_engine import ProcessDuplexEngineClient
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

logger = init_logger(__name__)

# First-request compiles (Triton mRoPE, vision pos-embed, Talker torch.compile,
# Code2Wav) landed ~30–40 s after the socket was already "ready" on a cold
# process, then the short reply still has to be spoken. Hold clients past that.
DUPLEX_WARMUP_CLIENT_WAIT_S = 180
DUPLEX_WARMUP_AUDIO_WAIT_S = 150

#: Outcomes of the warmup's audio/text wait. They stay distinct: a timeout is not a
#: listen, and neither of them is a first audio delta or completed text response.
WARMUP_OUTCOME_AUDIO = "audio"
WARMUP_OUTCOME_TEXT = "text"
WARMUP_OUTCOME_LISTEN = "listen"
WARMUP_OUTCOME_TIMEOUT = "timeout"


def classify_warmup_outcome(
    event_type: object, *, kind: str | None, text_only: bool = False, response: object = None
) -> str | None:
    """Map a wire event to a warmup outcome, or ``None`` to keep waiting.

    First audio ends the wait outside text-only mode; text-only mode waits for
    a completed ``response.done`` with text, since a listen can precede that
    reply. Otherwise ``response.listen`` ends it only for the silent-frames
    probe: that model refused the unit because it heard no turn, and there is
    then no audio path to warm. The video probe forces a spoken reply on
    purpose (it needs speech to warm Talker/Code2Wav), so a listen there is not
    success and the wait continues.
    """
    if text_only and event_type == "response.done" and _completed_text_response(response):
        return WARMUP_OUTCOME_TEXT
    if event_type == "response.output_audio.delta" and not text_only:
        return WARMUP_OUTCOME_AUDIO
    if event_type == "response.listen" and kind != "video_turn" and not text_only:
        return WARMUP_OUTCOME_LISTEN
    return None


async def wait_for_warmup_outcome(
    recv: Callable[[], Awaitable[dict[str, object]]],
    timeout_s: float,
    *,
    kind: str | None,
    text_only: bool = False,
) -> str:
    """Wait for the warmup to finish and report audio, text, listen or timeout."""
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return WARMUP_OUTCOME_TIMEOUT
        try:
            event = await asyncio.wait_for(recv(), timeout=remaining)
        except (TimeoutError, asyncio.TimeoutError):
            return WARMUP_OUTCOME_TIMEOUT
        outcome = classify_warmup_outcome(
            event.get("type"), kind=kind, text_only=text_only, response=event.get("response")
        )
        if outcome is not None:
            return outcome


_WARMUP_SAMPLE_RATE_HZ = 16000
_WARMUP_AUDIO_SECONDS = 0.4
# 640x480 is divisible by Qwen3-VL's patch*merge (32) and above the
# processor's shortest_edge, so the vision tower sees a real camera-sized grid
# rather than an upscaled thumbnail.
_WARMUP_FRAME_WIDTH = 640
_WARMUP_FRAME_HEIGHT = 480
_WARMUP_INSTRUCTIONS = "启动暖机。请只用这四个字回答，不要输出 silent：暖机完成。"


class _WarmupAudioAppendKwargs(TypedDict, total=False):
    format: str
    sample_rate_hz: int
    is_speech: bool
    video_frames: list[str]


@dataclass(frozen=True)
class _ProcessWarmupProfile:
    """Serializable startup facts, without loading a model plugin in the API process."""

    engine: ProcessDuplexEngineClient

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return self.engine.duplex_capabilities

    def silence_unit_payload(self) -> dict[str, object]:
        return dict(self.engine.metadata.warmup_audio)


def lookup_duplex_plugin(engine_client: object) -> object | None:
    from vllm_omni.engine.duplex.process_engine import ProcessDuplexEngineClient

    engine = getattr(engine_client, "engine", None)
    if isinstance(engine, ProcessDuplexEngineClient):
        return _ProcessWarmupProfile(engine)
    plugin = getattr(engine, "plugin", None) if engine is not None else None
    if plugin is None:
        plugin = getattr(engine_client, "plugin", None)
    return plugin


def _warmup_modalities(engine_client: object) -> list[str]:
    capabilities = getattr(engine_client, "duplex_capabilities", None)
    return list(getattr(capabilities, "default_output_modalities", ("audio", "text")))


def _warmup_timeout_reason(kind: str | None, text_only: bool) -> str:
    """What a timed-out warmup waited for; only the silent-frames audio probe ends on a listen."""
    if text_only:
        return "no completed text reply"
    if kind == "video_turn":
        return "no audio"
    return "neither audio nor a listen event"


def _completed_text_response(response: object) -> bool:
    """A terminal response includes text only after the projector completes it."""
    if not isinstance(response, dict) or response.get("status") != "completed":
        return False
    output = response.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "output_text" and bool(part.get("text")) for part in content
        ):
            return True
    return False


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


_DEFAULT_SILENCE_SAMPLES = 16000
_DEFAULT_SILENCE_SAMPLE_RATE_HZ = 16000


def warmup_silence_unit(plugin: object | None) -> dict[str, object]:
    """The append the silent-frames warmup sends: the plugin's silence unit, or the 16 kHz default."""
    factory = getattr(plugin, "silence_unit_payload", None)
    if callable(factory):
        unit = dict(factory())
        if isinstance(unit.get("audio"), str) and isinstance(unit.get("format"), str):
            unit.setdefault("sample_rate_hz", _DEFAULT_SILENCE_SAMPLE_RATE_HZ)
            return unit
    samples = int(getattr(plugin, "silence_continuation_samples", _DEFAULT_SILENCE_SAMPLES))
    return {
        "audio": base64.b64encode(bytes(samples * 4)).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": _DEFAULT_SILENCE_SAMPLE_RATE_HZ,
    }


async def wait_for_duplex_warmup(websocket) -> None:
    """Wait at most ``DUPLEX_WARMUP_CLIENT_WAIT_S`` (180 s) for startup warmup.

    Single-frontend probes mark their connection with ``vllm_omni_warmup=1``.
    Multiple frontends use local engine probes, so no connection bypasses the gate.
    """
    warmup_done = getattr(websocket.app.state, "duplex_warmup_done", None)
    if warmup_done is None or warmup_done.is_set():
        return
    # Multiple frontends use local probes that never enter through a websocket,
    # so only single-frontend probes may bypass the wait.
    if (
        int(getattr(websocket.app.state, "api_server_count", 1) or 1) == 1
        and websocket.query_params.get("vllm_omni_warmup") == "1"
    ):
        return
    try:
        await asyncio.wait_for(warmup_done.wait(), timeout=DUPLEX_WARMUP_CLIENT_WAIT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning(
            "Duplex warmup still running after %d s; admitting the client anyway.",
            DUPLEX_WARMUP_CLIENT_WAIT_S,
        )


async def _warmup_local_duplex(engine_client: DuplexOmni, plugin: object | None, kind: str, warmup_frames: int) -> None:
    """Warm each owned group through this frontend's engine, without HTTP.

    Before client admission, sequential open/close advances the placement
    cursor over all owned groups without exposing a forced-group API.
    """
    engine = engine_client.engine
    if (
        engine_client.duplex_session_config.replica_policy != "least_sessions"
        or engine_client.duplex_capabilities.supports_session_resume is not False
    ):
        logger.warning(
            "Duplex local warmup skipped: multiple frontends require least_sessions and a non-resumable model."
        )
        return
    if engine.duplex_frontend_max_sessions == 0:
        return
    group_count = engine.duplex_frontend_group_count
    unit = warmup_silence_unit(plugin)
    capabilities = getattr(plugin, "capabilities", None)
    native_append = (
        bool(capabilities(max_sessions=1).supports_core_resumable_request) if callable(capabilities) else True
    )
    session: dict[str, object] = {
        "modalities": _warmup_modalities(engine_client),
        "input_audio_format": unit["format"],
        "output_audio_format": "pcm16",
        "sample_rate_hz": unit["sample_rate_hz"],
        "audio": {"input": {"sample_rate_hz": unit["sample_rate_hz"]}},
        "idle_timeout_s": 60,
        "turn_detection": None,
        "extra_body": {"auto_response": native_append},
    }
    audio = base64.b64decode(str(unit["audio"]))
    append_kwargs: _WarmupAudioAppendKwargs = {
        "format": str(unit["format"]),
        "sample_rate_hz": int(cast(int, unit["sample_rate_hz"])),
    }
    if kind == "video_turn":
        session.update(
            input_audio_format="pcm_f32le",
            sample_rate_hz=_WARMUP_SAMPLE_RATE_HZ,
            audio={"input": {"sample_rate_hz": _WARMUP_SAMPLE_RATE_HZ}},
            instructions=_WARMUP_INSTRUCTIONS,
            extra_body={"auto_response": True},
        )
        audio = base64.b64decode(_short_pcm_f32le_b64())
        append_kwargs = {
            "format": "pcm_f32le",
            "sample_rate_hz": _WARMUP_SAMPLE_RATE_HZ,
            "is_speech": True,
            "video_frames": [_warmup_jpeg_b64()],
        }
    text_only = "audio" not in _warmup_modalities(engine_client)
    if text_only:
        session["initial_user_text"] = "Reply briefly with ready."
        warmup_frames = max(warmup_frames, engine_client.duplex_capabilities.text_turn_priming_units, 1)
    for group_number in range(group_count):
        handle = await engine_client.open_session(session, timeout=30)
        saw_audio = asyncio.Event()
        saw_text = asyncio.Event()
        saw_listen = asyncio.Event()

        async def drain_events() -> None:
            async for event in handle.events():
                if event.type == "error":
                    raise RuntimeError(f"Duplex local warmup rejected: {event}")
                outcome = classify_warmup_outcome(
                    event.type, kind=kind, text_only=text_only, response=getattr(event, "response", None)
                )
                if outcome == WARMUP_OUTCOME_TEXT:
                    saw_text.set()
                elif outcome == WARMUP_OUTCOME_AUDIO:
                    saw_audio.set()
                elif outcome == WARMUP_OUTCOME_LISTEN:
                    saw_listen.set()

        pump = asyncio.create_task(drain_events())
        audio_ready = asyncio.create_task(saw_audio.wait())
        text_ready = asyncio.create_task(saw_text.wait())
        listen_ready = asyncio.create_task(saw_listen.wait())
        try:
            for _ in range(1 if kind == "video_turn" else warmup_frames):
                if pump.done():
                    pump.result()
                await asyncio.wait_for(handle.append_audio(audio, **append_kwargs), timeout=30)
                if kind != "video_turn":
                    await asyncio.sleep(0.08)
            if not native_append:
                await asyncio.wait_for(
                    handle.commit(final=True, create_response=True if kind == "video_turn" else None), timeout=30
                )
            done, _ = await asyncio.wait(
                (pump, audio_ready, text_ready, listen_ready),
                timeout=DUPLEX_WARMUP_AUDIO_WAIT_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pump in done:
                pump.result()
            if saw_audio.is_set():
                outcome = WARMUP_OUTCOME_AUDIO
            elif saw_text.is_set():
                outcome = WARMUP_OUTCOME_TEXT
            elif saw_listen.is_set():
                outcome = WARMUP_OUTCOME_LISTEN
            else:
                outcome = WARMUP_OUTCOME_TIMEOUT
            if outcome == WARMUP_OUTCOME_LISTEN:
                logger.info(
                    "Duplex local warmup group %d/%d: model chose to listen; no audio path to warm.",
                    group_number + 1,
                    group_count,
                )
            elif outcome == WARMUP_OUTCOME_TIMEOUT:
                logger.warning(
                    "Duplex local warmup group %d/%d: %s within %.0f s; nothing warmed.",
                    group_number + 1,
                    group_count,
                    _warmup_timeout_reason(kind, text_only),
                    DUPLEX_WARMUP_AUDIO_WAIT_S,
                )
            else:
                logger.info(
                    "Duplex local warmup group %d/%d: %s.",
                    group_number + 1,
                    group_count,
                    {
                        WARMUP_OUTCOME_AUDIO: "first audio received",
                        WARMUP_OUTCOME_TEXT: "text response completed",
                    }[outcome],
                )
        finally:
            try:
                await handle.close(reason="warmup_complete", timeout=10)
            finally:
                pump.cancel()
                audio_ready.cancel()
                text_ready.cancel()
                listen_ready.cancel()
                await asyncio.gather(pump, audio_ready, text_ready, listen_ready, return_exceptions=True)


async def _warmup_duplex_realtime(app, args, warmup_frames: int) -> None:
    """Run throwaway realtime sessions before real clients are admitted.

    Video-required models send one short non-silent audio chunk and one image
    so each stage compiles the shape a real turn uses. Audio-primary models
    keep the opt-in silent-frame path. ``/v1/realtime`` holds other connections
    until this finishes (see ``duplex_warmup_done``).
    Multiple frontends each probe their owned groups through the local engine;
    the single-frontend path retains its public websocket probe.
    """
    warmup_done = getattr(app.state, "duplex_warmup_done", None)
    try:
        count = int(getattr(app.state, "api_server_count", None) or getattr(args, "api_server_count", 1) or 1)
        if count > 1:
            engine_client = app.state.engine_client
            plugin = lookup_duplex_plugin(engine_client)
            kind = startup_warmup_kind(plugin, warmup_frames)
            if kind is not None:
                await _warmup_local_duplex(engine_client, plugin, kind, warmup_frames)
            return
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
        # One silence unit as the engine-side plugin defines it (DuplexOmniEngine.plugin):
        # the unit's format, rate and length are the model's, not a fixed 16 kHz.
        unit = warmup_silence_unit(plugin)
        frame_samples = len(base64.b64decode(str(unit["audio"]))) // 4
        capabilities = getattr(plugin, "capabilities", None)
        native_append = (
            bool(capabilities(max_sessions=1).supports_core_resumable_request) if callable(capabilities) else True
        )
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
            "modalities": _warmup_modalities(engine_client),
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "idle_timeout_s": 60,
            "turn_detection": None,
            "extra_body": {"auto_response": native_append},
        }
        if kind != "video_turn":
            session["input_audio_format"] = unit["format"]
            session["sample_rate_hz"] = unit["sample_rate_hz"]
            session["audio"] = {"input": {"sample_rate_hz": unit["sample_rate_hz"]}}
        if kind == "video_turn":
            # The default AURA prompt answers with <|silent|> unless the user
            # asked for something. This session is thrown away; force one short
            # spoken reply so Talker and Code2Wav compile too. auto_response
            # matches the web demo: a commit must actually start a turn.
            session["instructions"] = _WARMUP_INSTRUCTIONS
            session["extra_body"] = {"auto_response": True}
        text_only = "audio" not in _warmup_modalities(engine_client)
        if text_only:
            session["initial_user_text"] = "Reply briefly with ready."
            warmup_frames = max(warmup_frames, engine_client.duplex_capabilities.text_turn_priming_units, 1)
        async with websockets.connect(url, max_size=64 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": session}))
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
                while sent < warmup_frames:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": unit["audio"],
                                "format": unit["format"],
                                "sample_rate_hz": unit["sample_rate_hz"],
                            }
                        )
                    )
                    sent += 1
                    await asyncio.sleep(0.08)
            if not native_append:
                commit: dict[str, object] = {"type": "input_audio_buffer.commit", "final": True}
                if kind == "video_turn":
                    commit["create_response"] = True
                await ws.send(json.dumps(commit))

            # First output audio means Code2Wav ran, so the Talker compile
            # ahead of it has run too. Do not wait out the whole utterance.
            async def _recv_event() -> dict[str, object]:
                return json.loads(await ws.recv())

            outcome = await wait_for_warmup_outcome(
                _recv_event,
                DUPLEX_WARMUP_AUDIO_WAIT_S,
                kind=kind,
                text_only=text_only,
            )
            if outcome == WARMUP_OUTCOME_LISTEN:
                logger.info(
                    "Duplex warmup: model chose to listen to the warmup unit; "
                    "no audio path to warm (Talker/Code2Wav stay cold)."
                )
            elif outcome == WARMUP_OUTCOME_TIMEOUT:
                logger.warning(
                    "Duplex warmup: %s within %.0f s; nothing warmed.",
                    _warmup_timeout_reason(kind, text_only),
                    DUPLEX_WARMUP_AUDIO_WAIT_S,
                )
            # Close the session EXPLICITLY: a bare websocket close parks the
            # session in its disconnect grace window, where it would keep
            # occupying the max_sessions=1 slot and block the first client.
            await ws.send(json.dumps({"type": "session.close"}))
            await _recv_until(lambda e: e.get("type") == "session.closed", 10)
        if kind == "video_turn":
            logger.info(
                "Duplex warmup finished in %.1f s (one audio+frame turn, output %s).",
                time.time() - start,
                {
                    WARMUP_OUTCOME_AUDIO: "received",
                    WARMUP_OUTCOME_TEXT: "text response completed",
                    WARMUP_OUTCOME_LISTEN: "NOT received",
                    WARMUP_OUTCOME_TIMEOUT: "NOT received",
                }[outcome],
            )
        else:
            logger.info(
                "Duplex warmup finished in %.1f s (%d frames, %s).",
                time.time() - start,
                sent,
                {
                    WARMUP_OUTCOME_AUDIO: "first audio received",
                    WARMUP_OUTCOME_TEXT: "text response completed",
                    WARMUP_OUTCOME_LISTEN: "model listened; no audio warmed",
                    WARMUP_OUTCOME_TIMEOUT: (
                        f"{_warmup_timeout_reason(kind, text_only)} within the wait; nothing warmed"
                    ),
                }[outcome],
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
    "wait_for_duplex_warmup",
    "warmup_silence_unit",
]
