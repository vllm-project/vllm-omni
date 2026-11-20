# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Static host and same-origin proxy for the shared realtime browser UI."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol
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


class ToolExecutor(Protocol):
    """Runs the client side of function calling on the browser's behalf.

    An executor that sets ``uses_latest_frame`` also receives the camera frame
    bound to the response that made the call (frames from
    ``input_audio_buffer.append`` since the previous commit, copied when that
    response is created). It gets an empty string when that response had none.
    Executors that do not set it keep the
    two-argument ``run`` and never see the frame.
    """

    @property
    def realtime_tools(self) -> list[dict[str, Any]]: ...

    async def run(self, name: str, arguments: str, frame_b64: str = "") -> str: ...


def _json_event(text: str) -> dict[str, Any] | None:
    try:
        event = json.loads(text)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _session_update_with_tools(event: dict[str, Any], tools: ToolExecutor) -> str | None:
    """Return a ``session.update`` that gained tools, or None when it already has them."""
    if event.get("type") != "session.update" or not isinstance(event.get("session"), dict):
        return None
    if "tools" in event["session"]:
        return None
    session = {**event["session"], "tools": tools.realtime_tools}
    return json.dumps({**event, "session": session})


def _with_session_tools(text: str, tools: ToolExecutor) -> str:
    """Add the executor's tools to a ``session.update`` that has none."""
    event = _json_event(text)
    if event is None:
        return text
    updated = _session_update_with_tools(event, tools)
    return text if updated is None else updated


def _note_client_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    """Reset a follow-up on a user commit, and keep the newest camera frame.

    Frames seen since the previous commit belong to the turn this commit closes:
    they move to ``pending_frame`` and ``last_frame`` starts empty, so a later
    audio-only turn never reuses a frame from before the camera was turned off.
    A commit the backend does not admit must not replace the frame of a
    response that is already running, so the pending frame is copied onto a
    response only when that response is created.
    """
    if event.get("type") == "input_audio_buffer.commit":
        state["pending_frame"] = str(state.get("last_frame") or "")
        state["turn_frame"] = state["pending_frame"]
        state["last_frame"] = ""
        state["gen"] += 1
        state["count"] = 0
        state["awaiting_done"] = False
        state["ready"] = False
        state["final_answer"] = False
        return
    if event.get("type") != "input_audio_buffer.append":
        return
    frames = event.get("video_frames")
    if not isinstance(frames, list):
        return
    for frame in reversed(frames):
        if isinstance(frame, str) and frame:
            state["last_frame"] = frame
            return


def _begin_tool_call(state: dict[str, Any]) -> None:
    """Count one in-flight call for this user turn. A new turn starts from zero."""
    if state.get("inflight_gen") != state["gen"]:
        state["inflight_gen"] = state["gen"]
        state["inflight"] = 0
    state["inflight"] = int(state.get("inflight") or 0) + 1


def _executor_uses_latest_frame(tools: ToolExecutor) -> bool:
    return getattr(tools, "uses_latest_frame", False) is True


def _finished_function_call(event: dict[str, Any]) -> dict[str, Any] | None:
    """The call in a ``response.output_item.done``; the arguments event carries no name."""
    item = event.get("item")
    if event.get("type") != "response.output_item.done" or not isinstance(item, dict):
        return None
    if item.get("type") != "function_call" or not isinstance(item.get("call_id"), str) or not item["call_id"]:
        return None
    return item


# A failed search result makes the model call again. Stop opening follow-ups
# after this many, or one question can spin for the life of the session.
# Matches the native gateway MAX_TOOL_ROUNDS=5; one extra create after the
# cap lets AURA inject its silent+nudge answer turn.
_MAX_TOOL_FOLLOWUPS = 5


def _is_user_commit(text: str) -> bool:
    event = _json_event(text)
    return event is not None and event.get("type") == "input_audio_buffer.commit"


def _event_response_id(event: dict[str, Any]) -> str | None:
    response_id = event.get("response_id")
    return response_id if isinstance(response_id, str) and response_id else None


def _bind_admitted_frame(state: dict[str, Any], response_id: str | None) -> None:
    """Copy the frame of the latest commit onto a response the backend admitted.

    Client commits rotate ``pending_frame`` before the backend decides whether
    to submit them. A video-only commit that stays unsubmitted therefore leaves
    the previous response's frame alone.
    """
    if response_id is None:
        return
    frames = state.setdefault("frames_by_response", {})
    if response_id in frames:
        return
    frames[response_id] = str(state.get("pending_frame") or "")


def _frame_for_response(state: dict[str, Any], response_id: str | None) -> str:
    """The frame bound when this response was admitted.

    A response that has not been created yet still reads the latest commit, so
    a caller that never sees ``response.created`` keeps the previous behavior.
    """
    frames = state.get("frames_by_response")
    if response_id is not None and isinstance(frames, dict) and response_id in frames:
        return str(frames[response_id] or "")
    return str(state.get("turn_frame") or "")


def _hold_follow_up_for_response_done(state: dict[str, Any], response_id: str | None = None) -> None:
    """The function-call item is published before ``response.done``.

    A follow-up ``response.create`` in that gap is rejected with
    ``response_already_active``. ``response_id`` names the response that owns the
    call: with overlapping turns another response can finish first, and only the
    owner's ``response.done`` frees the follow-up.
    """
    state["awaiting_done"] = True
    state["pending_gen"] = state["gen"]
    state["pending_response_id"] = response_id


def _calls_still_running(state: dict[str, Any]) -> bool:
    return state.get("inflight_gen") == state["gen"] and int(state.get("inflight") or 0) > 0


def _follow_up_ready(state: dict[str, Any], generation: int) -> bool:
    """True when this finished call may start the one follow-up for the turn.

    False while ``response.done`` has not arrived, or while another call from
    the same turn is still running. The done handler starts it in the first
    case; the last call starts it in the second.
    """
    if state.get("inflight_gen") == generation:
        state["inflight"] = max(0, int(state.get("inflight") or 0) - 1)
    if state["gen"] != generation:
        return False
    if state["awaiting_done"] or _calls_still_running(state):
        state["ready"] = True
        return False
    return True


def _take_deferred_follow_up(state: dict[str, Any], response_id: str | None = None) -> bool:
    """Call on ``response.done``. True when every tool result for this turn is in.

    A ``response.done`` for a different response than the one that owns the
    call does not release the wait. When either id is unknown the wait is
    released as before.
    """
    owner = state.get("pending_response_id")
    if owner is not None and response_id is not None and owner != response_id:
        return False
    state["awaiting_done"] = False
    if state["ready"] and state["pending_gen"] == state["gen"] and not _calls_still_running(state):
        state["ready"] = False
        return True
    return False


async def _answer_function_call(
    backend,
    tools: ToolExecutor,
    call: dict[str, Any],
    *,
    follow_up: bool = True,
    follow_state: dict[str, Any] | None = None,
    frame_b64: str = "",
) -> None:
    generation = follow_state["gen"] if follow_state is not None else None
    name = str(call.get("name") or "")
    arguments = str(call.get("arguments") or "")
    if _executor_uses_latest_frame(tools):
        output = await tools.run(name, arguments, frame_b64=frame_b64)
    else:
        output = await tools.run(name, arguments)
    item = {"type": "function_call_output", "call_id": call["call_id"], "output": output}
    await backend.send(json.dumps({"type": "conversation.item.create", "item": item}))
    if not follow_up:
        return
    if follow_state is not None and not _follow_up_ready(follow_state, generation):
        return
    await backend.send(json.dumps({"type": "response.create"}))


async def _pump_client_to_backend(
    client: WebSocket,
    backend,
    tools: ToolExecutor | None = None,
    tool_followups: dict[str, Any] | None = None,
) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                await backend.close()
                return
            if message.get("text") is not None:
                text = message["text"]
                event = _json_event(text)
                if event is not None and tool_followups is not None:
                    _note_client_event(tool_followups, event)
                if event is not None and tools is not None:
                    updated = _session_update_with_tools(event, tools)
                    if updated is not None:
                        text = updated
                await backend.send(text)
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def _pump_backend_to_client(
    client: WebSocket,
    backend,
    tools: ToolExecutor | None = None,
    tool_followups: dict[str, Any] | None = None,
) -> None:
    calls: set[asyncio.Task] = set()
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
                continue
            await client.send_text(message)
            if tools is None:
                continue
            event = _json_event(message)
            if event is None:
                continue
            kind = event.get("type")
            if tool_followups is not None and kind == "response.created":
                _bind_admitted_frame(tool_followups, _event_response_id(event))
            if (
                tool_followups is not None
                and kind == "response.done"
                and _take_deferred_follow_up(tool_followups, _event_response_id(event))
            ):
                await backend.send(json.dumps({"type": "response.create"}))
            call = _finished_function_call(event) if event is not None else None
            if call is not None:
                count = tool_followups["count"] if tool_followups is not None else 0
                follow_up = count < _MAX_TOOL_FOLLOWUPS
                final_answer = tool_followups is not None and not follow_up and not tool_followups.get("final_answer")
                if tool_followups is not None and (follow_up or final_answer):
                    if follow_up:
                        tool_followups["count"] = count + 1
                    else:
                        tool_followups["final_answer"] = True
                    _hold_follow_up_for_response_done(tool_followups, _event_response_id(event))
                    _begin_tool_call(tool_followups)
                task = asyncio.create_task(
                    _answer_function_call(
                        backend,
                        tools,
                        call,
                        follow_up=follow_up or final_answer,
                        follow_state=tool_followups,
                        # The frame bound to this response. A later commit that the
                        # backend has not admitted must not replace it.
                        frame_b64=(
                            _frame_for_response(tool_followups, _event_response_id(event))
                            if tool_followups is not None
                            else ""
                        ),
                    )
                )
                calls.add(task)
                task.add_done_callback(calls.discard)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise
    finally:
        for task in calls:
            task.cancel()
        if calls:
            await asyncio.gather(*calls, return_exceptions=True)


def _expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


def build_app(
    *,
    ws_backend: str | None = None,
    model: str | None = None,
    public_realtime_url: str | None = None,
    ref_audio: str | None = None,
    profile: str = "minicpm-native",
    turn_mode: str = "manual",
    speech_input: bool = False,
    tool_executor: ToolExecutor | None = None,
) -> FastAPI:
    if profile not in {"minicpm-native", "qwen3-turn", "aura-ptt", "joyai-turn"}:
        raise ValueError(f"Unknown profile: {profile}")
    if turn_mode not in {"manual", "vad", "semantic_vad"}:
        raise ValueError(f"Unknown turn mode: {turn_mode}")
    if turn_mode == "semantic_vad" and profile != "qwen3-turn":
        raise ValueError("--semantic-vad requires --profile qwen3-turn")
    if profile == "qwen3-turn" and ref_audio:
        raise ValueError("Qwen3 profile does not accept --ref-audio")
    if profile == "aura-ptt" and ref_audio:
        raise ValueError("AURA profile does not accept --ref-audio")
    if profile == "joyai-turn" and ref_audio:
        raise ValueError("JoyAI profile does not accept --ref-audio")
    ws_backend = ws_backend or ("ws://127.0.0.1:8091" if profile == "qwen3-turn" else "ws://127.0.0.1:8099")
    model = (
        model
        or {
            "minicpm-native": "openbmb/MiniCPM-o-4_5",
            "aura-ptt": "aurateam/AURA",
            "joyai-turn": "jdopensource/JoyAI-VL-Interaction-Preview",
            "qwen3-turn": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        }[profile]
    )
    app = FastAPI(title="Omni Realtime Web UI")
    index_path = APP_DIR / "index.html"
    app_version_hash = hashlib.sha256()
    for asset_path in (
        STATIC_DIR / "app.js",
        STATIC_DIR / "pcm_worklet.js",
        STATIC_DIR / "playback_worklet.js",
        *sorted((STATIC_DIR / "profiles").glob("*.js")),
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
                "profile": profile,
                "turnMode": turn_mode,
                "speechInput": speech_input,
                "realtimePath": public_realtime_url or "v1/realtime",
                "refAudio": ref_audio_uri,
                "appVersion": app_version,
            },
            ensure_ascii=True,
        ).replace("<", "\\u003c")
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
                tool_followups = {
                    "count": 0,
                    "gen": 0,
                    "pending_gen": 0,
                    "awaiting_done": False,
                    "ready": False,
                    "final_answer": False,
                    "inflight": 0,
                    "inflight_gen": None,
                    "last_frame": "",
                    "turn_frame": "",
                    "pending_frame": "",
                    "frames_by_response": {},
                    "pending_response_id": None,
                }
                tasks = {
                    asyncio.create_task(_pump_client_to_backend(websocket, backend, tool_executor, tool_followups)),
                    asyncio.create_task(_pump_backend_to_client(websocket, backend, tool_executor, tool_followups)),
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
        except (OSError, TimeoutError) as exc:
            message = (
                f"Cannot connect to realtime backend {ws_backend}. "
                "Check that vllm serve is running and --backend matches its --port. "
                "For containers or remote hosts, use the backend address reachable from the UI server."
            )
            logger.warning("%s (%s)", message, exc)
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "code": "backend_unavailable", "error": message})
                await websocket.close(code=1011)
        except Exception:
            logger.exception("Realtime WebSocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(default_profile: str = "minicpm-native", tool_executor: ToolExecutor | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--profile", choices=("minicpm-native", "qwen3-turn", "aura-ptt", "joyai-turn"), default=default_profile
    )
    parser.add_argument("--ws-backend", "--backend", dest="ws_backend")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", dest="turn_mode", action="store_const", const="manual")
    mode.add_argument("--vad", dest="turn_mode", action="store_const", const="vad")
    mode.add_argument(
        "--semantic-vad",
        dest="turn_mode",
        action="store_const",
        const="semantic_vad",
        help="Qwen3 semantic turn detection (requires the Smart Turn artifact).",
    )
    parser.set_defaults(turn_mode="manual")
    parser.add_argument("--speech-input", action="store_true", help="Enable JoyAI microphone input (ASR deployment).")
    parser.add_argument(
        "--public-realtime-url",
        help="Browser-visible ws:// or wss:// Realtime URL; defaults to the same-origin proxy.",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--ref-audio",
        help=(
            "Optional MiniCPM assistant voice override; when omitted, the server "
            "uses the model-bundled assets/HT_ref_audio.wav"
        ),
    )
    args = parser.parse_args()
    if args.profile != "minicpm-native" and args.ref_audio:
        parser.error("--ref-audio is only supported by minicpm-native")

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        build_app(
            ws_backend=args.ws_backend,
            model=args.model,
            public_realtime_url=args.public_realtime_url,
            ref_audio=args.ref_audio,
            profile=args.profile,
            turn_mode=args.turn_mode,
            speech_input=args.speech_input,
            tool_executor=tool_executor,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
