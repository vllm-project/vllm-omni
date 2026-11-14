# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E online tests for Qwen3-Omni /v1/realtime WebSocket (streaming PCM in, audio out).

Four scenarios:
- Ready CI: async_chunk on, smoke only (no send delay, no accuracy check).
- Merge CI: async_chunk on + send delay, full accuracy check.
- Merge CI: async_chunk off, no send delay, full accuracy check.
- Server VAD: two turns without client commits.

Turn tests that need async_chunk share one class-scoped server. pytest 9.1.1
(the CI pin) finalizes that fixture when the next test's param value differs
and does not revive it, so those tests stay contiguous and the async_chunk-off
server runs after them. The cold-start JIT replay runs first on that shared async-chunk server.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import websockets
from openai.types.realtime import (
    ConversationItemCreateEvent,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
    RealtimeFunctionTool,
    RealtimeSessionCreateRequest,
    ResponseCreateEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_client_event import RealtimeClientEvent
from openai.types.realtime.realtime_conversation_item_user_message import Content

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import validated_input_wav
from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    convert_audio_bytes_to_text,
    cosine_similarity_text,
    generate_synthetic_audio,
)
from tests.helpers.realtime import (
    WEATHER_TOOL_ANSWER,
    WEATHER_TOOL_OUTPUT,
    WEATHER_TOOL_PROMPT,
    WEATHER_TOOL_SPEC,
)
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.engine.duplex.smart_turn import (
    SMART_TURN_FILENAME,
    SMART_TURN_REPO_ID,
    SMART_TURN_REVISION,
)
from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_FILENAME,
    SILERO_VAD_REPO_ID,
    SILERO_VAD_REVISION,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("VLLM_OMNI_TEST_QWEN3_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SERVER_VAD_MODEL_PATH = os.environ.get("VLLM_OMNI_TEST_SILERO_VAD_MODEL_PATH")
SEMANTIC_VAD_MODEL_PATH = os.environ.get("VLLM_OMNI_TEST_SMART_TURN_MODEL_PATH")

# Synthetic input for realtime E2E (``generate_synthetic_audio``); distinct cache file per phrase.
REALTIME_SYNTH_PHRASE_TEXT = (
    "Translate into Chinese: Beijing is the Capital of China. It is the center of culture and politics"
)
ISSUE_6474_SYNTH_PHRASE_TEXT = (
    "Can you tell me the current temperature and weather conditions in New York City? "
    "What about Los Angeles? Please compare them in detail using at least eight complete sentences."
)
# Simulate realtime upload pacing (``openai_realtime_client.py --send-delay-ms``).
SEND_DELAY_MS = 200
CLIENT_VAD_REPLAY_PATH = Path(__file__).resolve().parents[2] / "assets" / "livekit" / "client_vad_replay.jsonl"

# CI overlay bakes in async_chunk: False and covers CUDA/ROCm/XPU via ``platforms:``.
# Its seq and sampling caps stay: production max_tokens would change how long
# these realtime replies run, not only how the server is launched.
default_stage_config = get_deploy_config_path("ci/qwen3_omni_moe.yaml")
# One parsed deploy for every turn test that needs async_chunk, including
# server VAD. The flag lives in the YAML (the CLI override used to disagree
# with the overlay's ``async_chunk: false``). A pinned Silero path is applied
# only when the env var is set; otherwise VAD resolves the same way as before.
_async_chunk_updates: dict[str, bool | dict[str, str]] = {"async_chunk": True}
if SERVER_VAD_MODEL_PATH is not None:
    _async_chunk_updates["duplex_session"] = {"server_vad_model_path": SERVER_VAD_MODEL_PATH}
async_chunk_stage_config = modify_stage_config(default_stage_config, _async_chunk_updates)

realtime_sync_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=default_stage_config,
            use_stage_cli=True,
        ),
        id="sync",
    ),
]

# Same server object under two ids. pytest 9.1.1 reuses a class-scoped fixture
# for equal param values, including across ids, but only while those tests stay
# adjacent. ``server_vad`` keeps its node id; it does not boot a second model.
# ``async_chunk`` stays in the parsed deploy. The tool-call flags are the CLI
# surface added for the audio tool-call case; they ride this shared server.
_async_chunk_server = OmniServerParams(
    model=MODEL,
    stage_config_path=async_chunk_stage_config,
    use_stage_cli=True,
    server_args=[
        "--tool-call-parser",
        "hermes",
        "--enable-auto-tool-choice",
    ],
)
realtime_async_chunk_server_params = [
    pytest.param(_async_chunk_server, id="async_chunk"),
]
realtime_server_vad_server_params = [
    pytest.param(_async_chunk_server, id="server_vad"),
]

# Stage-log lines the cold-start replay test checks.
REALTIME_CONNECTION_LOG_MARKER = "[realtime] connection opened"
# vLLM's JIT monitor logs a kernel compiled after startup warmup with this
# text, once per kernel and process (``warning_once``).
RUNTIME_JIT_LOG_MARKER = "JIT compilation during inference"
MOE_PREFILL_WARMUP_LOG_MARKER = "MoE prefill warmup ran"


def _pcm16_mono_from_wav_bytes(wav_bytes: bytes, sample_rate_hz: int = 16000) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, sampwidth={wf.getsampwidth()}")
        if wf.getframerate() != sample_rate_hz:
            raise ValueError(f"Expected {sample_rate_hz} Hz input for /v1/realtime, got {wf.getframerate()} Hz")
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed PCM, comptype={wf.getcomptype()!r}")
        return wf.readframes(wf.getnframes())


def _wav_bytes_from_pcm16(pcm: bytes, sample_rate_hz: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm)
    return buf.getvalue()


async def _run_realtime_audio_roundtrip(
    host: str,
    port: int,
    model: str,
    pcm16: bytes,
    *,
    chunk_ms: int = 100,
    send_delay_ms: int = 0,
    completion_timeout_s: float = 600,
) -> dict:
    uri = f"ws://{host}:{port}/v1/realtime"
    incremental: list[bytes] = []
    output_sr = 24000
    text_chunks: list[str] = []
    final_text = ""
    delta_events = 0
    audio_done = False

    bytes_per_ms = 24000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "model": model}}))

        for i in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[i : i + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )
            if send_delay_ms > 0:
                await asyncio.sleep(send_delay_ms / 1000.0)

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=completion_timeout_s)
            if isinstance(message, bytes):
                continue

            event = json.loads(message)
            event_type = event.get("type")

            if event_type in {
                "session.created",
                "session.updated",
                "input_audio_buffer.committed",
                "conversation.item.added",
                "conversation.item.done",
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.content_part.done",
                "response.output_item.done",
            }:
                continue

            if event_type == "response.output_audio.delta":
                delta_events += 1
                sr = event.get("sample_rate_hz")
                if isinstance(sr, int) and sr > 0:
                    output_sr = sr
                audio_b64 = event.get("delta", "")
                if audio_b64:
                    incremental.append(base64.b64decode(audio_b64))
                continue

            if event_type == "response.output_audio_transcript.delta":
                d = event.get("delta", "")
                if d:
                    text_chunks.append(d)
                continue

            if event_type == "response.output_audio_transcript.done":
                final_text = event.get("transcript") or "".join(text_chunks)
                continue

            if event_type == "response.output_audio.done":
                audio_done = True
                continue

            if event_type == "response.done":
                break

            if event_type == "error":
                raise AssertionError(f"WebSocket error: {event}")

            raise AssertionError(f"Unexpected WebSocket event: {event}")

    out_pcm = b"".join(incremental)
    return {
        "output_pcm": out_pcm,
        "output_sample_rate": output_sr,
        "transcription_text": final_text or "".join(text_chunks),
        "delta_events": delta_events,
        "audio_done": audio_done,
    }


async def _send_client_event(ws: Any, event: RealtimeClientEvent) -> None:
    """Serialize a client event the same way the OpenAI SDK does."""
    await ws.send(event.to_json(exclude_defaults=True, exclude_unset=True))


async def _run_realtime_audio_tool_call(
    host: str,
    port: int,
    model: str,
    *,
    timeout_s: float = 600,
) -> dict:
    """Ask for an audio response that must call a tool, then feed the result back."""

    def summarize(events: list[dict]) -> dict:
        transcript = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") in ("response.output_audio_transcript.delta", "response.output_text.delta")
        )
        tool_calls = [
            event["item"]
            for event in events
            if event.get("type") == "response.output_item.done" and event.get("item", {}).get("type") == "function_call"
        ]
        first_control = next(
            (
                index
                for index, event in enumerate(events)
                if event.get("type") == "response.output_item.added"
                and event.get("item", {}).get("type") == "function_call"
            ),
            len(events),
        )
        return {
            "transcript": transcript,
            "tool_calls": tool_calls,
            "audio_deltas": sum(1 for event in events if event.get("type") == "response.output_audio.delta"),
            "audio_after_tool_call": sum(
                1 for event in events[first_control:] if event.get("type") == "response.output_audio.delta"
            ),
        }

    async def recv_response(ws) -> list[dict]:
        events: list[dict] = []
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            events.append(event)
            if event.get("type") == "error":
                raise AssertionError(f"WebSocket error: {event}")
            if event.get("type") == "response.done":
                return events

    async with websockets.connect(f"ws://{host}:{port}/v1/realtime", max_size=64 * 1024 * 1024) as ws:
        await _send_client_event(
            ws,
            SessionUpdateEvent(
                type="session.update",
                session=RealtimeSessionCreateRequest(
                    type="realtime",
                    model=model,
                    output_modalities=["audio"],
                    tools=[RealtimeFunctionTool(**WEATHER_TOOL_SPEC)],
                    tool_choice="auto",
                ),
            ),
        )
        await _send_client_event(
            ws,
            ConversationItemCreateEvent(
                type="conversation.item.create",
                item=RealtimeConversationItemUserMessage(
                    type="message",
                    role="user",
                    content=[Content(type="input_text", text=WEATHER_TOOL_PROMPT)],
                ),
            ),
        )
        await _send_client_event(ws, ResponseCreateEvent(type="response.create"))
        tool_turn = summarize(await recv_response(ws))

        answer_turn = None
        if tool_turn["tool_calls"]:
            call = tool_turn["tool_calls"][0]
            await _send_client_event(
                ws,
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item=RealtimeConversationItemFunctionCallOutput(
                        type="function_call_output",
                        call_id=call["call_id"],
                        output=WEATHER_TOOL_OUTPUT,
                    ),
                ),
            )
            await _send_client_event(ws, ResponseCreateEvent(type="response.create"))
            answer_turn = summarize(await recv_response(ws))

    return {"tool_turn": tool_turn, "answer_turn": answer_turn}


async def _append_pcm16_chunks(ws, pcm16: bytes, chunk_bytes: int) -> None:
    for offset in range(0, len(pcm16), chunk_bytes):
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16[offset : offset + chunk_bytes]).decode("utf-8"),
                }
            )
        )


async def _receive_server_vad_turn(ws) -> list[dict]:
    events: list[dict] = []
    while True:
        message = await asyncio.wait_for(ws.recv(), timeout=600)
        if isinstance(message, bytes):
            continue
        event = json.loads(message)
        events.append(event)
        if event.get("type") == "error":
            raise AssertionError(f"WebSocket error: {event}")
        if event.get("type") == "response.done":
            return events


async def _run_server_vad_audio_roundtrips(
    host: str,
    port: int,
    model: str,
    pcm16_24k: bytes,
    *,
    chunk_ms: int = 100,
    turns: int = 2,
    turn_detection: dict | None = None,
    silence_ms: int = 1500,
    duplex: bool = False,
) -> list[list[dict]]:
    chunk_bytes = max(24_000 * 2 // 1000 * chunk_ms, 2)
    turn_events: list[list[dict]] = []

    query = "?duplex=1" if duplex else ""
    async with websockets.connect(f"ws://{host}:{port}/v1/realtime{query}", max_size=64 * 1024 * 1024) as ws:
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "model": model,
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24_000},
                                    "turn_detection": turn_detection or {"type": "server_vad"},
                                }
                            },
                        },
                    }
                )
            )

            silence = bytes(24_000 * 2 * silence_ms // 1000)
            for _ in range(turns):
                await _append_pcm16_chunks(ws, pcm16_24k, chunk_bytes)
                await _append_pcm16_chunks(ws, silence, chunk_bytes)
                turn_events.append(await _receive_server_vad_turn(ws))
        finally:
            if duplex:
                # Disconnect leaves a resumable duplex session holding the
                # admission slot. Wait for its release before the next test.
                await ws.send(json.dumps({"type": "session.close"}))
                closing_events: list[dict] = []
                await _duplex_receive_until(
                    ws,
                    closing_events,
                    lambda: any(e["type"] == "session.closed" for e in closing_events),
                    timeout_s=15,
                )

    return turn_events


def _assert_auto_committed_turns(turns: list[list[dict]]) -> None:
    """Each turn: one speech start/stop, one commit and one terminal audio response, in order."""
    required_sequence = [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.done",
    ]
    input_item_ids: list[str] = []
    response_ids: list[str] = []
    for events in turns:
        event_types = [event["type"] for event in events]
        for event_type in required_sequence:
            if event_type != "response.output_audio.delta":
                assert event_types.count(event_type) == 1, event_types
        positions = [event_types.index(event_type) for event_type in required_sequence]
        assert positions == sorted(positions)

        started = next(event for event in events if event["type"] == "input_audio_buffer.speech_started")
        stopped = next(event for event in events if event["type"] == "input_audio_buffer.speech_stopped")
        committed = next(event for event in events if event["type"] == "input_audio_buffer.committed")
        created = next(event for event in events if event["type"] == "response.created")["response"]
        done = next(event for event in events if event["type"] == "response.done")["response"]

        assert started["item_id"] == stopped["item_id"] == committed["item_id"]
        input_item_id = committed["item_id"]
        history_events = [
            event
            for event in events
            if event["type"] in {"conversation.item.added", "conversation.item.done"}
            and event["item"]["id"] == input_item_id
        ]
        assert [event["type"] for event in history_events] == [
            "conversation.item.added",
            "conversation.item.done",
        ]
        assert all(event["item"]["role"] == "user" for event in history_events)
        output_pcm = b"".join(
            base64.b64decode(event["delta"])
            for event in events
            if event["type"] == "response.output_audio.delta" and event.get("delta")
        )
        assert output_pcm
        assert created["id"] == done["id"]
        # CI caps stage-0 text generation at 150 tokens, so a valid VAD turn can
        # finish with an incomplete response and reason=max_output_tokens.
        assert done["status"] in {"completed", "incomplete"}, done
        if done["status"] == "incomplete":
            assert done["status_details"]["reason"] == "max_output_tokens", done
        input_item_ids.append(input_item_id)
        response_ids.append(created["id"])

    assert len(set(input_item_ids)) == 2
    assert len(set(response_ids)) == 2


def _output_text(response: dict) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(text)
    return "".join(parts)


async def _run_client_vad_replay(
    host: str,
    port: int,
    model: str,
    replay_path: Path,
    *,
    wait_s: float = 10.0,
    answer_timeout_s: float = 60.0,
    stall_s: list[float] | None = None,
) -> list[str]:
    """Replay a captured LiveKit client-VAD session at its original speed.

    The capture ends with the recorder's own teardown: two ``response.cancel``
    events sent 5.93 s after its last ``response.create``. Replayed at their
    captured offsets they are not part of the conversation but a wall-clock
    deadline for the last turn -- a reply that is merely slower than the
    recording is cancelled before its first transcript delta and comes back as
    an empty answer (issue #9069). They are held back and replayed once the
    answers are collected, so the checks below compare model output rather than
    the recording's teardown. The cancels that precede a ``response.create`` are
    part of the interaction and still go out on time.

    When ``stall_s`` is given, it receives one entry per response: seconds from
    ``response.created`` to its first transcript delta (or to ``response.done``
    if no delta arrived).
    """
    with replay_path.open() as replay_file:
        records = [json.loads(line) for line in replay_file]
    assert records

    answers_by_response_id: dict[str, str] = {}
    response_order: list[str] = []
    completed_answers: dict[str, str] = {}
    pending_responses: dict[str, asyncio.Future] = {}
    errors: list[dict] = []
    created_at: dict[str, float] = {}
    first_output_at: dict[str, float] = {}

    last_create = max(
        (index for index, record in enumerate(records) if record["data"].get("type") == "response.create"),
        default=-1,
    )
    trailing_cancels = [
        record["data"]
        for index, record in enumerate(records)
        if index > last_create and record["data"].get("type") == "response.cancel"
    ]

    async def _await_last_turn() -> None:
        """Wait for the last started response to finish, without cancelling it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + answer_timeout_s
        while not response_order and loop.time() < deadline:
            await asyncio.sleep(0.01)
        if not response_order:
            return
        pending = pending_responses[response_order[-1]]
        with suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=max(deadline - loop.time(), 0.0))

    async with websockets.connect(
        f"ws://{host}:{port}/v1/realtime?model={model}",
        max_size=64 * 1024 * 1024,
    ) as ws:

        async def receive_responses() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "error":
                    errors.append(event)
                elif event_type == "response.created":
                    response_id = event["response"]["id"]
                    response_order.append(response_id)
                    answers_by_response_id[response_id] = ""
                    pending_responses[response_id] = asyncio.get_running_loop().create_future()
                    created_at[response_id] = asyncio.get_running_loop().time()
                elif event_type in {
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                    "response.output_text.delta",
                    "transcription.delta",
                }:
                    response_id = event.get("response_id")
                    if response_id in answers_by_response_id:
                        answers_by_response_id[response_id] += event.get("delta", "")
                        if event.get("delta"):
                            first_output_at.setdefault(response_id, asyncio.get_running_loop().time())
                elif event_type == "response.done":
                    response = event["response"]
                    response_id = response["id"]
                    text = answers_by_response_id.get(response_id, "")
                    completed_answers[response_id] = text or _output_text(response)
                    first_output_at.setdefault(response_id, asyncio.get_running_loop().time())
                    future = pending_responses.get(response_id)
                    if future is not None and not future.done():
                        future.set_result(None)

        receiver = asyncio.create_task(receive_responses())
        capture_start = records[0]["ts"]
        replay_start = asyncio.get_running_loop().time()

        try:
            for index, record in enumerate(records):
                target = replay_start + (record["ts"] - capture_start)
                delay = target - asyncio.get_running_loop().time()
                if delay > 0:
                    await asyncio.sleep(delay)

                message = record["data"]
                # The capture's reference-voice update is only valid when the
                # replay client uploads its optional reference-audio asset.
                if "__VOICE__" in json.dumps(message):
                    continue
                if index > last_create and message.get("type") == "response.cancel":
                    continue
                await ws.send(json.dumps(message))

            await _await_last_turn()
            # Replay the recorded teardown now that every answer is in.
            for message in trailing_cancels:
                await ws.send(json.dumps(message))
            await asyncio.sleep(wait_s)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    assert not [error for error in errors if error.get("error", {}).get("type") == "server_error"]
    if stall_s is not None:
        stall_s.extend(
            first_output_at[response_id] - created_at[response_id]
            for response_id in response_order
            if response_id in first_output_at
        )
    return [completed_answers[response_id] for response_id in response_order if response_id in completed_answers]


@pytest.fixture(scope="class")
def cached_smart_turn_artifact() -> str:
    """Prepare the pinned Smart Turn artifact before the serving subprocess starts."""
    if SEMANTIC_VAD_MODEL_PATH is not None:
        return SEMANTIC_VAD_MODEL_PATH

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=SMART_TURN_REPO_ID, filename=SMART_TURN_FILENAME, revision=SMART_TURN_REVISION)


@pytest.fixture(scope="class")
def cached_silero_vad_artifact() -> str:
    """Prepare the pinned artifact before the serving subprocess starts."""
    if SERVER_VAD_MODEL_PATH is not None:
        return SERVER_VAD_MODEL_PATH

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=SILERO_VAD_REPO_ID,
        filename=SILERO_VAD_FILENAME,
        revision=SILERO_VAD_REVISION,
    )


def _synthetic_pcm16_input(
    *,
    phrase_text: str = REALTIME_SYNTH_PHRASE_TEXT,
    duration_s: int = 10,
) -> bytes:
    syn = generate_synthetic_audio(
        duration_s,
        1,
        sample_rate=24000,
        phrase_text=phrase_text,
    )
    wav_bytes = base64.b64decode(syn["base64"])
    return _pcm16_mono_from_wav_bytes(wav_bytes, sample_rate_hz=24000)


def _assert_realtime_smoke(result: dict) -> None:
    out_pcm = result["output_pcm"]
    assert result["audio_done"], "No response.output_audio.done event"
    assert result["delta_events"] >= 1
    assert out_pcm, "No output PCM from response.output_audio.delta"
    assert len(out_pcm) % 2 == 0
    assert len(out_pcm) >= 4096, "Output audio unexpectedly small"
    assert result["output_sample_rate"] > 0


def _assert_realtime_accuracy(
    result: dict,
    whisper_model_size: str = "large-v3",
    threshold: float = 0.8,
) -> None:
    """Assert that whisper transcription of audio output matches model text.

    Args:
        result: Roundtrip result dict from ``_run_realtime_audio_roundtrip``.
        whisper_model_size: Whisper model used to transcribe the generated audio
                   for the accuracy check. Defaults to ``large-v3``: the default
                   ``small`` model mishears short Chinese TTS clips (observed:
                   北京→韦京 and a dropped leading sentence, sim=0.443), which
                   caused spurious sim<0.8 failures under async_chunk codec
                   variability even though audio generation was correct. large-v3
                   transcribes these clips reliably, so a failure here now points
                   at the model, not the ASR grader.
        threshold: Minimum cosine similarity (with length penalty) required to
                   pass. Default 0.8. Do not lower per-callsite without data:
                   at 0.35 the assertion no longer detects real audio
                   regressions. If a variant genuinely needs a different gate
                   (e.g. whisper partial transcripts under async_chunk), propose
                   it in its own PR with measurements.
    """
    final_text = (result["transcription_text"] or "").strip()
    assert final_text, "Expected non-empty transcription (model text stream)"

    wav_out = _wav_bytes_from_pcm16(result["output_pcm"], result["output_sample_rate"])
    whisper_text = convert_audio_bytes_to_text(wav_out, model_size=whisper_model_size).strip()
    assert whisper_text, "Whisper returned empty string for synthesized output audio"

    sim = cosine_similarity_text(whisper_text.lower(), final_text.lower())
    assert sim > threshold, (
        f"Output audio transcript should match model text (sim={sim:.3f}, "
        f"threshold={threshold}): "
        f"whisper={whisper_text!r}, model_text={final_text!r}"
    )


async def _conformance_recv_until(ws, predicate, collected=None, *, timeout_s=120):
    events = collected if collected is not None else []

    async def _loop():
        while not predicate(events):
            raw = await ws.recv()
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            events.append(event)
            if event.get("type") == "error":
                err = event.get("error", event)
                if err.get("type") == "server_error":
                    raise AssertionError(f"Server error: {event}")

    await asyncio.wait_for(_loop(), timeout=timeout_s)
    return events


def _conformance_events_of_type(events, t):
    return [e for e in events if e.get("type") == t]


async def _conformance_open_session(host, port, model):
    ws = await websockets.connect(f"ws://{host}:{port}/v1/realtime", max_size=64 * 1024 * 1024)
    await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "model": model}}))
    events = await _conformance_recv_until(ws, lambda evts: any(e["type"] == "session.updated" for e in evts))
    return ws, events


class TestQwen3OmniRealtimeWebSocket:
    @pytest.fixture(scope="class")
    def omni_server(self, request, run_level):
        # Class scope, not module: this process has to exit before the
        # duplex module boots on the same GPUs.
        # Keep equal ``omni_server`` params adjacent; pytest 9.1.1 drops the
        # fixture when the next value differs and will not reuse the earlier one.
        from tests.helpers.fixtures.runtime import omni_fixture_lock
        from tests.helpers.runtime import iter_omni_server

        yield from iter_omni_server(request, run_level, omni_fixture_lock)

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_livekit_client_vad_replay_without_runtime_jit(self, omni_server) -> None:
        """The shared server's first session answers all three replay turns without compiling kernels mid-request.

        Without the MoE prefill warmup, the thinker and talker JIT-compile
        ``fused_moe_kernel`` configs on every turn of this replay (#9069 follow-up).
        The JIT monitor logs only the first compile per kernel, so a green run
        means this replay hit no un-warmed kernel; the warmup log check below is
        what shows the warmup itself ran.
        """
        # Keep this first among the shared server's tests. The connection-count
        # assertion below fails if another realtime session ran before it.
        stalls: list[float] = []
        answers = asyncio.run(
            _run_client_vad_replay(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                CLIENT_VAD_REPLAY_PATH,
                stall_s=stalls,
            )
        )
        print(f"client_vad_replay first-output stall per turn (s): {[round(s, 3) for s in stalls]}")

        assert len(answers) == 3, answers
        assert "qwen" in answers[0].lower(), answers
        assert "paris" in answers[1].lower(), answers
        assert "paris" in answers[2].lower(), answers

        logs = omni_server.read_stage_logs()
        connections = sum(log.count(REALTIME_CONNECTION_LOG_MARKER) for log in logs.values())
        assert connections == 1, (
            f"expected a fresh server whose only realtime session is this replay, saw {connections}"
        )

        runtime_jit = [
            f"stage {stage_id}: {line.strip()}"
            for stage_id, log in sorted(logs.items())
            for line in log.splitlines()
            if RUNTIME_JIT_LOG_MARKER in line
        ]
        assert not runtime_jit, "kernels JIT-compiled mid-request:\n" + "\n".join(runtime_jit)

        warmed = sorted(stage_id for stage_id, log in logs.items() if MOE_PREFILL_WARMUP_LOG_MARKER in log)
        assert warmed == [0, 1], f"MoE prefill warmup should run in the thinker (0) and talker (1), ran in {warmed}"

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_livekit_client_vad_replay(self, omni_server) -> None:
        """Replay the captured client-VAD session at speed 1 and check its answers."""
        answers = asyncio.run(
            _run_client_vad_replay(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                CLIENT_VAD_REPLAY_PATH,
            )
        )

        assert len(answers) == 3, answers
        assert "qwen" in answers[0].lower(), answers
        assert "paris" in answers[1].lower(), answers
        assert "paris" in answers[2].lower(), answers

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_streaming_audio_input_pcm_output_async_chunk(self, omni_server) -> None:
        """Merge CI: async_chunk on, paced upload, full accuracy check."""
        pcm16 = _synthetic_pcm16_input()

        result = asyncio.run(
            _run_realtime_audio_roundtrip(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16,
                chunk_ms=100,
                send_delay_ms=SEND_DELAY_MS,
            )
        )

        _assert_realtime_smoke(result)
        _assert_realtime_accuracy(result)

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_long_audio_response_completes_async_chunk(self, omni_server) -> None:
        """Regression for #6474: long async-chunk audio must emit response.output_audio.done."""
        issue_6474_pcm16 = _synthetic_pcm16_input(
            phrase_text=ISSUE_6474_SYNTH_PHRASE_TEXT,
        )
        issue_6474_result = asyncio.run(
            _run_realtime_audio_roundtrip(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                issue_6474_pcm16,
                chunk_ms=100,
                send_delay_ms=SEND_DELAY_MS,
                completion_timeout_s=180,
            )
        )

        _assert_realtime_smoke(issue_6474_result)
        output_duration_s = len(issue_6474_result["output_pcm"]) / (2 * issue_6474_result["output_sample_rate"])
        assert output_duration_s > 5, (
            f"Expected an issue-like audio response longer than 5 seconds, got {output_duration_s:.2f}s"
        )

    # -- Protocol conformance (same async_chunk server as the cases above) -- #

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_session_lifecycle_events(self, omni_server) -> None:
        """session.created must precede session.updated; both appear exactly once."""

        async def _run():
            ws, events = await _conformance_open_session(omni_server.host, omni_server.port, omni_server.model)
            async with ws:
                created = _conformance_events_of_type(events, "session.created")
                updated = _conformance_events_of_type(events, "session.updated")
                assert len(created) == 1
                assert len(updated) == 1
                ci = next(i for i, e in enumerate(events) if e["type"] == "session.created")
                ui = next(i for i, e in enumerate(events) if e["type"] == "session.updated")
                assert ci < ui, "session.created must precede session.updated"
                assert "id" in created[0].get("session", {})

        asyncio.run(_run())

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_empty_commit_returns_error(self, omni_server) -> None:
        """Committing an empty audio buffer must return an error event."""

        async def _run():
            ws, _ = await _conformance_open_session(omni_server.host, omni_server.port, omni_server.model)
            async with ws:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                events = await _conformance_recv_until(
                    ws, lambda evts: any(e["type"] == "error" for e in evts), timeout_s=30
                )
                errors = _conformance_events_of_type(events, "error")
                assert errors
                assert "empty" in json.dumps(errors[0]).lower()

        asyncio.run(_run())

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_audio_response_parses_tool_call(self, omni_server) -> None:
        """Audio turns must surface function calls, hide their markup, and speak the tool result."""
        result = asyncio.run(_run_realtime_audio_tool_call(omni_server.host, omni_server.port, omni_server.model))

        tool_turn = result["tool_turn"]
        assert len(tool_turn["tool_calls"]) == 1, tool_turn["transcript"]
        call = tool_turn["tool_calls"][0]
        assert call["name"] == "get_weather"
        assert "paris" in (call["arguments"] or "").lower()
        for marker in ("<tool_call>", "</tool_call>"):
            assert marker not in tool_turn["transcript"]
        assert tool_turn["audio_after_tool_call"] == 0, tool_turn

        answer_turn = result["answer_turn"]
        assert answer_turn is not None
        assert answer_turn["audio_deltas"] > 0, "the tool result turn produced no audio"
        assert WEATHER_TOOL_ANSWER in answer_turn["transcript"]
        for marker in ("<tool_call>", "</tool_call>"):
            assert marker not in answer_turn["transcript"]

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_server_vad_server_params, indirect=True)
    def test_server_vad_multi_turn_without_client_commit(
        self,
        cached_silero_vad_artifact: str,
        omni_server,
    ) -> None:
        """Two Qwen turns are endpointed without client commits."""
        assert cached_silero_vad_artifact
        pcm16_24k = _duplex_fixture_pcm(validated_input_wav())

        turns = asyncio.run(
            _run_server_vad_audio_roundtrips(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16_24k,
                chunk_ms=100,
                turns=2,
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": True,
                    "interrupt_response": True,
                },
            )
        )

        assert len(turns) == 2
        updated_session = next(event["session"] for event in turns[0] if event["type"] == "session.updated")
        effective_turn_detection = updated_session["audio"]["input"]["turn_detection"]
        assert effective_turn_detection["type"] == "server_vad"
        assert effective_turn_detection["threshold"] == 0.5
        assert effective_turn_detection["prefix_padding_ms"] == 300
        assert effective_turn_detection["silence_duration_ms"] == 500
        assert effective_turn_detection["create_response"] is True
        assert effective_turn_detection["interrupt_response"] is True
        _assert_auto_committed_turns(turns)

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_sync_server_params, indirect=True)
    def test_streaming_audio_input_pcm_output(self, omni_server) -> None:
        """Merge CI: async_chunk off, no send delay, full accuracy check."""
        pcm16 = _synthetic_pcm16_input()

        result = asyncio.run(
            _run_realtime_audio_roundtrip(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16,
                chunk_ms=100,
                send_delay_ms=0,
            )
        )

        _assert_realtime_smoke(result)
        _assert_realtime_accuracy(result)


def _duplex_fixture_pcm(path: Path, *, duration_s: float | None = None) -> bytes:
    """Replay checked-in speech at the web client's negotiated 24 kHz PCM16 rate."""
    pcm = np.frombuffer(_pcm16_mono_from_wav_bytes(path.read_bytes()), dtype="<i2")
    if duration_s is not None:
        pcm = pcm[: int(duration_s * 16000)]
    positions = np.arange(len(pcm) * 3 // 2) * (2.0 / 3.0)
    return np.interp(positions, np.arange(len(pcm)), pcm).astype("<i2").tobytes()


async def _duplex_receive_until(ws, events: list[dict], predicate, *, timeout_s: float = 180) -> None:
    # One deadline covers the whole operation, including an endless stream that
    # would otherwise reset a per-recv timeout forever.
    async def receive() -> None:
        while not predicate():
            raw = await ws.recv()
            assert isinstance(raw, str), "Expected JSON Realtime events"
            event = json.loads(raw)
            events.append(event)
            assert event.get("type") != "error", event

    await asyncio.wait_for(receive(), timeout=timeout_s)
