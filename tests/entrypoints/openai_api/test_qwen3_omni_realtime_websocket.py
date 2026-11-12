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
from typing import Any

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
from tests.entrypoints.openai_api.helpers import (
    CLIENT_VAD_REPLAY_PATH,
    MODEL,
    SERVER_VAD_MODEL_PATH,
    _assert_auto_committed_turns,
    _duplex_fixture_pcm,
    _pcm16_mono_from_wav_bytes,
    _run_client_vad_replay,
    _run_server_vad_audio_roundtrips,
)
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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


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
