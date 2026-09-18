# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E online tests for Qwen3-Omni /v1/realtime WebSocket (streaming PCM in, audio out).

Four scenarios:
- Ready CI: async_chunk on, smoke only (no send delay, no accuracy check).
- Merge CI: async_chunk on + send delay, full accuracy check.
- Merge CI: async_chunk off, no send delay, full accuracy check.
- Server VAD: two turns without client commits.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import websockets
import yaml

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import validated_input_wav, validated_soft_interrupt_wav
from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    convert_audio_bytes_to_text,
    cosine_similarity_text,
    generate_synthetic_audio,
)
from tests.helpers.runtime import OmniServer, OmniServerParams, get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_FILENAME,
    SILERO_VAD_REPO_ID,
    SILERO_VAD_REVISION,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("VLLM_OMNI_TEST_QWEN3_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SERVER_VAD_MODEL_PATH = os.environ.get("VLLM_OMNI_TEST_SILERO_VAD_MODEL_PATH")

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
default_stage_config = get_deploy_config_path("ci/qwen3_omni_moe.yaml")
server_vad_stage_config = modify_stage_config(
    default_stage_config,
    {
        "async_chunk": True,
        "duplex_session": (
            {"server_vad_model_path": SERVER_VAD_MODEL_PATH} if SERVER_VAD_MODEL_PATH is not None else {}
        ),
    },
)

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

realtime_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=default_stage_config,
            use_stage_cli=True,
            server_args=["--async-chunk"],
        ),
        id="async_chunk",
    ),
]

realtime_async_chunk_1gpu_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_omni_moe_1gpu.yaml"),
            use_stage_cli=True,
            # The replayed session's session.update sets tool_choice="auto"
            # with tools=[]; without a configured tool-call parser, tool-call
            # markup the model emits has nowhere to be intercepted and can
            # leak into the transcript stream.
            server_args=[
                "--async-chunk",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
            ],
            # This config colocates all three stages on one GPU. The stage-CLI
            # flow launches each stage as an independent process with no
            # cross-process memory-profiling lock (unlike ``vllm serve --omni
            # --deploy``), so firing them a couple seconds apart lets stage 1
            # profile GPU memory before stage 0's own footprint is committed
            # and OOM. Serialize the launches instead.
            sequential_stage_launch=True,
        ),
        id="async_chunk_1gpu",
    ),
]

realtime_server_vad_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=server_vad_stage_config,
            use_stage_cli=True,
        ),
        id="server_vad",
    ),
]


def _pcm16_mono_16k_from_wav_bytes(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, sampwidth={wf.getsampwidth()}")
        if wf.getframerate() != 16000:
            raise ValueError(f"Expected 16 kHz input for /v1/realtime, got {wf.getframerate()} Hz")
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

    bytes_per_ms = 16000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

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

        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=completion_timeout_s)
            if isinstance(message, bytes):
                continue

            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "session.created":
                continue

            if event_type == "response.output_audio.delta":
                delta_events += 1
                sr = event.get("sample_rate_hz")
                if isinstance(sr, int) and sr > 0:
                    output_sr = sr
                audio_b64 = event.get("audio", "")
                if audio_b64:
                    incremental.append(base64.b64decode(audio_b64))
                continue

            if event_type == "transcription.delta":
                d = event.get("delta", "")
                if d:
                    text_chunks.append(d)
                continue

            if event_type == "transcription.done":
                final_text = event.get("text", "") or "".join(text_chunks)
                continue

            if event_type == "response.output_audio.done":
                break

            if event_type == "error":
                raise AssertionError(f"WebSocket error: {event}")

            raise AssertionError(f"Unexpected WebSocket event: {event}")

    out_pcm = b"".join(incremental)
    return {
        "output_pcm": out_pcm,
        "output_sample_rate": output_sr,
        "transcription_text": final_text if final_text else "".join(text_chunks),
        "delta_events": delta_events,
    }


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
    pcm16: bytes,
    *,
    chunk_ms: int = 100,
    turns: int = 2,
) -> list[list[dict]]:
    chunk_bytes = max(16_000 * 2 // 1000 * chunk_ms, 2)
    turn_events: list[list[dict]] = []

    async with websockets.connect(f"ws://{host}:{port}/v1/realtime", max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": model,
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": 16_000},
                                "turn_detection": {"type": "server_vad"},
                            }
                        },
                    },
                }
            )
        )

        silence = bytes(16_000 * 2 * 3 // 2)
        for _ in range(turns):
            await _append_pcm16_chunks(ws, pcm16, chunk_bytes)
            await _append_pcm16_chunks(ws, silence, chunk_bytes)
            turn_events.append(await _receive_server_vad_turn(ws))

    return turn_events


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
) -> list[str]:
    """Replay a captured LiveKit client-VAD session at its original speed."""
    with replay_path.open() as replay_file:
        records = [json.loads(line) for line in replay_file]
    assert records

    answers_by_response_id: dict[str, str] = {}
    response_order: list[str] = []
    completed_answers: dict[str, str] = {}
    errors: list[dict] = []

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
                elif event_type in {
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                    "response.output_text.delta",
                    "transcription.delta",
                }:
                    response_id = event.get("response_id")
                    if response_id in answers_by_response_id:
                        answers_by_response_id[response_id] += event.get("delta", "")
                elif event_type == "response.done":
                    response = event["response"]
                    response_id = response["id"]
                    text = answers_by_response_id.get(response_id, "")
                    completed_answers[response_id] = text or _output_text(response)

        receiver = asyncio.create_task(receive_responses())
        capture_start = records[0]["ts"]
        replay_start = asyncio.get_running_loop().time()

        try:
            for record in records:
                target = replay_start + (record["ts"] - capture_start)
                delay = target - asyncio.get_running_loop().time()
                if delay > 0:
                    await asyncio.sleep(delay)

                message = record["data"]
                # The capture's reference-voice update is only valid when the
                # replay client uploads its optional reference-audio asset.
                if "__VOICE__" in json.dumps(message):
                    continue
                await ws.send(json.dumps(message))

            await asyncio.sleep(wait_s)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    assert not [error for error in errors if error.get("error", {}).get("type") == "server_error"]
    return [completed_answers[response_id] for response_id in response_order if response_id in completed_answers]


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
        sample_rate=16000,
        phrase_text=phrase_text,
    )
    wav_bytes = base64.b64decode(syn["base64"])
    return _pcm16_mono_16k_from_wav_bytes(wav_bytes)


def _server_vad_pcm16_input() -> bytes:
    """Load the fixed single-turn speech fixture used by the Server VAD E2E."""
    return _pcm16_mono_16k_from_wav_bytes(validated_input_wav().read_bytes())


def _assert_realtime_smoke(result: dict) -> None:
    out_pcm = result["output_pcm"]
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


class TestQwen3OmniRealtimeWebSocket:
    @pytest.fixture(scope="class")
    def omni_server(self, request, run_level):
        # Release the legacy turn deployment before the duplex class below
        # starts another model server on the same pair of GPUs.
        from tests.helpers.fixtures.runtime import omni_fixture_lock
        from tests.helpers.runtime import iter_omni_server

        yield from iter_omni_server(request, run_level, omni_fixture_lock)

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_1gpu_server_params, indirect=True)
    def test_livekit_client_vad_replay_1gpu(self, omni_server) -> None:
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
        # Stage 0 sampling uses top_k=1 (effectively greedy), so these
        # responses are deterministic given fixed weights/inputs.
        assert answers[0] == (
            "Hello! I'm Qwen-Omni, a multimodal large-scale language model developed by "
            "Alibaba's Tongyi Lab. How can I assist you?"
        ), answers
        assert answers[1] == "The capital of France is Paris.", answers
        assert answers[2] == (
            "As of the most recent data, the population of Paris (the city proper) is "
            "approximately 2.1 million people. However, if you include the larger "
            "metropolitan area known as *Île-de-France*, the population exceeds 12 "
            "million, making it the largest urban area in the European Union."
        ), answers

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
        assert "assistant" in answers[0].lower(), answers
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
        deploy = load_deploy_config(server_vad_stage_config)
        assert deploy.session_mode == "turn"
        assert deploy.async_chunk is True
        assert deploy.duplex_session.server_vad_model_path == SERVER_VAD_MODEL_PATH
        pcm16 = _server_vad_pcm16_input()

        turns = asyncio.run(
            _run_server_vad_audio_roundtrips(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16,
                chunk_ms=100,
                turns=2,
            )
        )

        assert len(turns) == 2
        updated_session = next(event["session"] for event in turns[0] if event["type"] == "session.updated")
        effective_turn_detection = updated_session["audio"]["input"]["turn_detection"]
        assert effective_turn_detection["type"] == "server_vad"
        assert effective_turn_detection["silence_duration_ms"] == 500
        assert effective_turn_detection["create_response"] is True
        assert effective_turn_detection["interrupt_response"] is False
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
            assert done["status"] == "completed"
            input_item_ids.append(input_item_id)
            response_ids.append(created["id"])

        assert len(set(input_item_ids)) == 2
        assert len(set(response_ids)) == 2

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


# These regressions use the engine-owned Qwen duplex plugin. The older class
# above intentionally exercises the separate turn/STT deployment.
_DUPLEX_REPEATS = int(os.environ.get("VLLM_OMNI_TEST_QWEN_DUPLEX_REPEATS", "20"))
_DUPLEX_LONG_INSTRUCTIONS = "Answer the user's question in English using at least eight complete sentences."
_DUPLEX_SHORT_INSTRUCTIONS = "Answer the user's question in one short English sentence."


class _DuplexTestRuntime(NamedTuple):
    server: OmniServer
    processor: Any
    artifacts: Path


@pytest.fixture(scope="class")
def qwen_duplex_server(tmp_path_factory, cached_silero_vad_artifact, run_level):
    """Load real weights once for all interruption repetitions and history checks."""
    from transformers import AutoProcessor

    assert run_level in {"advanced_model", "full_model"}, "Duplex playback acceptance requires real weights"
    artifacts = tmp_path_factory.mktemp("qwen_duplex_playback")
    deploy = artifacts / "deploy.yaml"
    # An absolute base keeps the production YAML's relative inheritance intact
    # when the test's Silero override lives in a temporary directory.
    deploy.write_text(
        yaml.safe_dump(
            {
                "base_config": get_deploy_config_path("qwen3_omni_duplex.yaml"),
                "duplex_session": {"server_vad_model_path": cached_silero_vad_artifact},
            }
        )
    )
    assert load_deploy_config(deploy).session_mode == "duplex"
    model = get_model_prefix() + MODEL
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    with OmniServer(
        model,
        ["--deploy-config", str(deploy), "--init-timeout", "900", "--stage-init-timeout", "600"],
        env_dict={"VLLM_OMNI_QWEN_VISUAL_DEBUG_DIR": str(artifacts)},
    ) as server:
        yield _DuplexTestRuntime(server=server, processor=processor, artifacts=artifacts)


def _duplex_fixture_pcm(path: Path, *, duration_s: float | None = None) -> bytes:
    """Replay checked-in speech at the web client's negotiated 24 kHz PCM16 rate."""
    pcm = np.frombuffer(_pcm16_mono_16k_from_wav_bytes(path.read_bytes()), dtype="<i2")
    if duration_s is not None:
        pcm = pcm[: int(duration_s * 16000)]
    positions = np.arange(len(pcm) * 3 // 2) * (2.0 / 3.0)
    return np.interp(positions, np.arange(len(pcm)), pcm).astype("<i2").tobytes()


def _duplex_response_id(event: dict) -> str | None:
    return event.get("response_id") or event.get("response", {}).get("id")


def _duplex_response_events(events: list[dict], response_id: str, event_type: str) -> list[dict]:
    return [event for event in events if event["type"] == event_type and _duplex_response_id(event) == response_id]


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


async def _duplex_send_pcm(ws, pcm: bytes, *, paced: bool = False, trailing_silence: bool = True) -> None:
    if trailing_silence:
        pcm += bytes(24000 * 2)  # One second exceeds the configured 500 ms endpoint.
    chunk_bytes = 24000 * 2 // 5
    for offset in range(0, len(pcm), chunk_bytes):
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[offset : offset + chunk_bytes]).decode(),
                }
            )
        )
        if paced:
            await asyncio.sleep(0.2)


async def _duplex_add_image(ws) -> None:
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (448, 336), (32, 96, 160)).save(image, format="JPEG")
    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "camera_1",
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(image.getvalue()).decode(),
                        }
                    ],
                },
            }
        )
    )


@asynccontextmanager
async def _duplex_recorded_session(runtime, log_path: Path, *, instructions: str, vad: bool = True):
    events: list[dict] = []
    server = runtime.server
    async with websockets.connect(
        f"ws://{server.host}:{server.port}/v1/realtime?duplex=1", max_size=64 * 1024 * 1024
    ) as ws:
        failed = True
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "model": server.model,
                            "instructions": instructions,
                            "temperature": 0,
                            "max_output_tokens": 384 if vad else 64,
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000},
                                    "turn_detection": {
                                        "type": "server_vad",
                                        "threshold": 0.5,
                                        "prefix_padding_ms": 300,
                                        "silence_duration_ms": 500,
                                        "create_response": True,
                                        "interrupt_response": True,
                                    }
                                    if vad
                                    else None,
                                },
                                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                            },
                        },
                    }
                )
            )
            await _duplex_receive_until(ws, events, lambda: any(e["type"] == "session.updated" for e in events))
            session = next(e["session"] for e in events if e["type"] == "session.updated")
            if vad:
                accepted = session["audio"]["input"]["turn_detection"]
                assert accepted["interrupt_response"] is True
                assert accepted["create_response"] is True
            await _duplex_add_image(ws)
            yield ws, events, session["id"]
            failed = False
        finally:
            # Closing the transport alone leaves a resumable session occupying
            # the deployment's single session slot. Always release the session.
            try:
                await ws.send(json.dumps({"type": "session.close"}))
                await _duplex_receive_until(
                    ws, events, lambda: any(e["type"] == "session.closed" for e in events), timeout_s=15
                )
            except Exception:
                if not failed:
                    raise
            finally:
                log_path.write_text(json.dumps(events, ensure_ascii=False, indent=2))


def _duplex_assert_prompt(
    runtime,
    session_id: str,
    *,
    audio_count: int,
    instructions: str,
    request_count: int,
    assistant_texts: list[str] | None = None,
):
    captures = []
    for path in runtime.artifacts.glob("*/input.json"):
        capture = json.loads(path.read_text())
        if duplex_resource_request_belongs_to_session(capture["request_id"], session_id):
            captures.append((path.stat().st_mtime_ns, capture))
    assert len(captures) == request_count, captures
    capture = max(captures, key=lambda item: item[0])[1]
    assert capture["audio_count"] == audio_count
    # The camera belongs to the first user turn and leaves with its audio.
    image_retained = request_count <= 4
    assert len(capture["images"]) == int(image_retained)
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    if assistant_texts is None:
        assistant_texts = [""] * (audio_count - 1)
    assert len(assistant_texts) == audio_count - 1
    for index in range(audio_count):
        if index:
            messages.append({"role": "assistant", "content": assistant_texts[index - 1]})
        parts = [{"type": "image"}] if index == 0 and image_retained else []
        parts.append({"type": "audio"})
        messages.append({"role": "user", "content": parts})
    expected = runtime.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert capture["prompt"] == expected
    # Use the checkpoint tokenizer too: a repr(messages) processor stub cannot
    # establish that an empty assistant survives as a real turn separator.
    tokenizer = runtime.processor.tokenizer
    actual_ids = tokenizer.encode(capture["prompt"], add_special_tokens=False)
    separator_ids = tokenizer.encode("<|im_start|>assistant\n<|im_end|>\n", add_special_tokens=False)
    assert sum(
        actual_ids[index : index + len(separator_ids)] == separator_ids for index in range(len(actual_ids))
    ) == assistant_texts.count("")


def _duplex_assert_completed_audio(events: list[dict], response_id: str) -> None:
    terminal = _duplex_response_events(events, response_id, "response.done")
    assert len(terminal) == 1, terminal
    assert terminal[0]["response"]["status"] == "completed", terminal
    audio = _duplex_response_events(events, response_id, "response.output_audio.delta")
    assert sum(len(base64.b64decode(event["delta"])) for event in audio) > 4800
    assert any(
        event.get("delta")
        for event in events
        if _duplex_response_id(event) == response_id
        and event["type"] in {"response.output_text.delta", "response.output_audio_transcript.delta"}
    ), "The follow-up must produce text as well as audio"


async def _run_duplex_playback_interruption(runtime, log_path: Path, *, generation_finished: bool) -> None:
    async with _duplex_recorded_session(runtime, log_path, instructions=_DUPLEX_LONG_INSTRUCTIONS) as (
        ws,
        events,
        session_id,
    ):
        await _duplex_send_pcm(ws, _duplex_fixture_pcm(validated_input_wav()))
        await _duplex_receive_until(ws, events, lambda: any(e["type"] == "response.created" for e in events))
        old_id = next(_duplex_response_id(e) for e in events if e["type"] == "response.created")
        assert old_id
        await _duplex_receive_until(
            ws,
            events,
            lambda: (
                sum(
                    len(base64.b64decode(e["delta"]))
                    for e in _duplex_response_events(events, old_id, "response.output_audio.delta")
                )
                > 4800
            ),
        )
        if generation_finished:
            await _duplex_receive_until(ws, events, lambda: _duplex_response_events(events, old_id, "response.done"))
        await ws.send(
            json.dumps({"type": "playback.ack", "response_id": old_id, "played_ms": 100, "committed_ms": 100})
        )
        await _duplex_receive_until(ws, events, lambda: any(e["type"] == "playback.acknowledged" for e in events))
        ack = next(e["event"] for e in events if e["type"] == "playback.acknowledged")
        assert ack["item_id"] == f"item_{old_id}"
        assert ack["committed_ms"] == 100
        assert ack["playback"]["sent_ms"] > 100
        cursor = len(events)
        sender = asyncio.create_task(
            _duplex_send_pcm(ws, _duplex_fixture_pcm(validated_soft_interrupt_wav(), duration_s=4.7), paced=True)
        )
        try:
            await _duplex_receive_until(
                ws, events, lambda: any(e["type"] == "input_audio_buffer.speech_started" for e in events[cursor:])
            )
            speech_index = next(
                i for i in range(cursor, len(events)) if events[i]["type"] == "input_audio_buffer.speech_started"
            )
            if not generation_finished:
                assert not _duplex_response_events(events[:speech_index], old_id, "response.done"), (
                    "Generation completed before speech interruption; the in-progress scenario was not exercised"
                )
            await _duplex_receive_until(
                ws,
                events,
                lambda: any(e["type"] == "response.done" and _duplex_response_id(e) != old_id for e in events[cursor:]),
            )
            await sender
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
        new_id = next(
            _duplex_response_id(e)
            for e in events[cursor:]
            if e["type"] == "response.created" and _duplex_response_id(e) != old_id
        )
    # Include everything observed through session.closed when checking duplicates.
    terminals = _duplex_response_events(events, old_id, "response.done")
    assert len(terminals) == 1, terminals
    assert terminals[0]["response"]["status"] == ("completed" if generation_finished else "cancelled")
    if generation_finished:
        cleared = _duplex_response_events(events, old_id, "output_audio_buffer.cleared")
        assert len(cleared) == 1, cleared
        boundary = events.index(cleared[0])
    else:
        boundary = events.index(terminals[0])
    assert not _duplex_response_events(events[boundary + 1 :], old_id, "response.output_audio.delta")
    assert new_id and new_id != old_id
    _duplex_assert_completed_audio(events, new_id)
    _duplex_assert_prompt(runtime, session_id, audio_count=2, instructions=_DUPLEX_LONG_INSTRUCTIONS, request_count=2)


async def _run_duplex_history_pruning(runtime, log_path: Path) -> None:
    async with _duplex_recorded_session(runtime, log_path, instructions=_DUPLEX_SHORT_INSTRUCTIONS, vad=False) as (
        ws,
        events,
        session_id,
    ):
        response_ids = []
        heard_texts: list[str] = []
        pcm = _duplex_fixture_pcm(validated_input_wav())
        for turn in range(6):
            cursor = len(events)
            await _duplex_send_pcm(ws, pcm, trailing_silence=False)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))
            await _duplex_receive_until(ws, events, lambda: any(e["type"] == "response.done" for e in events[cursor:]))
            response_id = next(_duplex_response_id(e) for e in events[cursor:] if e["type"] == "response.done")
            assert response_id and response_id not in response_ids
            response_ids.append(response_id)
            _duplex_assert_completed_audio(events, response_id)
            _duplex_assert_prompt(
                runtime,
                session_id,
                audio_count=min(turn + 1, 4),
                instructions=_DUPLEX_SHORT_INSTRUCTIONS,
                request_count=turn + 1,
                assistant_texts=heard_texts[-3:],
            )
            # Complete playback so pruning is checked against real assistant
            # answers paired with retained user audio. Repeated empty assistant
            # turns can make Qwen emit EOS; interruption tests cover those slots.
            done = _duplex_response_events(events, response_id, "response.done")[0]["response"]
            played_ms = done["metadata"]["playback"]["sent_ms"]
            heard_texts.append(
                "".join(
                    e["delta"]
                    for e in _duplex_response_events(events, response_id, "response.output_audio_transcript.delta")
                ).strip()
            )
            assert heard_texts[-1]
            cursor = len(events)
            await ws.send(
                json.dumps(
                    {
                        "type": "playback.ack",
                        "response_id": response_id,
                        "played_ms": played_ms,
                        "committed_ms": played_ms,
                    }
                )
            )
            await _duplex_receive_until(
                ws, events, lambda: any(e["type"] == "playback.acknowledged" for e in events[cursor:])
            )
            ack = next(e["event"] for e in events[cursor:] if e["type"] == "playback.acknowledged")
            assert ack["item_id"] == f"item_{response_id}"
            assert ack["committed_ms"] == ack["playback"]["sent_ms"] == played_ms
            assert ack["history_committed"] is True
    for response_id in response_ids:
        _duplex_assert_completed_audio(events, response_id)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
class TestQwen3OmniDuplexPlayback:
    @pytest.mark.parametrize("generation_finished", [False, True], ids=["generating", "queued-playback"])
    @pytest.mark.parametrize("repetition", range(_DUPLEX_REPEATS))
    def test_speech_interrupts_playback(
        self, qwen_duplex_server, tmp_path: Path, generation_finished: bool, repetition: int
    ):
        """Reuse one live deployment; every repetition opens an independent call."""
        asyncio.run(
            _run_duplex_playback_interruption(
                qwen_duplex_server, tmp_path / f"events-{repetition}.json", generation_finished=generation_finished
            )
        )

    def test_mixed_image_audio_history_pruning(self, qwen_duplex_server, tmp_path: Path):
        asyncio.run(_run_duplex_history_pruning(qwen_duplex_server, tmp_path / "events.json"))
