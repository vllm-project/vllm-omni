# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni duplex playback, interruption, and semantic VAD regressions."""

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

import pytest
import websockets
import yaml

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import validated_input_wav, validated_soft_interrupt_wav
from tests.entrypoints.openai_api.helpers import (
    MODEL,
    _assert_auto_committed_turns,
    _duplex_fixture_pcm,
    _duplex_receive_until,
    _run_server_vad_audio_roundtrips,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# These regressions use the engine-owned Qwen duplex plugin. The realtime
# WebSocket module exercises the separate turn/STT deployment.
# ``advanced_model`` (merge) uses 5 interruptions; ``full_model`` (nightly)
# keeps the 20-repeat race hunt. Marks select the case, not ``--run-level``.
_DUPLEX_REPEATS_ADVANCED = 5
_DUPLEX_REPEATS_FULL = 20
_DUPLEX_LONG_INSTRUCTIONS = "Answer the user's question in English using at least eight complete sentences."
_DUPLEX_SHORT_INSTRUCTIONS = "Answer the user's question in one short English sentence."
_SEMANTIC_VAD_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "semantic_vad"


class _DuplexTestRuntime(NamedTuple):
    server: OmniServer
    processor: Any
    artifacts: Path


@pytest.fixture(scope="class")
def qwen_duplex_server(tmp_path_factory, cached_silero_vad_artifact, cached_smart_turn_artifact, run_level):
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
                "duplex_session": {
                    "server_vad_model_path": cached_silero_vad_artifact,
                    "semantic_vad_model_path": cached_smart_turn_artifact,
                },
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


def _duplex_response_id(event: dict) -> str | None:
    return event.get("response_id") or event.get("response", {}).get("id")


def _duplex_response_events(events: list[dict], response_id: str, event_type: str) -> list[dict]:
    return [event for event in events if event["type"] == event_type and _duplex_response_id(event) == response_id]


async def _duplex_send_pcm(
    ws, pcm: bytes, *, paced: bool = False, trailing_silence: bool = True, silence_ms: int = 1000
) -> None:
    if trailing_silence:
        pcm += bytes(24000 * 2 * silence_ms // 1000)
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
async def _duplex_recorded_session(
    runtime,
    log_path: Path,
    *,
    instructions: str,
    vad: bool = True,
    turn_detection: dict | None = None,
    include_image: bool = True,
):
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
                            "type": "realtime",
                            "model": server.model,
                            "instructions": instructions,
                            "temperature": 0,
                            "max_output_tokens": 384 if vad else 64,
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000},
                                    "turn_detection": turn_detection
                                    or {
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
                accepted = session["turn_detection"]
                assert accepted["interrupt_response"] is True
                assert accepted["create_response"] is True
                assert accepted["type"] == (turn_detection or {"type": "server_vad"})["type"]
            if include_image:
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


async def _run_duplex_playback_interruption(
    runtime, log_path: Path, *, generation_finished: bool, turn_detection: dict | None = None
) -> None:
    silence_ms = 5000 if turn_detection and turn_detection["type"] == "semantic_vad" else 1000
    async with _duplex_recorded_session(
        runtime, log_path, instructions=_DUPLEX_LONG_INSTRUCTIONS, turn_detection=turn_detection
    ) as (
        ws,
        events,
        session_id,
    ):
        await _duplex_send_pcm(ws, _duplex_fixture_pcm(validated_input_wav()), silence_ms=silence_ms)
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
            _duplex_send_pcm(
                ws,
                _duplex_fixture_pcm(validated_soft_interrupt_wav(), duration_s=4.7),
                paced=True,
                silence_ms=silence_ms,
            )
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
    assert sum(e["type"] == "input_audio_buffer.committed" for e in events) == 2
    assert sum(e["type"] == "response.created" for e in events) == 2
    _duplex_assert_completed_audio(events, new_id)
    _duplex_assert_prompt(runtime, session_id, audio_count=2, instructions=_DUPLEX_LONG_INSTRUCTIONS, request_count=2)


async def _run_semantic_vad_endpoint(runtime, log_path: Path, *, fixture: str) -> list[dict]:
    """Replay a whole annotated utterance, including pauses, without client commits."""
    pcm = _duplex_fixture_pcm(_SEMANTIC_VAD_ASSETS / fixture)
    async with _duplex_recorded_session(
        runtime,
        log_path,
        instructions=_DUPLEX_SHORT_INSTRUCTIONS,
        turn_detection={"type": "semantic_vad", "eagerness": "auto"},
        include_image=False,
    ) as (ws, events, _):
        sender = asyncio.create_task(_duplex_send_pcm(ws, pcm, paced=True, silence_ms=5000))
        try:
            await _duplex_receive_until(ws, events, lambda: any(e["type"] == "response.done" for e in events))
            await sender
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender
    # session.closed is ordered after all appends. Include late events so an
    # early first reply cannot hide a second commit/reply during the same input.
    return events


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


@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
class TestQwen3OmniDuplexPlayback:
    @pytest.mark.advanced_model
    @pytest.mark.parametrize(
        ("fixture", "min_silence_ms", "max_silence_ms"),
        [
            pytest.param("complete.wav", 1000, 1700, id="complete-before-timeout"),
            pytest.param("short_pause.wav", 1000, 1700, id="hold-800ms-clause-pause"),
            pytest.param("thinking_pause.wav", 1000, 1700, id="hold-3800ms-incomplete-pause"),
            pytest.param("abandoned.wav", 3800, 4400, id="incomplete-timeout"),
        ],
    )
    def test_semantic_vad_endpoint_timing(
        self, qwen_duplex_server, tmp_path: Path, fixture: str, min_silence_ms: int, max_silence_ms: int
    ) -> None:
        """Guard the 1200 ms default, real semantic hold, and 4000 ms fallback."""
        events = asyncio.run(_run_semantic_vad_endpoint(qwen_duplex_server, tmp_path / "events.json", fixture=fixture))
        event_types = [event["type"] for event in events]
        sequence = [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "response.created",
            "response.output_audio.done",
            "response.done",
        ]
        for event_type in sequence:
            assert event_types.count(event_type) == 1, event_types
        positions = [event_types.index(event_type) for event_type in sequence]
        assert positions == sorted(positions)
        started, stopped, committed = (events[position] for position in positions[:3])
        assert started["item_id"] == stopped["item_id"] == committed["item_id"]
        # The fixtures end at the annotated speech end, without trailing silence.
        # Use the input audio clock, not network or GPU response latency. Bounds
        # allow Silero frame rounding and a small acoustic end-of-speech offset.
        with wave.open(str(_SEMANTIC_VAD_ASSETS / fixture), "rb") as wav:
            speech_end_ms = wav.getnframes() * 1000 / wav.getframerate()
        silence_ms = stopped["audio_end_ms"] - speech_end_ms
        assert min_silence_ms <= silence_ms <= max_silence_ms, (fixture, silence_ms, stopped)
        response_id = _duplex_response_id(events[positions[3]])
        assert response_id
        _duplex_assert_completed_audio(events, response_id)

    @pytest.mark.advanced_model
    @pytest.mark.parametrize("generation_finished", [False, True], ids=["generating", "queued-playback"])
    @pytest.mark.parametrize("repetition", range(2))
    def test_semantic_vad_speech_interrupts_playback(
        self, qwen_duplex_server, tmp_path: Path, generation_finished: bool, repetition: int
    ) -> None:
        """One spoken interruption produces exactly one follow-up with semantic VAD."""
        asyncio.run(
            _run_duplex_playback_interruption(
                qwen_duplex_server,
                tmp_path / f"events-{repetition}.json",
                generation_finished=generation_finished,
                turn_detection={"type": "semantic_vad", "eagerness": "auto"},
            )
        )

    @pytest.mark.advanced_model
    @pytest.mark.parametrize("generation_finished", [False, True], ids=["generating", "queued-playback"])
    @pytest.mark.parametrize("repetition", range(_DUPLEX_REPEATS_ADVANCED))
    def test_speech_interrupts_playback(
        self, qwen_duplex_server, tmp_path: Path, generation_finished: bool, repetition: int
    ):
        """Reuse one live deployment; every repetition opens an independent call."""
        asyncio.run(
            _run_duplex_playback_interruption(
                qwen_duplex_server, tmp_path / f"events-{repetition}.json", generation_finished=generation_finished
            )
        )

    @pytest.mark.full_model
    @hardware_test(res={"rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("generation_finished", [False, True], ids=["generating", "queued-playback"])
    @pytest.mark.parametrize("repetition", range(_DUPLEX_REPEATS_FULL))
    def test_speech_interrupts_playback_full(
        self, qwen_duplex_server, tmp_path: Path, generation_finished: bool, repetition: int
    ):
        """Same contract as ``test_speech_interrupts_playback``, 20 nightly repeats."""
        asyncio.run(
            _run_duplex_playback_interruption(
                qwen_duplex_server, tmp_path / f"events-{repetition}.json", generation_finished=generation_finished
            )
        )

    @pytest.mark.advanced_model
    def test_semantic_vad_multi_turn_without_client_commit(self, qwen_duplex_server) -> None:
        """Two Qwen turns are endpointed by Silero + Smart Turn without client commits."""
        server = qwen_duplex_server.server
        turns = asyncio.run(
            _run_server_vad_audio_roundtrips(
                server.host,
                server.port,
                server.model,
                _duplex_fixture_pcm(validated_input_wav()),
                chunk_ms=100,
                turns=2,
                turn_detection={"type": "semantic_vad", "eagerness": "auto"},
                # Longer than the 4 s ``auto`` limit, so the turn ends even
                # when Smart Turn calls it incomplete.
                silence_ms=5000,
                duplex=True,
            )
        )

        assert len(turns) == 2
        updated_session = next(event["session"] for event in turns[0] if event["type"] == "session.updated")
        # The normalized (defaults filled) setting is the top-level ``turn_detection``.
        effective_turn_detection = updated_session["turn_detection"]
        assert effective_turn_detection["type"] == "semantic_vad"
        assert effective_turn_detection["eagerness"] == "auto"
        assert effective_turn_detection["create_response"] is True
        _assert_auto_committed_turns(turns)

    @pytest.mark.advanced_model
    def test_mixed_image_audio_history_pruning(self, qwen_duplex_server, tmp_path: Path):
        asyncio.run(_run_duplex_history_pruning(qwen_duplex_server, tmp_path / "events.json"))
