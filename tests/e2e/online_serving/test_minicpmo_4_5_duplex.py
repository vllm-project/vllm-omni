# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CI coverage for the MiniCPM-o 4.5 native-duplex Realtime API."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
import websockets

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    demo_args,
    duplex_camera_frames,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import (
    _ref_audio_data_url,
    run_demo,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_multi_session,
)
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import build_realtime_url
from vllm_omni.experimental.fullduplex.video_stacking import concat_frames_b64

pytestmark = pytest.mark.omni


def _assert_positive_int(value: object) -> None:
    assert isinstance(value, int)
    assert value > 0


def _assert_request_metrics(metrics: object, *, expected_count: int) -> None:
    assert isinstance(metrics, list)
    assert len(metrics) == expected_count
    for request_index, request in enumerate(metrics):
        assert isinstance(request["session_id"], str)
        assert request["request_index"] == request_index
        assert isinstance(request["response_id"], str)
        assert request["ttft_ms"] is not None and request["ttft_ms"] >= 0
        assert request["ttfp_ms"] >= 0
        assert request["rtf"] is not None and request["rtf"] >= 0
        assert request["audio_generation_ms"] >= 0
        assert request["audio_duration_ms"] > 0


def _assert_session_metrics(metrics: object, *, expected_count: int) -> None:
    assert isinstance(metrics, dict)
    assert isinstance(metrics["session_id"], str)
    assert metrics["audio_turn_count"] == expected_count
    assert metrics["mean_ttft_ms"] is not None and metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_ttfp_ms"] is not None and metrics["mean_ttfp_ms"] >= 0
    assert metrics["mean_rtf"] is not None and metrics["mean_rtf"] >= 0


async def _receive_protocol_events(ws, required_types: set[str], *, timeout_s: float) -> list[dict[str, object]]:
    async def receive() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        seen: set[str] = set()
        while not required_types.issubset(seen):
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            event_type = event.get("type")
            if event_type == "error":
                raise AssertionError(f"WebSocket protocol smoke received an error: {event}")
            if isinstance(event_type, str):
                seen.add(event_type)
        return events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def _run_protocol_smoke(*, url: str, model: str, ref_audio: Path) -> list[dict[str, object]]:
    websocket_url = build_realtime_url(url, model, autostart=False)
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                    },
                }
            )
        )
        events = await _receive_protocol_events(
            ws,
            {"session.created", "session.updated"},
            timeout_s=60,
        )
        await ws.send(json.dumps({"type": "session.close"}))
        events.extend(await _receive_protocol_events(ws, {"session.closed"}, timeout_s=60))
    return events


async def _run_text_only_response_create(
    *,
    url: str,
    model: str,
    ref_audio: Path,
    text: str,
    timeout_s: float = 120.0,
) -> dict[str, object]:
    """Drive a text prompt with no audio and return the server's answer.

    A Realtime client asks for speech from text with a user message item
    followed by ``response.create``. A model-native duplex session generates
    from audio units, so this returns whatever the server answers rather than
    assuming a response.
    """
    websocket_url = build_realtime_url(url, model, autostart=False)
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": model,
                        "modalities": ["audio", "text"],
                        "output_audio_format": "pcm16",
                        "turn_detection": None,
                        "temperature": 0.0,
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {"auto_response": False},
                    },
                }
            )
        )
        await _receive_protocol_events(ws, {"session.created", "session.updated"}, timeout_s=60)
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create"}))

        async def receive_outcome() -> dict[str, object]:
            while True:
                raw = await ws.recv()
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                if isinstance(event, dict) and event.get("type") in {"error", "response.done"}:
                    return event

        return await asyncio.wait_for(receive_outcome(), timeout=timeout_s)


async def _run_seeded_text_to_audio(
    *,
    url: str,
    model: str,
    ref_audio: Path | None,
    text: str,
    modalities: tuple[str, ...] = ("audio", "text"),
    silence_seconds: float = 12.0,
    timeout_s: float = 180.0,
) -> dict[str, object]:
    """Speak a seeded text: the duplex route's text-to-speech shape.

    A model-native session takes its text once, in the session context
    (``duplex_initial_user_text``), and then generates per audio unit. Silence
    carries no content of its own, so it only advances the clock and lets the
    model answer the seeded turn.
    """
    websocket_url = build_realtime_url(url, model, autostart=False)
    audio_bytes = 0
    transcript: list[str] = []
    output_text: list[str] = []
    seen: list[str] = []
    session_payload: dict[str, object] = {
        "model": model,
        "modalities": list(modalities),
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": None,
        "temperature": 0.0,
        "extra_body": {
            "auto_response": True,
            "force_listen_count": 0,
            "duplex_initial_user_text": text,
        },
    }
    if ref_audio is not None:
        session_payload["ref_audio"] = _ref_audio_data_url(str(ref_audio))
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": session_payload}))

        async def reader() -> None:
            nonlocal audio_bytes
            while True:
                raw = await ws.recv()
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                if isinstance(event_type, str):
                    seen.append(event_type)
                delta = event.get("delta")
                if event_type == "response.output_audio.delta" and isinstance(delta, str):
                    audio_bytes += len(base64.b64decode(delta))
                elif event_type == "response.output_audio_transcript.delta" and isinstance(delta, str):
                    transcript.append(delta)
                elif event_type == "response.output_text.delta" and isinstance(delta, str):
                    output_text.append(delta)
                elif event_type == "error":
                    raise AssertionError(f"seeded text turn received an error: {event}")

        reader_task = asyncio.create_task(reader())
        silence = bytes(2 * 16_000 * 200 // 1000)
        sent_ms = 0
        try:
            while sent_ms < silence_seconds * 1000 and "response.done" not in seen:
                sent_ms += 200
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(silence).decode("ascii"),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": 16_000,
                            "duration_ms": 200,
                            "audio_end_ms": sent_ms,
                        }
                    )
                )
                await asyncio.sleep(0.2)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s
            while loop.time() < deadline and "response.done" not in seen:
                await asyncio.sleep(0.5)
            if reader_task.done():
                await reader_task
            # Close explicitly. Dropping the socket parks the session in its
            # disconnect grace, where it keeps holding an admission slot and
            # starves the tests that follow.
            await ws.send(json.dumps({"type": "session.close"}))
            deadline = loop.time() + 30
            while loop.time() < deadline and "session.closed" not in seen:
                await asyncio.sleep(0.25)
        finally:
            reader_task.cancel()
    return {
        "audio_bytes": audio_bytes,
        "transcript": "".join(transcript),
        "output_text": "".join(output_text),
        "event_types": seen,
    }


@pytest.mark.core_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_websocket_protocol_smoke(omni_server) -> None:
    ref_audio = resolve_ref_audio()
    events = asyncio.run(
        _run_protocol_smoke(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=ref_audio,
        )
    )
    event_types = [event.get("type") for event in events]
    assert "session.created" in event_types
    assert "session.updated" in event_types
    assert event_types[-1] == "session.closed"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_response_required(omni_server, tmp_path: Path) -> None:
    args = demo_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "single_session",
    )
    args.turns = 2
    # Every turn replays the same active-speech window as the first one. The default
    # shorter follow-up window is a different mid-utterance slice, which the native
    # duplex model may legitimately answer with "listen" instead of a response.
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    result = asyncio.run(run_demo(args))
    assert result["ok"] is True
    _assert_positive_int(result["audio_delta_count"])
    assert result["done_count"] == 2
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True
    _assert_request_metrics(result["request_metrics"], expected_count=2)
    _assert_session_metrics(result["session_metrics"], expected_count=2)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_video_input(omni_server, tmp_path: Path) -> None:
    """Audio plus a 1 fps camera track, the omni-duplex video contract.

    Frames ride the audio appends, so the response contract is unchanged: what
    this covers is that interleaved vision input keeps the turn intact instead
    of stalling or erroring out mid-segment.
    """
    args = demo_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "video_input",
    )
    args.turns = 2
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    frames = duplex_camera_frames(seconds=4, cache_dir=tmp_path / "camera")
    args.video_frames_b64 = frames
    # Each unit also carries a composite of its interior sub-frames, so the
    # append exercises the official two-image frame_list that carries motion.
    args.video_stacked_frames_b64 = [concat_frames_b64([frames[index]] * 2) for index in range(len(frames))]

    result = asyncio.run(run_demo(args))

    assert result["ok"] is True
    assert result["video_frame_count"] == 4
    assert result["video_stacked_frame_count"] == 4
    _assert_positive_int(result["audio_delta_count"])
    assert result["done_count"] == 2
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True
    _assert_request_metrics(result["request_metrics"], expected_count=2)
    _assert_session_metrics(result["session_metrics"], expected_count=2)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize(
    ("locale", "text"),
    [
        ("en", "Please say exactly: the quick brown fox jumps over the lazy dog."),
        ("zh", "请朗读：今天天气很好，我们一起去公园散步。"),
    ],
)
def test_duplex_seeded_text_to_audio(omni_server, locale: str, text: str) -> None:
    """text -> audio over the duplex route, in both locales.

    This is the case the deleted turn-based ``text -> audio`` tests covered.
    It is also the shape the Seed-TTS Realtime backend uses, so it guards the
    benchmark path as well as the modality.
    """
    result = asyncio.run(
        _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            text=text,
        )
    )

    assert "response.done" in result["event_types"], result["event_types"]
    assert int(result["audio_bytes"]) > 0, f"{locale} seeded text produced no audio"
    assert str(result["transcript"]).strip(), f"{locale} seeded text produced audio with no transcript"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_seeded_text_to_text_needs_no_reference_voice(omni_server) -> None:
    """text -> text, the one duplex session that opens without a reference voice.

    ``ref_audio`` is required only when the session asks for audio output, so a
    ``modalities: ["text"]`` session is the duplex equivalent of the deleted
    turn-based ``text -> text`` case. The model is model-native and still
    speaks its answer, so this asserts the text side and the absence of the
    ``ref_audio_required`` rejection, not the absence of audio.
    """
    result = asyncio.run(
        _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=None,
            text="What is the capital of France? Answer in one short sentence.",
            modalities=("text",),
        )
    )

    assert "response.done" in result["event_types"], result["event_types"]
    produced_text = str(result["output_text"]) or str(result["transcript"])
    assert produced_text.strip(), f"text-only session produced nothing: {result['event_types']}"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_seeded_text_to_long_audio_output(omni_server) -> None:
    """A longer answer, so Code2Wav runs across many frames rather than one.

    Replaces the deleted ``text_to_audio_long_output`` case. The assertion is
    on the amount of audio rather than its content: the point is that a long
    answer streams to completion, not that the model says any given word.
    """
    result = asyncio.run(
        _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            text="What is the capital of China? Answer in about 40 words.",
            silence_seconds=30.0,
            timeout_s=240.0,
        )
    )

    assert "response.done" in result["event_types"], result["event_types"]
    # 24 kHz mono pcm16: 2 s of speech is 96000 bytes, comfortably more than a
    # single Code2Wav frame and well under a 40-word answer.
    assert int(result["audio_bytes"]) > 96_000, f"expected a long answer, got {result['audio_bytes']} bytes"
    assert str(result["transcript"]).strip()


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_seeded_long_form_generation(omni_server) -> None:
    """Long-form Chinese generation, replacing the deleted long-form case."""
    result = asyncio.run(
        _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            text="帮我讲一个100字的故事。",
            silence_seconds=30.0,
            timeout_s=240.0,
        )
    )

    assert "response.done" in result["event_types"], result["event_types"]
    assert int(result["audio_bytes"]) > 96_000, f"expected long-form audio, got {result['audio_bytes']} bytes"
    assert str(result["transcript"]).strip()


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_sequential_sessions_are_independent(omni_server) -> None:
    """A second session must answer its own prompt, not replay the first one.

    Replaces the deleted ``sequential_requests_independent`` case. Sessions are
    engine-resident now, so this is the check that one session's context does
    not leak into the next one on the same stages.
    """
    first = asyncio.run(
        _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            text="What is the capital of France? Answer in one short sentence.",
        )
    )
    second = asyncio.run(
        _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            text="What is the capital of China? Answer in one short sentence.",
        )
    )

    for result in (first, second):
        assert "response.done" in result["event_types"], result["event_types"]
        assert int(result["audio_bytes"]) > 0
        assert str(result["transcript"]).strip()

    # Different prompts must not produce the same answer: that would mean the
    # second session inherited the first one's context.
    assert str(first["transcript"]).strip() != str(second["transcript"]).strip()


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_text_only_response_create_is_rejected(omni_server) -> None:
    """A duplex session generates from audio: a text-only prompt is refused, not ignored.

    Every session on this route is model-native, so the chat-completion
    fallback that used to answer ``conversation.item.create`` +
    ``response.create`` with synthesized speech is gone. What a text-prompt
    client gets instead is this typed rejection, which is the contract the
    Seed-TTS Realtime backend has to be ported to.
    """
    outcome = asyncio.run(
        _run_text_only_response_create(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            text="Please say: the quick brown fox jumps over the lazy dog.",
        )
    )

    assert outcome.get("type") == "error", outcome
    error = outcome.get("error")
    assert isinstance(error, dict), outcome
    assert error.get("code") == "response_create_without_input", outcome
    assert error.get("type") == "invalid_request_error", outcome


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_still_image_input(omni_server, tmp_path: Path) -> None:
    """A spoken turn with one still image attached.

    The still-image case is a one-element frame list, which the client repeats
    on every unit of the turn: unlike the camera track it never advances, so
    this covers the image path the turn-based ``image -> text + audio`` request
    used to cover.
    """
    args = demo_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "still_image",
    )
    args.turns = 2
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    args.video_frames_b64 = duplex_camera_frames(seconds=1, cache_dir=tmp_path / "still")[:1]

    result = asyncio.run(run_demo(args))

    assert result["ok"] is True
    assert result["video_frame_count"] == 1
    assert result["video_stacked_frame_count"] == 0
    _assert_positive_int(result["audio_delta_count"])
    assert result["done_count"] == 2
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True
    _assert_request_metrics(result["request_metrics"], expected_count=2)
    _assert_session_metrics(result["session_metrics"], expected_count=2)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_two_sessions_resume_and_takeover(omni_server, tmp_path: Path) -> None:
    result = asyncio.run(
        run_multi_session(
            multi_session_args(
                omni_server=omni_server,
                input_wav=validated_input_wav(),
                ref_audio=resolve_ref_audio(),
                output_dir=tmp_path / "multi_session",
                response_required=True,
            )
        )
    )
    assert result["ok"] is True
    assert result["session_count"] == 2
    assert result["resume"]["ok"] is True
    assert result["takeover"]["ok"] is True
    assert not result["failures"]
    assert all(session["audio_delta_count"] > 0 for session in result["sessions"])
    assert all(session["done_count"] == 1 for session in result["sessions"])
    assert all(session["error_count"] == 0 for session in result["sessions"])
    for session in result["sessions"]:
        _assert_request_metrics(session["request_metrics"], expected_count=1)
        _assert_session_metrics(session["session_metrics"], expected_count=1)
