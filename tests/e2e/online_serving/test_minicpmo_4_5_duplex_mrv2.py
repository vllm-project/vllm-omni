# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight duplex concurrency guard for CUDA and ROCm MRv2 defaults."""

import asyncio
import base64
import json
from contextlib import suppress
from io import BytesIO
from pathlib import Path

import pytest
import regex as re
import soundfile as sf
from PIL import Image, ImageDraw

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    ASSET_DIR,
    MODEL,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import run_multi_session
from tests.e2e.online_serving.test_minicpmo_4_5_duplex import _run_seeded_text_to_audio
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.clients.duplex import DuplexClient, EventCollector, wait_for_condition
from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

pytestmark = [pytest.mark.omni, pytest.mark.advanced_model]

_SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("minicpmo_4_5.yaml"),
    use_stage_cli=False,
    server_args=["--trust-remote-code"],
)
_THINKER_ONLY = _SERVER._replace(stage_config_path=get_deploy_config_path("minicpmo_4_5_duplex_thinker_only.yaml"))


def _color_frame(color: str, turn: int, unit: int) -> str:
    image = Image.new("RGB", (448, 448), "#efefef")
    offset = (unit % 5 - 2) * 5
    box = (67 + offset, 67, 381 + offset, 381)
    draw = ImageDraw.Draw(image)
    draw_shape = draw.ellipse if turn <= 2 else draw.rectangle
    draw_shape(box, fill=color)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=94)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    ("omni_server", "expect_audio"),
    [pytest.param(_SERVER, True, id="three-stage"), pytest.param(_THINKER_ONLY, False, id="thinker-only")],
    indirect=["omni_server"],
    scope="module",
)
def test_mrv2_duplex_barge_in_preserves_history(omni_server, expect_audio: bool):
    audio_delta = "response.output_audio.delta"
    text_deltas = {"response.output_text.delta", "response.output_audio_transcript.delta"}
    pcm_second = 32_000
    colors = ("#147bec", "#e21f36", "#15953e", "#ffcf00")
    config = create_duplex_session_config(
        modalities=None,
        temperature=0.0,
        turn_detection={
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": False,
            "interrupt_response": True,
        },
        idle_timeout_s=115,
        overlap_policy="barge_in_on_speech",
        instructions="Listen to the user and answer in English. Use only the requested words.",
        extra_body={"auto_response": True, "silence_continuation": True},
    )

    async def run():
        collector = EventCollector()
        event_index = 0
        audio_end_ms = 0
        async with DuplexClient(
            realtime_url(omni_server), model=omni_server.model, config=config, heartbeat_interval_s=None, reconnect=None
        ) as client:

            async def wait_event(*types, response_id=None, label):
                match = None

                def find_event():
                    nonlocal event_index, match
                    while event_index < len(collector.events):
                        event = collector.events[event_index]
                        event_index += 1
                        assert event["type"] != "error", event
                        if event["type"] in types and (
                            response_id is None or collector.response_id(event) == response_id
                        ):
                            match = event
                            return True
                    return False

                await wait_for_condition(find_event, timeout_s=75, label=label)
                return match

            async def feed(turn: int):
                nonlocal audio_end_ms
                samples, sample_rate = sf.read(ASSET_DIR / "barge_in" / f"turn{turn}.flac", dtype="int16")
                assert sample_rate == 16_000 and samples.ndim == 1
                lead_in = bytes(pcm_second // 10)
                # Give the native turn policy a full quiet unit before commit.
                trailing_silence = bytes(pcm_second * 6 // 5)
                pcm = lead_in + samples.astype("<i2", copy=False).tobytes() + trailing_silence
                if len(pcm) % pcm_second == 0:
                    pcm += lead_in
                clock = asyncio.get_running_loop()
                started = clock.time()
                for unit, offset in enumerate(range(0, len(pcm), pcm_second)):
                    chunk = pcm[offset : offset + pcm_second]
                    duration_ms = len(chunk) // 32
                    audio_end_ms += duration_ms
                    has_more = offset + pcm_second < len(pcm)
                    # Model listen bias leaves server VAD driven by the PCM.
                    await client.send(
                        {
                            "type": "input_audio_buffer.append",
                            "format": "pcm16",
                            "sample_rate_hz": sample_rate,
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "audio_end_ms": audio_end_ms,
                            "duration_ms": duration_ms,
                            "video_frames": [_color_frame(colors[turn - 1], turn, unit)],
                            **({"force_listen": True} if has_more else {}),
                        }
                    )
                    if has_more:
                        await asyncio.sleep(max(0, started + unit + 1 - clock.time()))

            async def finish_turn(turn, expected_text):
                await client.commit(create_response=True)
                done = await wait_event("response.done", label=f"T{turn} completion")
                response = done["response"]
                assert response["status"] == "completed", response
                response_id = response["id"]
                text = "".join(
                    part.get("text") or part.get("transcript") or ""
                    for item in response["output"]
                    for part in item["content"]
                )
                assert re.findall(r"\w+", text.lower()) == expected_text, f"T{turn}: {text!r}, {response['output']}"
                assert bool(collector.audio_bytes(response_id)) == expect_audio, f"T{turn}: {response_id}"
                if expect_audio:
                    sent_ms = response["metadata"]["playback"]["sent_ms"]
                    assert sent_ms > 0, response
                    item_id = response["output"][0]["id"]
                    await client.ack_playback(sent_ms, response_id=response_id, item_id=item_id)
                    ack = await wait_event("playback.acknowledged", label="playback ACK")
                    assert ack["event"]["item_id"] == item_id and ack["event"]["history_committed"], ack
                return response

            reader = asyncio.create_task(collector.consume(client))
            try:
                await asyncio.sleep(0)  # Subscribe before the first input.
                await feed(1)
                first = await finish_turn(1, ["blue"])
                await asyncio.sleep(0.25)

                await feed(2)
                await client.commit(create_response=True)
                output_types = (audio_delta,) if expect_audio else tuple(text_deltas)
                old = await wait_event(*output_types, label="T2 counting output")
                old_id = collector.response_id(old)
                assert old_id and old["delta"]

                await feed(3)
                # Observe cancellation before committing T3.
                await wait_event("input_audio_buffer.speech_started", label="T3 server-VAD onset")
                decision = await wait_event("overlap.decision", label="T3 barge-in decision")
                assert decision["action"] == "barge_in", decision
                assert decision["reason"] == "server_vad_speech_started", decision
                cancelled = await wait_event("response.done", response_id=old_id, label="T2 cancellation")
                assert cancelled["response"]["status"] == "cancelled", cancelled
                assert cancelled["response"]["status_details"]["reason"] == "turn_detected", cancelled

                await finish_turn(3, ["green"])
                await asyncio.sleep(0.25)
                await feed(4)
                await finish_turn(4, ["blue", "yellow"])

                # The retrieval is also a barrier covering frames received
                # after the last answer, including any late old-response deltas.
                first_item = first["output"][0]["id"]
                await client.send({"type": "conversation.item.retrieve", "item_id": first_item})
                retrieved = await wait_event("conversation.item.retrieved", label="first conversation history item")
                assert retrieved["item"]["id"] == first_item and retrieved["item"]["status"] == "completed", retrieved
                old_events = [event["type"] for event in collector.events if collector.response_id(event) == old_id]
                tail = old_events[old_events.index("response.done") + 1 :]
                assert not set(tail) & (text_deltas | {audio_delta, "response.done"}), tail
            finally:
                reader.cancel()
                with suppress(asyncio.CancelledError):
                    await reader

    asyncio.run(run())


# ``minicpmo_4_5.yaml`` resolves all three stages to MRv2 on CUDA and ROCm.
# The AMD READY lane executes the resolver contract before selecting these
# real-weight cases.


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
@pytest.mark.parametrize("sessions", [1, 2, 4])
def test_mrv2_duplex_overlapping_turns(omni_server, tmp_path: Path, sessions: int):
    args = multi_session_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / f"concurrency_{sessions}",
        response_required=True,
    )
    args.sessions = sessions
    args.turns = 2
    # Keep the complete response-required utterance. The 1.4 s active slice
    # can end mid-utterance, so the native policy may legitimately keep listening.
    args.turn_duration_ms = [0] * args.turns
    # Native model turns need not map one-to-one to physical input commits.
    # Still require a completed text/audio answer for each input, and drain
    # and acknowledge any additional model responses before the next input.
    args.allow_extra_responses = True
    args.synchronized_start = True
    args.disconnect_session_index = None
    args.takeover_session_index = None
    result = asyncio.run(run_multi_session(args))
    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
    assert result["session_count"] == sessions
    assert result["identity_isolation_ok"] is True
    assert not result["failures"]
    session_results = result["sessions"]
    assert isinstance(session_results, list), result
    for session in session_results:
        assert session["audio_delta_count"] > 0, session
        assert session["done_count"] >= args.turns, session
        assert session["turn_outcomes"] == ["speak"] * args.turns, session
        assert session["error_count"] == 0, session


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
def test_mrv2_duplex_new_session_during_long_response(omni_server):
    # Unequal prompts and a delayed second session prevent identical clients
    # from advancing in lockstep and hiding shared condition-state bugs.
    ref_audio = resolve_ref_audio()

    async def speak(text, delay):
        await asyncio.sleep(delay)
        return await _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=ref_audio,
            text=text,
            silence_seconds=30.0,
        )

    async def overlap():
        return await asyncio.gather(
            speak("What is the capital of China? Answer in about 40 words.", 0),
            speak("What is the capital of France? Answer in one short sentence.", 2),
        )

    first, second = asyncio.run(overlap())
    for result in (first, second):
        assert "response.done" in result["event_types"], result
        assert result["audio_bytes"] > 0, result
        assert str(result["transcript"]).strip(), result
    assert first["audio_bytes"] > 96_000, first
    assert str(first["transcript"]).strip() != str(second["transcript"]).strip()


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
def test_mrv2_duplex_server_answers_image_chat_from_the_image(omni_server, openai_client):
    """The duplex Thinker also serves /v1/chat/completions: its image features must reach the prompt."""
    buffer = BytesIO()
    Image.new("RGB", (224, 224), (255, 0, 0)).save(buffer, format="JPEG")
    image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    request_config = {
        "model": omni_server.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What color is this image? Answer with one word."},
                ],
            }
        ],
        "stream": True,
        "modalities": ["text"],
        "key_words": {"text": ["red"]},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    openai_client.send_omni_request(request_config)
