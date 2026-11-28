# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight duplex concurrency guard for CUDA and ROCm MRv2 defaults."""

import asyncio
import json
from pathlib import Path

import pytest
import websockets

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    MODEL,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import DemoState
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import run_multi_session
from tests.e2e.online_serving.test_minicpmo_4_5_duplex import _run_seeded_text_to_audio
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.clients.duplex import build_realtime_url

pytestmark = [pytest.mark.omni, pytest.mark.advanced_model]

_SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("minicpmo_4_5.yaml"),
    use_stage_cli=False,
    server_args=["--trust-remote-code"],
)


async def _run_interrupted_recall(
    *, url: str, model: str, cancel: str, secret: str, event_path: Path, interruptions: int = 1
):
    """Send the follow-up immediately after cancel, without playback ACKs or filler inputs."""
    events = []
    try:
        async with websockets.connect(build_realtime_url(url, model, autostart=False), max_size=None) as ws:

            async def send(event):
                await ws.send(json.dumps(event))

            async def receive_until(predicate):
                async def receive():
                    while True:
                        event = json.loads(await ws.recv())
                        events.append(event)
                        assert event.get("type") != "error", event
                        if predicate(event):
                            return event

                return await asyncio.wait_for(receive(), timeout=60)

            await send(
                {
                    "type": "session.update",
                    "session": {
                        "model": model,
                        "modalities": ["text", "audio"],
                        "temperature": 0.0,
                        "extra_body": {"auto_response": True, "force_listen_count": 0},
                        "turn_detection": None,
                        "overlap_policy": "listen_only",
                        "playback_commit_policy": "ack_only",
                    },
                }
            )
            await receive_until(lambda e: e.get("type") == "session.updated")
            await send(
                {
                    "type": "input.text.append",
                    "text": f"Remember the secret word {secret}. Do not say it unless asked.",
                    "force_listen": True,
                }
            )
            audio_types = {"response.output_audio.delta", "response.audio.delta"}
            for index in range(interruptions):
                # Exercise the reported back-to-back inputs without a LISTEN barrier.
                await send(
                    {
                        "type": "input.text.append",
                        "text": "Count slowly out loud from one to fifty in English. Say each number. Start now.",
                    }
                )
                audio = await receive_until(lambda e: e.get("type") in audio_types and e.get("delta"))
                old_id = audio["response_id"]
                assert not any(
                    e.get("type") == "response.done" and e.get("response", {}).get("id") == old_id for e in events
                ), "The counting response must still be active when interrupted"
                await send({"type": cancel, "response_id": old_id})
                if index == interruptions - 1:
                    await send(
                        {
                            "type": "input.text.append",
                            "text": "What is the secret word I asked you to remember? Say only that word.",
                        }
                    )
                cancelled = await receive_until(
                    lambda e: e.get("type") == "response.done" and e.get("response", {}).get("id") == old_id
                )
                assert cancelled["response"]["status"] == "cancelled", cancelled
                if index < interruptions - 1:
                    await asyncio.sleep(0.2)
            boundary = len(events)
            done = await receive_until(
                lambda e: e.get("type") == "response.done" and e.get("response", {}).get("id") != old_id
            )
            tail = events[boundary:]
            transcript_types = {
                "response.output_audio_transcript.delta",
                "response.audio_transcript.delta",
                "response.output_text.delta",
            }
            new_id = done["response"]["id"]
            transcript = "".join(
                e.get("delta", "") for e in tail if e.get("type") in transcript_types and e.get("response_id") == new_id
            )
            packets = [e for e in tail if e.get("type") in audio_types and e.get("delta")]
            await send({"type": "session.close"})
            await receive_until(lambda e: e.get("type") == "session.closed")
            assert DemoState(events=events).stale_audio_delta_count() == 0, "Received audio for a cancelled response"
            assert done["response"]["status"] == "completed", done
            assert packets, tail
            assert all(e["response_id"] == done["response"]["id"] for e in packets), tail
            return transcript
    finally:
        event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2))


def _recalls_own_secret(transcript: str, secret: str, session_secrets: tuple[str, ...]) -> bool:
    words = set("".join(c if c.isalnum() else " " for c in transcript.casefold()).split())
    counting_words = (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty".split()
    )
    return (
        words.intersection(session_secrets) == {secret}
        and words.isdisjoint(counting_words)
        and not any(word.isdigit() for word in words)
    )


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
@pytest.mark.parametrize(
    "cancel,sessions,interruptions",
    [
        ("response.cancel", 1, 1),
        ("output_audio_buffer.clear", 1, 1),
        ("barge_in", 1, 1),
        ("response.cancel", 2, 1),
        ("response.cancel", 1, 2),
    ],
)
def test_mrv2_duplex_immediate_recall_after_interrupt(
    omni_server, tmp_path: Path, cancel: str, sessions: int, interruptions: int
):
    secrets = ("robin", "squirrel")[:sessions]

    async def recall():
        return await asyncio.gather(
            *(
                _run_interrupted_recall(
                    url=realtime_url(omni_server),
                    model=omni_server.model,
                    cancel=cancel,
                    secret=secret,
                    event_path=tmp_path / f"{secret}.json",
                    interruptions=interruptions,
                )
                for secret in secrets
            )
        )

    for secret, transcript in zip(secrets, asyncio.run(recall()), strict=True):
        assert _recalls_own_secret(transcript, secret, secrets), (secret, transcript)


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
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
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
