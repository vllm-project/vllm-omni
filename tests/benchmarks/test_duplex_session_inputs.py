# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import random
from typing import Any

import pytest
from PIL import Image

from vllm_omni.benchmarks import duplex_session_inputs as inputs
from vllm_omni.engine.duplex.config import DuplexCapabilities

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]

_MINICPMO = DuplexCapabilities(required_input_modalities=frozenset({"audio"}))
_AURA = DuplexCapabilities(
    required_input_modalities=frozenset({"video"}), optional_input_modalities=frozenset({"audio"})
)
_QWEN_LIKE = DuplexCapabilities(optional_input_modalities=frozenset(), supports_image_input=True)


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        ({}, inputs.FRAME_TRANSPORT_APPEND),
        ({"chunk_period_ms": 1000}, inputs.FRAME_TRANSPORT_APPEND),
        (_MINICPMO.as_dict(), inputs.FRAME_TRANSPORT_APPEND),
        (_AURA.as_dict(), inputs.FRAME_TRANSPORT_APPEND),
        (_QWEN_LIKE.as_dict(), inputs.FRAME_TRANSPORT_IMAGE_ITEMS),
    ],
)
def test_frame_transport_follows_advertised_input_modalities(capabilities, expected):
    assert inputs.select_frame_transport(capabilities) == expected


def test_frame_transport_refuses_a_session_without_any_visual_input():
    caps = DuplexCapabilities(optional_input_modalities=frozenset()).as_dict()
    with pytest.raises(ValueError, match="neither video_frames"):
        inputs.select_frame_transport(caps)


def test_session_turn_detection_modes():
    assert inputs.session_turn_detection("none") is None
    assert inputs.session_turn_detection("server_vad") == {"type": "server_vad"}
    with pytest.raises(ValueError, match="turn detection"):
        inputs.session_turn_detection("semantic_vad")


def _jpeg(side: int) -> bytes:
    buffer = io.BytesIO()
    # Smooth images compress too well to exceed the budget; noise does not.
    image = Image.frombytes("RGB", (side, side), random.Random(0).randbytes(side * side * 3))
    image.save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_image_window_keeps_the_newest_frames_within_the_server_bound():
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    window = inputs.ImageItemWindow(send, item_prefix="frame")
    await window.push([base64.b64encode(b"small").decode()] * (inputs.MAX_IMAGE_ITEMS + 2))

    creates = [event for event in sent if event["type"] == "conversation.item.create"]
    deletes = [event["item_id"] for event in sent if event["type"] == "conversation.item.delete"]
    assert [event["item"]["id"] for event in creates] == [f"frame_{index}" for index in range(10)]
    assert deletes == ["frame_0", "frame_1"]
    # Each delete precedes the create that would overflow the window.
    assert sent.index({"type": "conversation.item.delete", "item_id": "frame_0"}) < sent.index(creates[8])
    part = creates[0]["item"]["content"][0]
    assert part == {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(b"small").decode()}
    assert creates[0]["item"]["role"] == "user" and window.sent == 10


@pytest.mark.asyncio
async def test_image_window_downscales_frames_that_would_overflow_the_budget():
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    large = _jpeg(1024)
    assert len(large) * 4 // 3 > 4 * 1024 * 1024 // 9
    await inputs.ImageItemWindow(send, item_prefix="frame").push([large])
    url = sent[0]["item"]["content"][0]["image_url"]
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        assert max(image.size) == 640
    assert len(url) * inputs.MAX_IMAGE_ITEMS < 4 * 1024 * 1024


def test_server_turns_pending_tracks_open_speech_and_responses():
    events: list[dict[str, object]] = [{"type": "input_audio_buffer.speech_started"}]
    assert inputs.server_turns_pending(events)
    events.append({"type": "input_audio_buffer.speech_stopped"})
    assert not inputs.server_turns_pending(events)
    events.append({"type": "response.created", "response": {"id": "r1"}})
    assert inputs.server_turns_pending(events)
    events.append({"type": "response.done", "response_id": "r1"})
    assert not inputs.server_turns_pending(events)


def test_turn_activity_ignores_playback_acknowledgements():
    events: list[dict[str, object]] = [
        {"type": "input_audio_buffer.speech_started"},
        {"type": "playback.acknowledged"},
        {"type": "response.output_audio.delta"},
        {"type": "conversation.item.created"},
    ]
    assert inputs.turn_activity_count(events) == 2
