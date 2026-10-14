# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end streaming serving smoke test for LingBot-World v2.

One ``WS /v1/realtime/video`` session must drive the whole rollout and return
one video chunk per AR block. This is the entry point the recipe documents;
``tests/e2e/offline_inference/test_lingbot_world_v2_stepwise.py`` covers the
same step-execution contract at the latent level, below the transport.
"""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get(
    "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH",
    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
)
# The pipeline resizes the first frame to the requested geometry, so any real
# photograph conditions the rollout; the shared vLLM asset keeps the test
# runnable with no environment setup and adds no binary to this repository.
_IMAGE_ASSET = "2560px-Gfp-wisconsin-madison-the-nature-boardwalk"
_IMAGE_PATH_OVERRIDE = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH")

# The AR cache geometry is fixed at load time, so a request must ask for
# exactly the resolution the deploy config declares.
_HEIGHT = 480
_WIDTH = 832
_FRAMES_PER_BLOCK = 3
_TEMPORAL_COMPRESSION = 4
# One three-frame action list per generated chunk.
_CAMERA_ACTION_SCRIPT = [
    [["w"], ["w"], ["w"]],
    [["a"], [], []],
    [[], [], []],
]
_NUM_CHUNKS = len(_CAMERA_ACTION_SCRIPT)
_NUM_FRAMES = (_NUM_CHUNKS * _FRAMES_PER_BLOCK - 1) * _TEMPORAL_COMPRESSION + 1
_PROMPT = "The camera moves slowly forward through the scene."

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
]


def _first_frame() -> Path:
    if _IMAGE_PATH_OVERRIDE:
        return Path(_IMAGE_PATH_OVERRIDE).expanduser().resolve()
    return Path(ImageAsset(_IMAGE_ASSET).get_path("jpg"))


lingbot_world_v2_stepwise_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("lingbot_world_v2_stepwise.yaml"),
        ),
        id="lingbot_world_v2_stepwise",
    )
]


def _session_start(model: str, image: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    media_type = mimetypes.guess_type(image.name)[0] or "image/png"
    encoded = base64.b64encode(image.read_bytes()).decode()
    payload = {
        "type": "session.start",
        "model": model,
        "prompt": _PROMPT,
        # LingBot is image-conditioned: without a first frame the request is
        # rejected during pre-processing.
        "image_reference": {"image_url": f"data:{media_type};base64,{encoded}"},
        "width": _WIDTH,
        "height": _HEIGHT,
        "num_frames": _NUM_FRAMES,
        "fps": 16,
        "seed": 42,
        "extra_params": {"flow_shift": 5.0, "camera_action_script": _CAMERA_ACTION_SCRIPT},
    }
    payload.update(overrides or {})
    return payload


async def _stream_session(url: str, model: str, image: Path) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    import websockets

    events: list[dict[str, Any]] = []
    media = bytearray()
    done: dict[str, Any] = {}
    async with websockets.connect(url, max_size=None, ping_interval=None) as websocket:
        await websocket.send(json.dumps(_session_start(model, image)))
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=900)
            if isinstance(message, bytes):
                media.extend(message)
                continue
            event = json.loads(message)
            events.append(event)
            if event["type"] == "error":
                pytest.fail(f"server returned an error event: {event}")
            if event["type"] == "session.done":
                done = event
                break
    return events, bytes(media), done


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", lingbot_world_v2_stepwise_server_params, indirect=True)
def test_lingbot_world_v2_stepwise_streams_one_chunk_per_block(omni_server) -> None:
    """Serve the real checkpoint and stream N chunks from a single WS session."""

    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/video"
    events, media, done = asyncio.run(_stream_session(url, omni_server.model, _first_frame()))

    started = [event for event in events if event["type"] == "video.start"]
    assert len(started) == 1, "one session.start must open exactly one rollout"

    # A trailer chunk may follow the media chunks; only media chunks carry a
    # generation index, and there is exactly one per AR block.
    media_chunks = [event for event in events if event["type"] == "video.chunk_metadata" and event["kind"] == "media"]
    assert [event["generation_chunk_index"] for event in media_chunks] == list(range(_NUM_CHUNKS))
    for event in media_chunks:
        assert event["request_id"] == started[0]["request_id"]
        assert event["num_frames"] > 0
        assert event["byte_length"] > 0

    assert len(media) == sum(event["byte_length"] for event in events if event["type"] == "video.chunk_metadata")
    assert done["stopped"] is False
    assert done["chunks"] >= _NUM_CHUNKS


# Paced-playback scenarios. The first chunk decodes 13 media frames (0.8125 s
# at 16 fps), so with a 1.5 s window the admission of the second chunk is
# gated as soon as the first chunk lands and the client has not reported any
# playback: the gate engages deterministically regardless of GPU speed.
_PACED_MAX_LEAD_MS = 1500
_PACED_HOLD_S = 5.0
_PACED_NUM_CHUNKS = 6
_PACED_ACTIONS = [[["w"], ["w"], ["w"]]] * _PACED_NUM_CHUNKS
_PACED_NUM_FRAMES = (_PACED_NUM_CHUNKS * _FRAMES_PER_BLOCK - 1) * _TEMPORAL_COMPRESSION + 1


def _paced_overrides() -> dict[str, Any]:
    return {
        "num_frames": _PACED_NUM_FRAMES,
        "streaming_pacing": {"max_lead_ms": _PACED_MAX_LEAD_MS},
        "extra_params": {"flow_shift": 5.0, "camera_action_script": _PACED_ACTIONS},
    }


async def _paced_bounds_session(url: str, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hold the playhead through the first chunk, then keep it caught up."""
    import websockets

    events: list[dict[str, Any]] = []
    media = bytearray()
    received_seconds = 0.0
    caught_up = False
    async with websockets.connect(url, max_size=None, ping_interval=None) as websocket:
        await websocket.send(json.dumps(_session_start(model, _first_frame(), _paced_overrides())))
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=900)
            if isinstance(message, bytes):
                media.extend(message)
                continue
            event = json.loads(message)
            events.append(event)
            if event["type"] == "error":
                pytest.fail(f"server returned an error event: {event}")
            if event["type"] == "video.chunk_metadata" and event["kind"] == "media":
                received_seconds += event["num_frames"] / 16.0
                if not caught_up:
                    # Let the engine hold the gated chunk for a while, then
                    # report the playhead as fully caught up and keep it
                    # there for the rest of the session.
                    await asyncio.sleep(_PACED_HOLD_S)
                    caught_up = True
                await websocket.send(json.dumps({"type": "session.playback", "position_seconds": received_seconds}))
            if event["type"] == "session.done":
                return events, event
    raise AssertionError("unreachable")


async def _paced_stop_session(url: str, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Never report playback; stop while the engine holds the second chunk."""
    import websockets

    events: list[dict[str, Any]] = []
    media = bytearray()
    received_seconds = 0.0
    stop_sent = False
    async with websockets.connect(url, max_size=None, ping_interval=None) as websocket:
        await websocket.send(json.dumps(_session_start(model, _first_frame(), _paced_overrides())))
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=900)
            if isinstance(message, bytes):
                media.extend(message)
                continue
            event = json.loads(message)
            events.append(event)
            if event["type"] == "error":
                pytest.fail(f"server returned an error event: {event}")
            if event["type"] == "video.chunk_metadata" and event["kind"] == "media":
                received_seconds += event["num_frames"] / 16.0
                if not stop_sent and received_seconds > 0:
                    # The engine is now gating the second chunk on a playhead
                    # this client never reports; stop mid-wait instead.
                    await websocket.send(json.dumps({"type": "session.stop"}))
                    stop_sent = True
            if event["type"] == "session.done":
                assert stop_sent, "session finished before the paced wait engaged"
                return events, event
    raise AssertionError("unreachable")


def _assert_paced_bounds_contract(events: list[dict[str, Any]], done: dict[str, Any]) -> None:
    media_chunks = [event for event in events if event["type"] == "video.chunk_metadata" and event["kind"] == "media"]
    assert [event["output_chunk_index"] for event in media_chunks] == list(range(_PACED_NUM_CHUNKS))
    for event in media_chunks:
        pacing = event["pacing"]
        assert pacing["lead_seconds"] <= _PACED_MAX_LEAD_MS / 1000.0 + 1e-6
        assert pacing["retained_bytes"] > 0
    # The held playhead must have blocked the second chunk's admission.
    assert max(event["pacing"]["wait_seconds"] for event in media_chunks) >= _PACED_HOLD_S / 2
    assert done["stopped"] is False
    assert done["chunks"] >= _PACED_NUM_CHUNKS


def _assert_paced_stop_contract(events: list[dict[str, Any]], done: dict[str, Any], url: str, model: str) -> None:
    media_chunks = [event for event in events if event["type"] == "video.chunk_metadata" and event["kind"] == "media"]
    assert 0 < len(media_chunks) < _PACED_NUM_CHUNKS
    assert done["stopped"] is True
    # Stop must not wait for the gated chunk or the idle timeout; the server
    # reports the cancellation response latency on the stop terminal.
    assert 0 <= done["stop_latency_s"] < 30.0
    # An aborted paced session must not leave the engine paused for the next one.
    followup_events, _media, followup_done = asyncio.run(_stream_session(url, model, _first_frame()))
    assert followup_done["stopped"] is False
    assert any(event["type"] == "video.chunk_metadata" for event in followup_events)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", lingbot_world_v2_stepwise_server_params, indirect=True)
def test_lingbot_world_v2_stepwise_paced_playback_bounds_lead(omni_server) -> None:
    """Paced playback keeps the emitted lead within the window and gates chunks."""

    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/video"
    events, done = asyncio.run(_paced_bounds_session(url, omni_server.model))
    _assert_paced_bounds_contract(events, done)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", lingbot_world_v2_stepwise_server_params, indirect=True)
def test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session(omni_server) -> None:
    """Stop during a backpressure wait aborts promptly and leaves the server usable."""

    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/video"
    events, done = asyncio.run(_paced_stop_session(url, omni_server.model))
    _assert_paced_stop_contract(events, done, url, omni_server.model)


lingbot_world_v2_stepwise_tp2_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("lingbot_world_v2_stepwise_tp2.yaml"),
        ),
        id="lingbot_world_v2_stepwise_tp2",
    )
]

lingbot_world_v2_stepwise_sp2_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("lingbot_world_v2_stepwise_sp2.yaml"),
        ),
        id="lingbot_world_v2_stepwise_sp2",
    )
]


def _run_paced_contract(omni_server) -> None:
    """Admission gating and mid-wait cancellation on the served topology."""
    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/video"
    _assert_paced_bounds_contract(*asyncio.run(_paced_bounds_session(url, omni_server.model)))
    _assert_paced_stop_contract(*asyncio.run(_paced_stop_session(url, omni_server.model)), url, omni_server.model)


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", lingbot_world_v2_stepwise_tp2_server_params, indirect=True)
def test_lingbot_world_v2_stepwise_paced_contract_tp2(omni_server) -> None:
    """The paced admission and cancellation contract holds across a TP=2 rank pair."""

    _run_paced_contract(omni_server)


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", lingbot_world_v2_stepwise_sp2_server_params, indirect=True)
def test_lingbot_world_v2_stepwise_paced_contract_sp2(omni_server) -> None:
    """The paced admission and cancellation contract holds across an SP=2 rank pair."""

    _run_paced_contract(omni_server)
